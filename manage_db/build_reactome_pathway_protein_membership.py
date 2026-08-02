"""Build staged Reactome source-native pathway_contains_protein artifacts.

This module intentionally stages, rather than promotes, ``pathway_contains_protein``.
It consumes Reactome's protein-native UniProt→Reactome mapping export, keeps only
human ``R-HSA-*`` pathway memberships, maps UniProt accessions to existing KG
``protein`` nodes when the mapping is unambiguous, rejects ambiguous/unmapped
protein endpoints instead of projecting through genes, and materializes edges,
evidence, rejects, and a validation report.
"""

from __future__ import annotations

import argparse
import json
import shutil
import urllib.request
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import pyarrow.parquet as pq

from manage_db import kg_evidence
from manage_db.kg_schema import EDGE_PARQUET_COLUMNS
from manage_db.kg_storage import read_nodes, write_edges, open_kg_root

RELATION = "pathway_contains_protein"
DISPLAY_RELATION = "contains protein"
SOURCE = "Reactome"
SOURCE_DATASET = "UniProt2Reactome_All_Levels"
SOURCE_URL = "https://reactome.org/download/current/UniProt2Reactome_All_Levels.txt"
LICENSE = "Reactome; review https://reactome.org/license before canonical promotion"
EVIDENCE_TYPE = "database_record"
EXTRA_EVIDENCE_COLUMNS = [
    "membership_type",
    "source_db",
    "source_pathway_id",
    "source_pathway_url",
    "source_pathway_name",
    "source_protein_id",
    "reactome_evidence_code",
    "species",
    "mapping_confidence",
    "mapping_method",
    "release",
    "created_at",
]
REJECT_COLUMNS = [
    "source_protein_id",
    "source_pathway_id",
    "source_pathway_url",
    "source_pathway_name",
    "reactome_evidence_code",
    "species",
    "reason",
    "mapping_candidates",
]


@dataclass(slots=True)
class BuildResult:
    edges: pd.DataFrame
    evidence: pd.DataFrame
    rejected: pd.DataFrame
    validation: dict[str, Any]


def _clean(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _edge_key(relation: str, x_id: str, y_id: str) -> str:
    return f"{relation}|{x_id}|{y_id}"


def _release_from_headers(headers: Mapping[str, str]) -> str:
    last_modified = headers.get("Last-Modified") or headers.get("last-modified") or ""
    if last_modified:
        try:
            return parsedate_to_datetime(last_modified).date().isoformat()
        except Exception:
            return last_modified
    etag = (headers.get("ETag") or headers.get("etag") or "").strip('"')
    return f"etag-{etag}" if etag else date.today().isoformat()


def cache_reactome_mapping(
    url: str = SOURCE_URL,
    raw_cache_dir: str | Path = "artifacts/cache/raw/reactome/uniprot2reactome_all_levels",
) -> tuple[Path, dict[str, Any]]:
    """Download/cache the current Reactome UniProt→pathway export."""

    raw_cache = Path(raw_cache_dir)
    raw_cache.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "hermes-agent txgnn pathway_contains_protein audit"})
    with urllib.request.urlopen(req, timeout=180) as response:
        headers = dict(response.headers.items())
        release = _release_from_headers(headers)
        release_dir = raw_cache / release
        release_dir.mkdir(parents=True, exist_ok=True)
        target = release_dir / "UniProt2Reactome_All_Levels.txt"
        if not target.exists():
            with target.open("wb") as out:
                shutil.copyfileobj(response, out)
    manifest = {
        "source": SOURCE,
        "source_dataset": SOURCE_DATASET,
        "url": url,
        "cached_path": str(target),
        "release": release,
        "headers": {k: headers.get(k, "") for k in ["Last-Modified", "ETag", "Content-Length", "Content-Type"]},
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(release_dir / "manifest.json", manifest)
    return target, manifest


def read_uniprot2reactome(path: str | Path) -> pd.DataFrame:
    """Read Reactome UniProt2Reactome export as a typed DataFrame."""

    cols = ["uniprot_id", "reactome_id", "url", "pathway_name", "evidence_code", "species"]
    return pd.read_csv(path, sep="\t", names=cols, dtype=str, keep_default_na=False)


def build_uniprot_to_protein_map(*, node_root: str | None = None, protein_nodes_path: str | Path | None = None) -> tuple[dict[str, str], dict[str, list[str]], dict[str, Any]]:
    """Return unambiguous UniProt accession → KG protein node mappings."""

    if protein_nodes_path:
        proteins = pq.read_table(protein_nodes_path, columns=["id", "uniprot_id"]).to_pandas()
        source = str(protein_nodes_path)
    elif node_root:
        root = open_kg_root(node_root)
        proteins = read_nodes(root, "protein", columns=["id", "uniprot_id"])
        source = f"{node_root.rstrip('/')}/nodes/protein.parquet"
    else:
        return {}, {}, {"mapping_supplied": False, "source": "", "uniprot_with_mapping": 0, "ambiguous_uniprot": 0}

    groups: dict[str, set[str]] = {}
    for _, row in proteins.iterrows():
        uniprot = _clean(row.get("uniprot_id"))
        protein_id = _clean(row.get("id"))
        if not uniprot or not protein_id:
            continue
        for token in [p.strip() for p in uniprot.replace(";", ",").split(",")]:
            if token:
                groups.setdefault(token, set()).add(protein_id)
    unambiguous = {u: next(iter(ids)) for u, ids in groups.items() if len(ids) == 1}
    ambiguous = {u: sorted(ids) for u, ids in groups.items() if len(ids) > 1}
    return unambiguous, ambiguous, {
        "mapping_supplied": True,
        "source": source,
        "uniprot_with_mapping": len(groups),
        "unambiguous_uniprot": len(unambiguous),
        "ambiguous_uniprot": len(ambiguous),
        "policy": "accept only UniProt accessions with exactly one nodes/protein.uniprot_id match; reject ambiguous multi-ENSP mappings rather than projecting through genes",
    }


def load_pathway_ids(*, node_root: str | None = None, pathway_nodes_path: str | Path | None = None) -> tuple[set[str], dict[str, Any]]:
    if pathway_nodes_path:
        pathways = pq.read_table(pathway_nodes_path, columns=["id"]).to_pandas()
        source = str(pathway_nodes_path)
    elif node_root:
        root = open_kg_root(node_root)
        pathways = read_nodes(root, "pathway", columns=["id"])
        source = f"{node_root.rstrip('/')}/nodes/pathway.parquet"
    else:
        return set(), {"pathway_nodes_supplied": False, "source": "", "pathway_node_count": 0}
    ids = set(pathways["id"].astype(str))
    return ids, {"pathway_nodes_supplied": True, "source": source, "pathway_node_count": len(ids)}


def build_from_reactome_mapping(
    mapping: pd.DataFrame,
    *,
    uniprot_to_protein: Mapping[str, str],
    ambiguous_uniprot: Mapping[str, list[str]],
    pathway_ids: set[str],
    release: str,
    source_url: str = SOURCE_URL,
    created_at: str | None = None,
) -> BuildResult:
    created_at = created_at or datetime.now(timezone.utc).isoformat()
    edges: dict[tuple[str, str], dict[str, Any]] = {}
    evidence_rows: list[dict[str, Any]] = []
    rejects: list[dict[str, Any]] = []
    source_counts: Counter[str] = Counter()

    for idx, row in mapping.iterrows():
        uniprot = _clean(row.get("uniprot_id"))
        pathway_id = _clean(row.get("reactome_id"))
        species = _clean(row.get("species"))
        evidence_code = _clean(row.get("evidence_code"))
        pathway_url = _clean(row.get("url"))
        pathway_name = _clean(row.get("pathway_name"))

        if species != "Homo sapiens" or not pathway_id.startswith("R-HSA-"):
            continue
        source_counts["human_r_hsa_rows"] += 1
        reject_base = {
            "source_protein_id": uniprot,
            "source_pathway_id": pathway_id,
            "source_pathway_url": pathway_url,
            "source_pathway_name": pathway_name,
            "reactome_evidence_code": evidence_code,
            "species": species,
        }
        if pathway_id not in pathway_ids:
            rejects.append({**reject_base, "reason": "missing_pathway_node", "mapping_candidates": ""})
            continue
        if uniprot in ambiguous_uniprot:
            rejects.append({**reject_base, "reason": "ambiguous_uniprot_to_protein", "mapping_candidates": ",".join(ambiguous_uniprot[uniprot])})
            continue
        protein_id = uniprot_to_protein.get(uniprot, "")
        if not protein_id:
            rejects.append({**reject_base, "reason": "unmapped_uniprot_to_protein", "mapping_candidates": ""})
            continue

        edge_pair = (pathway_id, protein_id)
        edges.setdefault(
            edge_pair,
            {
                "x_id": pathway_id,
                "x_type": "pathway",
                "y_id": protein_id,
                "y_type": "protein",
                "relation": RELATION,
                "display_relation": DISPLAY_RELATION,
                "source": SOURCE,
                "credibility": 3,
            },
        )
        source_record_id = f"Reactome:{SOURCE_DATASET}:{release}:{uniprot}:{pathway_id}:{evidence_code}:{idx}"
        text_span = json.dumps(
            {
                "endpoint_policy": "Reactome UniProt protein endpoint mapped directly to an unambiguous KG protein node; no gene-to-protein projection",
                "source_db": SOURCE,
                "source_dataset": SOURCE_DATASET,
                "source_protein_id": uniprot,
                "source_pathway_id": pathway_id,
                "source_pathway_url": pathway_url,
                "source_pathway_name": pathway_name,
                "species": species,
                "reactome_evidence_code": evidence_code,
                "membership_type": "all_levels_pathway_protein_membership",
                "protein_mapping_confidence": "exact_unambiguous_uniprot_xref",
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        edge_key = _edge_key(RELATION, pathway_id, protein_id)
        evidence_rows.append(
            {
                "edge_key": edge_key,
                "relation": RELATION,
                "x_id": pathway_id,
                "x_type": "pathway",
                "y_id": protein_id,
                "y_type": "protein",
                "evidence_type": EVIDENCE_TYPE,
                "source": SOURCE,
                "source_dataset": SOURCE_DATASET,
                "source_record_id": source_record_id,
                "paper_id": "",
                "dataset_id": "",
                "study_id": "",
                "evidence_score": pd.NA,
                "effect_size": pd.NA,
                "p_value": pd.NA,
                "direction": RELATION,
                "confidence_interval": "",
                "predicate": evidence_code or "reactome_pathway_membership",
                "text_span": text_span,
                "section": "",
                "extraction_method": "Reactome UniProt2Reactome_All_Levels source-native protein pathway mapping",
                "license": LICENSE,
                "release": release,
                "created_at": created_at,
                "membership_type": "all_levels_pathway_protein_membership",
                "source_db": SOURCE,
                "source_pathway_id": pathway_id,
                "source_pathway_url": pathway_url,
                "source_pathway_name": pathway_name,
                "source_protein_id": uniprot,
                "reactome_evidence_code": evidence_code,
                "species": species,
                "mapping_confidence": "exact_unambiguous_uniprot_xref",
                "mapping_method": "nodes/protein.uniprot_id exact single-match",
            }
        )

    edges_df = pd.DataFrame(edges.values(), columns=[name for name, _ in EDGE_PARQUET_COLUMNS])
    evidence_df = pd.DataFrame(evidence_rows)
    rejected_df = pd.DataFrame(rejects, columns=REJECT_COLUMNS)
    validation = validate_outputs(edges_df, evidence_df, rejected_df, source_counts, release=release, source_url=source_url, created_at=created_at)
    return BuildResult(edges_df, evidence_df, rejected_df, validation)


def validate_outputs(edges: pd.DataFrame, evidence: pd.DataFrame, rejected: pd.DataFrame, source_counts: Counter[str], *, release: str, source_url: str, created_at: str) -> dict[str, Any]:
    validation: dict[str, Any] = {
        "ok": True,
        "created_at": created_at,
        "staging_only": True,
        "canonical_promotion": False,
        "source": SOURCE,
        "source_dataset": SOURCE_DATASET,
        "source_url": source_url,
        "source_release": release,
        "counts": {
            "pathway_contains_protein_edges": int(len(edges)),
            "pathway_contains_protein_evidence": int(len(evidence)),
            "rejected_source_rows": int(len(rejected)),
        },
        "source_counts": dict(source_counts),
        "checks": {},
        "warnings": [],
        "canonical_promotion_recommendation": "Review staged-only Reactome all-level protein pathway memberships, especially all-level semantics and rejected ambiguous UniProt mappings, before canonical promotion.",
    }
    checks = validation["checks"]
    duplicate_edges = int(edges.duplicated(subset=["relation", "x_id", "y_id"]).sum()) if not edges.empty else 0
    edge_keys = set(edges.apply(lambda r: _edge_key(str(r["relation"]), str(r["x_id"]), str(r["y_id"])), axis=1)) if not edges.empty else set()
    evidence_keys = set(evidence["edge_key"].astype(str)) if not evidence.empty else set()
    unsupported = edge_keys - evidence_keys
    orphan = evidence_keys - edge_keys
    checks["edges_unique"] = {"ok": duplicate_edges == 0, "duplicate_rows": duplicate_edges}
    checks["edge_evidence_support"] = {
        "ok": not unsupported and not orphan,
        "edges_without_evidence_count": len(unsupported),
        "evidence_without_edge_count": len(orphan),
        "edges_without_evidence": sorted(unsupported)[:20],
        "evidence_without_edge": sorted(orphan)[:20],
    }
    checks["endpoint_types"] = {
        "ok": (edges.empty or (set(edges["x_type"]) == {"pathway"} and set(edges["y_type"]) == {"protein"}))
        and (evidence.empty or (set(evidence["x_type"]) == {"pathway"} and set(evidence["y_type"]) == {"protein"})),
        "edge_x_types": sorted(set(edges["x_type"])) if not edges.empty else [],
        "edge_y_types": sorted(set(edges["y_type"])) if not edges.empty else [],
    }
    checks["source_native_policy"] = {
        "ok": True,
        "note": "Rows come from Reactome UniProt2Reactome protein endpoints and are never generated from pathway_contains_gene or gene IDs.",
    }
    checks["mapping_policy"] = {
        "ok": True,
        "policy": "Only unambiguous UniProt→KG protein node mappings are accepted; ambiguous/unmapped accessions and missing pathway nodes are rejected.",
        "rejected_reason_counts": rejected["reason"].value_counts().to_dict() if not rejected.empty else {},
    }
    validation["ok"] = all(check.get("ok", False) for check in checks.values())
    if edges.empty:
        validation["warnings"].append("No edges staged; check protein/pathway node inputs and source filters.")
        validation["ok"] = False
    return validation


def _write_parquet(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.reset_index(drop=True).to_parquet(path, index=False)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def build_staged_reactome_pathway_proteins(
    *,
    input_path: str | Path | None = None,
    node_root: str | None = None,
    protein_nodes_path: str | Path | None = None,
    pathway_nodes_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    raw_cache_dir: str | Path = "artifacts/cache/raw/reactome/uniprot2reactome_all_levels",
    source_url: str = SOURCE_URL,
) -> dict[str, Any]:
    if input_path is None:
        input_file, manifest = cache_reactome_mapping(source_url, raw_cache_dir)
    else:
        input_file = Path(input_path)
        manifest = {"source": SOURCE, "source_dataset": SOURCE_DATASET, "url": source_url, "cached_path": str(input_file), "release": input_file.parent.name or "manual"}
    release = str(manifest.get("release") or "")
    out_dir = Path(output_dir) if output_dir else Path("artifacts/staged") / f"reactome-pathway-contains-protein-{date.today().isoformat()}"
    mapping = read_uniprot2reactome(input_file)
    uniprot_to_protein, ambiguous_uniprot, protein_mapping_stats = build_uniprot_to_protein_map(node_root=node_root, protein_nodes_path=protein_nodes_path)
    pathway_ids, pathway_stats = load_pathway_ids(node_root=node_root, pathway_nodes_path=pathway_nodes_path)
    result = build_from_reactome_mapping(
        mapping,
        uniprot_to_protein=uniprot_to_protein,
        ambiguous_uniprot=ambiguous_uniprot,
        pathway_ids=pathway_ids,
        release=release,
        source_url=source_url,
    )

    _write_parquet(out_dir / "edges" / f"{RELATION}.parquet", result.edges)
    _write_parquet(out_dir / "evidence" / f"{RELATION}.parquet", result.evidence)
    _write_parquet(out_dir / "mappings" / "reactome_pathway_contains_protein_rejected.parquet", result.rejected)
    # Also write a canonical-schema evidence copy for audit_edge_evidence compatibility.
    canonical_evidence = kg_evidence._coerce_evidence_frame(result.evidence, RELATION)
    _write_parquet(out_dir / "evidence_canonical" / f"{RELATION}.parquet", canonical_evidence)

    report = {
        **result.validation,
        "output_dir": str(out_dir),
        "inputs": {"uniprot2reactome": str(input_file), "node_root": node_root or "", "protein_nodes": str(protein_nodes_path or ""), "pathway_nodes": str(pathway_nodes_path or ""), "manifest": manifest},
        "protein_mapping_stats": protein_mapping_stats,
        "pathway_stats": pathway_stats,
        "artifacts": {
            "edges": str(out_dir / "edges" / f"{RELATION}.parquet"),
            "evidence": str(out_dir / "evidence" / f"{RELATION}.parquet"),
            "canonical_evidence": str(out_dir / "evidence_canonical" / f"{RELATION}.parquet"),
            "rejected": str(out_dir / "mappings" / "reactome_pathway_contains_protein_rejected.parquet"),
        },
    }
    _write_json(out_dir / "validation" / "reactome_pathway_contains_protein_report.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=None, help="Optional cached UniProt2Reactome_All_Levels.txt; downloads current release if omitted")
    parser.add_argument("--raw-cache-dir", default="artifacts/cache/raw/reactome/uniprot2reactome_all_levels", help="Repo-local raw cache for release-pinned Reactome mapping")
    parser.add_argument("--node-root", default="", help="KG root containing nodes/protein.parquet and nodes/pathway.parquet, e.g. gs://jouvencekb/main")
    parser.add_argument("--protein-nodes", default=None, help="Optional local protein.parquet for tests/offline builds")
    parser.add_argument("--pathway-nodes", default=None, help="Optional local pathway.parquet for tests/offline builds")
    parser.add_argument("--output-dir", default=None, help="Defaults to artifacts/staged/reactome-pathway-contains-protein-YYYY-MM-DD")
    parser.add_argument("--source-url", default=SOURCE_URL)
    args = parser.parse_args(argv)
    report = build_staged_reactome_pathway_proteins(
        input_path=args.input,
        node_root=args.node_root or None,
        protein_nodes_path=args.protein_nodes,
        pathway_nodes_path=args.pathway_nodes,
        output_dir=args.output_dir,
        raw_cache_dir=args.raw_cache_dir,
        source_url=args.source_url,
    )
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0 if report.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
