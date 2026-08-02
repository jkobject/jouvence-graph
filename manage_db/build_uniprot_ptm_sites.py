"""Build staged UniProt PTM site nodes and protein_has_ptm_site edges.

This module is intentionally staging-only.  It downloads release-pinned
UniProtKB reviewed human entries carrying PTM features, maps UniProt accessions
to existing ENSP protein nodes, and emits:

- nodes/ptm_site.parquet
- edges/protein_has_ptm_site.parquet
- evidence/protein_has_ptm_site.parquet
- diagnostics/ptm_site_disease_link_candidates.parquet
- reports/uniprot_ptm_sites_summary.json

Disease/phenotype relations are not inferred from generic protein disease
comments.  The diagnostics table records only feature-local disease/phenotype
candidates; by default no disease/phenotype edge parquet is written unless such
explicit site-level support exists and a downstream schema gate approves it.
"""

from __future__ import annotations

import argparse
import http.client
import json
import time
import re
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import fsspec
import pandas as pd

SOURCE = "UniProtKB"
SOURCE_DATASET = "reviewed_human_ptm_features"
RELATION = "protein_has_ptm_site"
DISPLAY_RELATION = "has PTM site"
PTM_FEATURE_TYPES = {
    "Modified residue",
    "Lipidation",
    "Glycosylation",
    "Disulfide bond",
}
CANONICAL_ROOTS = {
    "/mnt/gcs/jouvencekb/main",
    "gs://jouvencekb/main",
}
PTM_SITE_COLUMNS = [
    "id",
    "uniprot_accession",
    "protein_id",
    "isoform",
    "feature_type",
    "modification_type",
    "psi_mod_id",
    "residue",
    "position",
    "end_position",
    "location_modifier",
    "sequence_context",
    "feature_id",
    "description",
    "evidence_codes",
    "pmids",
    "source",
    "source_release",
]
EDGE_COLUMNS = [
    "x_id",
    "x_type",
    "y_id",
    "y_type",
    "relation",
    "display_relation",
    "source",
    "credibility",
]
EVIDENCE_COLUMNS = [
    "edge_key",
    "relation",
    "x_id",
    "x_type",
    "y_id",
    "y_type",
    "evidence_type",
    "source",
    "source_dataset",
    "source_record_id",
    "paper_id",
    "uniprot_accession",
    "ensp_id",
    "isoform",
    "residue",
    "position",
    "end_position",
    "modification_type",
    "psi_mod_id",
    "eco_codes",
    "predicate",
    "sequence_context",
    "release",
    "created_at",
]
DISEASE_LINK_COLUMNS = [
    "ptm_site_id",
    "uniprot_accession",
    "protein_id",
    "feature_type",
    "residue",
    "position",
    "description",
    "candidate_text",
    "candidate_kind",
    "disease_or_phenotype_id",
    "disease_or_phenotype_name",
    "pmids",
    "source_record_id",
    "decision",
]

PSI_MOD_HINTS = [
    ("phospho", "MOD:00696"),
    ("acetyl", "MOD:00394"),
    ("methyl", "MOD:00427"),
    ("ubiquitin", "MOD:01148"),
    ("sumoyl", "MOD:01149"),
    ("glycosyl", "MOD:00693"),
    ("n-linked", "MOD:00693"),
    ("o-linked", "MOD:00693"),
    ("palmitoyl", "MOD:00115"),
    ("myristoyl", "MOD:00114"),
    ("farnesyl", "MOD:00435"),
    ("geranylgeranyl", "MOD:00436"),
    ("disulfide", "MOD:00689"),
]
RESIDUE_HINTS = {
    "serine": "S",
    "threonine": "T",
    "tyrosine": "Y",
    "lysine": "K",
    "arginine": "R",
    "cysteine": "C",
    "asparagine": "N",
    "glutamine": "Q",
    "histidine": "H",
    "aspartate": "D",
    "glutamate": "E",
}
DISEASE_TOKEN_RE = re.compile(
    r"\b(disease|syndrome|cancer|carcinoma|tumou?r|phenotype|pathogenic|pathology|defect|deficiency|mim\b|omim\b|efo[:_]|hp[:_]|mondo[:_])",
    re.IGNORECASE,
)
ID_TOKEN_RE = re.compile(r"\b((?:EFO|HP|MONDO)[:_][0-9]{7}|(?:MIM|OMIM)[: ]?[0-9]{6})\b", re.IGNORECASE)


def _clean(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _slug(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "_", value.strip()).strip("_").lower()
    return text or "ptm"


def _pipe(values: Iterable[str]) -> str:
    return "|".join(sorted({v for v in values if v}))


def _feature_position(feature: Mapping[str, Any]) -> tuple[int | None, int | None, str]:
    loc = feature.get("location") or {}
    start = loc.get("start") or {}
    end = loc.get("end") or {}
    start_value = start.get("value")
    end_value = end.get("value")
    modifier = _pipe([_clean(start.get("modifier")), _clean(end.get("modifier"))])
    try:
        start_int = int(start_value) if start_value is not None else None
    except (TypeError, ValueError):
        start_int = None
    try:
        end_int = int(end_value) if end_value is not None else start_int
    except (TypeError, ValueError):
        end_int = start_int
    return start_int, end_int, modifier


def _sequence_context(sequence: str, position: int | None, flank: int = 7) -> str:
    if not sequence or position is None or position < 1 or position > len(sequence):
        return ""
    start = max(0, position - 1 - flank)
    end = min(len(sequence), position + flank)
    return sequence[start:end]


def _residue_from_description(description: str) -> str:
    text = description.lower()
    for name, one_letter in RESIDUE_HINTS.items():
        if name in text:
            return one_letter
    return ""


def _psi_mod(description: str, feature_type: str) -> str:
    text = f"{description} {feature_type}".lower()
    for needle, mod_id in PSI_MOD_HINTS:
        if needle in text:
            return mod_id
    return ""


def _entry_sequence(entry: Mapping[str, Any]) -> str:
    seq = entry.get("sequence") or {}
    return _clean(seq.get("value"))


def _entry_disease_comments(entry: Mapping[str, Any]) -> list[dict[str, Any]]:
    comments = entry.get("comments") or []
    return [c for c in comments if _clean(c.get("commentType")).upper() == "DISEASE"]


def _extract_evidence(feature: Mapping[str, Any]) -> tuple[str, str]:
    eco: list[str] = []
    pmids: list[str] = []
    for ev in feature.get("evidences") or []:
        code = _clean(ev.get("evidenceCode"))
        if code:
            eco.append(code)
        if _clean(ev.get("source")).lower() == "pubmed":
            pmid = _clean(ev.get("id"))
            if pmid:
                pmids.append(f"PMID:{pmid}" if pmid.isdigit() else pmid)
    return _pipe(eco), _pipe(pmids)


def _site_id(accession: str, feature: Mapping[str, Any], position: int | None, end_position: int | None) -> str:
    desc = _clean(feature.get("description")) or _clean(feature.get("type"))
    ftype = _clean(feature.get("type"))
    span = str(position or "unknown") if position == end_position else f"{position or 'unknown'}-{end_position or 'unknown'}"
    fid = _clean(feature.get("featureId"))
    suffix = _slug(fid or desc)[:80]
    return f"PTMSITE:{accession}:{_slug(ftype)}:{span}:{suffix}"


def _protein_mapping(protein_nodes: pd.DataFrame) -> dict[str, str]:
    if "id" not in protein_nodes.columns or "uniprot_id" not in protein_nodes.columns:
        raise ValueError("protein nodes must contain id and uniprot_id columns")
    out: dict[str, str] = {}
    for _, row in protein_nodes[["id", "uniprot_id"]].dropna().iterrows():
        protein_id = _clean(row["id"])
        for token in _clean(row["uniprot_id"]).split("|"):
            accession = token.strip()
            if accession and accession not in out:
                out[accession] = protein_id
    return out


def _disease_link_candidates(
    entry: Mapping[str, Any],
    site_row: Mapping[str, Any],
    feature: Mapping[str, Any],
    *,
    generic_disease_comments: bool,
) -> list[dict[str, Any]]:
    """Return conservative feature-local disease candidates.

    Generic UniProt disease comments are deliberately not converted to site-level
    evidence.  They are only counted in summary diagnostics.
    """

    description = _clean(feature.get("description"))
    candidates: list[dict[str, Any]] = []
    if DISEASE_TOKEN_RE.search(description):
        ids = ID_TOKEN_RE.findall(description)
        candidates.append(
            {
                "ptm_site_id": site_row["id"],
                "uniprot_accession": site_row["uniprot_accession"],
                "protein_id": site_row["protein_id"],
                "feature_type": site_row["feature_type"],
                "residue": site_row["residue"],
                "position": site_row["position"],
                "description": description,
                "candidate_text": description,
                "candidate_kind": "feature_description",
                "disease_or_phenotype_id": _pipe([i.upper().replace("_", ":") for i in ids]),
                "disease_or_phenotype_name": "",
                "pmids": site_row["pmids"],
                "source_record_id": _clean(feature.get("featureId")) or site_row["id"],
                "decision": "candidate_only_requires_manual_review_explicit_site_level_assertion",
            }
        )
    # Keep the gate visible for reviewers without leaking generic protein disease
    # into graph edges.
    if generic_disease_comments:
        pass
    return candidates


def build_ptm_rows_from_uniprot_results(
    entries: Sequence[Mapping[str, Any]],
    protein_nodes: pd.DataFrame,
    *,
    release: str,
    created_at: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, int]]:
    created_at = created_at or datetime.now(timezone.utc).isoformat()
    accession_to_protein = _protein_mapping(protein_nodes)
    site_rows: list[dict[str, Any]] = []
    edge_rows: list[dict[str, Any]] = []
    evidence_rows: list[dict[str, Any]] = []
    disease_candidates: list[dict[str, Any]] = []
    counts = {
        "entries_seen": 0,
        "entries_with_ptm_features": 0,
        "ptm_features_seen": 0,
        "ptm_features_with_exact_position": 0,
        "ptm_features_mapped_to_protein": 0,
        "ptm_features_missing_protein_mapping": 0,
        "entries_with_generic_disease_comments": 0,
        "site_level_disease_candidate_rows": 0,
    }

    for entry in entries:
        counts["entries_seen"] += 1
        accession = _clean(entry.get("primaryAccession"))
        sequence = _entry_sequence(entry)
        disease_comments = _entry_disease_comments(entry)
        has_disease_comments = bool(disease_comments)
        if has_disease_comments:
            counts["entries_with_generic_disease_comments"] += 1
        ptm_features = [f for f in entry.get("features") or [] if _clean(f.get("type")) in PTM_FEATURE_TYPES]
        if ptm_features:
            counts["entries_with_ptm_features"] += 1
        protein_id = accession_to_protein.get(accession, "")
        for feature in ptm_features:
            counts["ptm_features_seen"] += 1
            position, end_position, modifier = _feature_position(feature)
            if position is None or end_position is None or modifier not in {"", "EXACT"}:
                continue
            counts["ptm_features_with_exact_position"] += 1
            if not protein_id:
                counts["ptm_features_missing_protein_mapping"] += 1
                continue
            counts["ptm_features_mapped_to_protein"] += 1
            description = _clean(feature.get("description"))
            feature_type = _clean(feature.get("type"))
            eco_codes, pmids = _extract_evidence(feature)
            site_id = _site_id(accession, feature, position, end_position)
            residue = sequence[position - 1] if sequence and position <= len(sequence) else _residue_from_description(description)
            site_row = {
                "id": site_id,
                "uniprot_accession": accession,
                "protein_id": protein_id,
                "isoform": "canonical",
                "feature_type": feature_type,
                "modification_type": description or feature_type,
                "psi_mod_id": _psi_mod(description, feature_type),
                "residue": residue,
                "position": str(position),
                "end_position": str(end_position),
                "location_modifier": modifier,
                "sequence_context": _sequence_context(sequence, position),
                "feature_id": _clean(feature.get("featureId")),
                "description": description,
                "evidence_codes": eco_codes,
                "pmids": pmids,
                "source": SOURCE,
                "source_release": release,
            }
            site_rows.append(site_row)
            edge_rows.append(
                {
                    "x_id": protein_id,
                    "x_type": "protein",
                    "y_id": site_id,
                    "y_type": "ptm_site",
                    "relation": RELATION,
                    "display_relation": DISPLAY_RELATION,
                    "source": f"{SOURCE}/{release}",
                    "credibility": 3,
                }
            )
            edge_key = f"{RELATION}|{protein_id}|{site_id}"
            source_record_id = _clean(feature.get("featureId")) or f"{accession}:{feature_type}:{position}:{description}"
            evidence_rows.append(
                {
                    "edge_key": edge_key,
                    "relation": RELATION,
                    "x_id": protein_id,
                    "x_type": "protein",
                    "y_id": site_id,
                    "y_type": "ptm_site",
                    "evidence_type": "database_record",
                    "source": SOURCE,
                    "source_dataset": SOURCE_DATASET,
                    "source_record_id": source_record_id,
                    "paper_id": "",
                    "uniprot_accession": accession,
                    "ensp_id": protein_id,
                    "isoform": "canonical",
                    "residue": site_row["residue"],
                    "position": site_row["position"],
                    "end_position": site_row["end_position"],
                    "modification_type": site_row["modification_type"],
                    "psi_mod_id": site_row["psi_mod_id"],
                    "eco_codes": eco_codes,
                    "predicate": feature_type,
                    "sequence_context": site_row["sequence_context"],
                    "release": release,
                    "created_at": created_at,
                }
            )
            for pmid in [p for p in pmids.split("|") if p]:
                evidence_rows.append(
                    {
                        "edge_key": edge_key,
                        "relation": RELATION,
                        "x_id": protein_id,
                        "x_type": "protein",
                        "y_id": site_id,
                        "y_type": "ptm_site",
                        "evidence_type": "paper",
                        "source": SOURCE,
                        "source_dataset": SOURCE_DATASET,
                        "source_record_id": f"{source_record_id}:{pmid}",
                        "paper_id": pmid,
                        "uniprot_accession": accession,
                        "ensp_id": protein_id,
                        "isoform": "canonical",
                        "residue": site_row["residue"],
                        "position": site_row["position"],
                        "end_position": site_row["end_position"],
                        "modification_type": site_row["modification_type"],
                        "psi_mod_id": site_row["psi_mod_id"],
                        "eco_codes": eco_codes,
                        "predicate": feature_type,
                        "sequence_context": site_row["sequence_context"],
                        "release": release,
                        "created_at": created_at,
                    }
                )
            disease_candidates.extend(
                _disease_link_candidates(entry, site_row, feature, generic_disease_comments=has_disease_comments)
            )

    sites = pd.DataFrame(site_rows, columns=PTM_SITE_COLUMNS).drop_duplicates(subset=["id"]).reset_index(drop=True)
    edges = pd.DataFrame(edge_rows, columns=EDGE_COLUMNS).drop_duplicates(subset=["x_id", "y_id", "relation"]).reset_index(drop=True)
    evidence = pd.DataFrame(evidence_rows, columns=EVIDENCE_COLUMNS).drop_duplicates(
        subset=["relation", "x_id", "y_id", "evidence_type", "source_record_id", "paper_id"], keep="last"
    ).reset_index(drop=True)
    candidates = pd.DataFrame(disease_candidates, columns=DISEASE_LINK_COLUMNS).drop_duplicates().reset_index(drop=True)
    counts["site_level_disease_candidate_rows"] = int(len(candidates))
    counts["ptm_site_nodes"] = int(len(sites))
    counts["protein_has_ptm_site_edges"] = int(len(edges))
    counts["protein_has_ptm_site_evidence_rows"] = int(len(evidence))
    return sites, edges, evidence, candidates, counts


def _next_link(headers: Mapping[str, str]) -> str | None:
    link = headers.get("Link") or headers.get("link") or ""
    for part in link.split(","):
        if 'rel="next"' in part:
            match = re.search(r"<([^>]+)>", part)
            if match:
                return match.group(1)
    return None


def _fetch_uniprot_page(url: str, *, attempts: int = 4) -> tuple[dict[str, Any], Mapping[str, str]]:
    last_error: BaseException | None = None
    for attempt in range(1, attempts + 1):
        req = urllib.request.Request(url, headers={"User-Agent": "txgnn-uniprot-ptm-sites/0.1"})
        try:
            with urllib.request.urlopen(req, timeout=120) as response:
                raw = response.read()
                payload = json.loads(raw)
                return payload, dict(response.headers.items())
        except (urllib.error.URLError, TimeoutError, http.client.IncompleteRead, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt == attempts:
                break
            time.sleep(min(2**attempt, 10))
    assert last_error is not None
    raise last_error


def fetch_uniprot_reviewed_human_ptm_entries(*, size: int = 100, max_pages: int | None = None) -> tuple[list[dict[str, Any]], str, int]:
    query = "(reviewed:true) AND (organism_id:9606) AND (ft_mod_res:* OR ft_lipid:* OR ft_carbohyd:* OR ft_disulfid:*)"
    params = {"query": query, "format": "json", "size": str(size)}
    url = "https://rest.uniprot.org/uniprotkb/search?" + urllib.parse.urlencode(params)
    entries: list[dict[str, Any]] = []
    release = ""
    total_results = 0
    pages = 0
    while url:
        pages += 1
        payload, headers = _fetch_uniprot_page(url)
        release = headers.get("X-UniProt-Release") or release
        total_results = int(headers.get("X-Total-Results") or total_results or 0)
        entries.extend(cast(list[dict[str, Any]], payload.get("results") or []))
        url = _next_link(headers)
        if max_pages is not None and pages >= max_pages:
            break
    return entries, release, total_results


def _write_table(path: str, frame: pd.DataFrame) -> None:
    fs, fs_path = fsspec.core.url_to_fs(path)
    parent = fs_path.rsplit("/", 1)[0] if "/" in fs_path else ""
    if parent:
        fs.makedirs(parent, exist_ok=True)
    frame.to_parquet(path, index=False)


def _write_json(path: str, payload: Mapping[str, Any]) -> None:
    fs, fs_path = fsspec.core.url_to_fs(path)
    parent = fs_path.rsplit("/", 1)[0] if "/" in fs_path else ""
    if parent:
        fs.makedirs(parent, exist_ok=True)
    with fs.open(fs_path, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _join_uri(root: str, *parts: str) -> str:
    return root.rstrip("/") + "/" + "/".join(p.strip("/") for p in parts)


def _refuse_canonical(output_root: str) -> None:
    normalized = output_root.rstrip("/")
    if normalized in CANONICAL_ROOTS:
        raise ValueError("refusing to write canonical KG root; choose a staging root")
    if "://" not in normalized:
        try:
            if str(Path(normalized).resolve()) in CANONICAL_ROOTS:
                raise ValueError("refusing to write canonical KG root; choose a staging root")
        except FileNotFoundError:
            pass


def build_uniprot_ptm_sites(
    *,
    output_root: str,
    protein_nodes_path: str,
    max_pages: int | None = None,
    entries_json_path: str | None = None,
) -> dict[str, Any]:
    _refuse_canonical(output_root)
    created_at = datetime.now(timezone.utc).isoformat()
    protein_nodes = pd.read_parquet(protein_nodes_path, columns=["id", "uniprot_id"])
    if entries_json_path:
        with fsspec.open(entries_json_path, "r") as handle:
            payload = json.load(handle)
        entries = payload["entries"] if isinstance(payload, dict) and "entries" in payload else payload
        release = payload.get("release", "fixture") if isinstance(payload, dict) else "fixture"
        total_results = len(entries)
    else:
        entries, release, total_results = fetch_uniprot_reviewed_human_ptm_entries(max_pages=max_pages)

    sites, edges, evidence, disease_candidates, counts = build_ptm_rows_from_uniprot_results(
        entries, protein_nodes, release=release, created_at=created_at
    )
    _write_table(_join_uri(output_root, "nodes", "ptm_site.parquet"), sites)
    _write_table(_join_uri(output_root, "edges", f"{RELATION}.parquet"), edges)
    _write_table(_join_uri(output_root, "evidence", f"{RELATION}.parquet"), evidence)
    _write_table(_join_uri(output_root, "diagnostics", "ptm_site_disease_link_candidates.parquet"), disease_candidates)

    summary = {
        **counts,
        "uniprot_release": release,
        "uniprot_total_query_results": int(total_results),
        "created_at": created_at,
        "staging_only": True,
        "recommendation": (
            "stage ptm_site nodes and protein_has_ptm_site edges; do not build "
            "ptm_site_associated_disease/phenotype until explicit site-level disease/phenotype assertions are curated"
        ),
        "outputs": {
            "ptm_site_nodes": _join_uri(output_root, "nodes", "ptm_site.parquet"),
            "protein_has_ptm_site_edges": _join_uri(output_root, "edges", f"{RELATION}.parquet"),
            "protein_has_ptm_site_evidence": _join_uri(output_root, "evidence", f"{RELATION}.parquet"),
            "disease_link_candidates": _join_uri(output_root, "diagnostics", "ptm_site_disease_link_candidates.parquet"),
        },
        "validation": {
            "missing_parent_protein_edges": 0,
            "disease_phenotype_edges_written": 0,
            "disease_phenotype_gate": "no generic protein disease + generic PTM joins were materialized",
        },
    }
    _write_json(_join_uri(output_root, "reports", "uniprot_ptm_sites_summary.json"), summary)
    return summary


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, help="Local or gs:// staging root")
    parser.add_argument(
        "--protein-nodes",
        default="gs://jouvencekb/main/nodes/protein.parquet",
        help="Read-only protein node parquet with id and uniprot_id columns",
    )
    parser.add_argument("--max-pages", type=int, default=None, help="Debug limit for UniProt pagination")
    parser.add_argument("--entries-json", default=None, help="Optional fixture/cache JSON instead of live UniProt download")
    args = parser.parse_args(list(argv) if argv is not None else None)
    summary = build_uniprot_ptm_sites(
        output_root=args.output_root,
        protein_nodes_path=args.protein_nodes,
        max_pages=args.max_pages,
        entries_json_path=args.entries_json,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
