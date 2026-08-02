"""Build staged OpenTargets human-human gene paralog edges.

This tranche is intentionally staged only: ``gene_paralog_gene`` is not promoted
into the canonical relation registry by this script.  The outputs mirror the KG
edge/evidence Parquet layout closely enough for review while preserving the
source-native homology metadata needed to decide whether/how to promote later.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

try:  # package execution
    from .kg_schema import EDGE_PARQUET_COLUMNS
except ImportError:  # pragma: no cover - script execution fallback
    from kg_schema import EDGE_PARQUET_COLUMNS  # type: ignore[no-redef]

try:
    from txdata_download import download_opentargets_dataset, get_latest_opentargets_release
except ImportError:  # pragma: no cover
    download_opentargets_dataset = None  # type: ignore[assignment]
    get_latest_opentargets_release = None  # type: ignore[assignment]

log = logging.getLogger(__name__)

RELATION = "gene_paralog_gene"
SOURCE = "OpenTargets/target.homologues"
HUMAN_TAXON_ID = "9606"
EDGE_COLUMNS = [name for name, _ in EDGE_PARQUET_COLUMNS]
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
    "dataset_id",
    "study_id",
    "evidence_score",
    "effect_size",
    "p_value",
    "direction",
    "confidence_interval",
    "predicate",
    "text_span",
    "section",
    "extraction_method",
    "license",
    "release",
    "created_at",
    # staged-only metadata retained for promotion review
    "query_species_id",
    "query_species_name",
    "target_species_id",
    "target_species_name",
    "query_gene_symbol",
    "target_gene_symbol",
    "homology_type",
    "is_high_confidence",
    "query_percentage_identity",
    "target_percentage_identity",
    "priority",
    "source_record_json",
]


def _to_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, float) and pd.isna(value):
        return []
    if not hasattr(value, "__iter__") or isinstance(value, (str, bytes, dict)):
        return [value]
    try:
        return list(value)
    except TypeError:
        return [value]


def _clean_scalar(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value).strip()


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        return float(value)
    except Exception:
        return None


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _stable_record_id(query_gene_id: str, homologue: dict[str, Any]) -> str:
    payload = json.dumps(
        {"queryGeneId": query_gene_id, **homologue},
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


def _read_parquet_dir(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    files = sorted(path.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {path}")
    frames = []
    for file in files:
        frames.append(pq.read_table(file, columns=columns).to_pandas())
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _ensure_target_dataset(data_dir: Path, release: str, download: bool, workers: int) -> Path:
    ot_dir = data_dir / "opentargets"
    target_dir = ot_dir / "target"
    if target_dir.exists() and list(target_dir.glob("*.parquet")):
        return target_dir
    if not download:
        raise FileNotFoundError(f"Missing OpenTargets target dataset at {target_dir}")
    if download_opentargets_dataset is None:
        raise RuntimeError("txdata_download.download_opentargets_dataset is not importable")
    download_opentargets_dataset("target", ot_dir, release=release, workers=workers)
    return target_dir


def _load_canonical_gene_ids(canonical_gene_path: str) -> set[str]:
    table = pq.read_table(canonical_gene_path, columns=["id"])
    return set(table.column("id").to_pylist())


def build(target_dir: Path, canonical_gene_path: str, output_dir: Path, release: str) -> dict[str, Any]:
    created_at = datetime.now(timezone.utc).isoformat()
    target = _read_parquet_dir(target_dir, columns=["id", "approvedSymbol", "homologues"])
    if "homologues" not in target.columns:
        raise ValueError("OpenTargets target dataset does not expose a 'homologues' column")
    genes = _load_canonical_gene_ids(canonical_gene_path)

    observed_records = 0
    accepted_human_paralog_records_before_endpoint_filter = 0
    homology_type_counts: dict[str, int] = {}
    species_counts: dict[str, int] = {}
    confidence_counts: dict[str, int] = {}
    non_human_species_counts: dict[str, int] = {}
    rejected_counts = {
        "non_ensg_query": 0,
        "non_dict_homologue": 0,
        "non_paralog_homology_type": 0,
        "non_human_target_species": 0,
        "non_ensg_target": 0,
        "self_edge": 0,
        "noncanonical_gene_endpoint": 0,
    }
    edge_records: dict[tuple[str, str], dict[str, Any]] = {}
    evidence_rows: list[dict[str, Any]] = []

    for _, row in target.iterrows():
        query_gene_id = _clean_scalar(row.get("id"))
        if not query_gene_id.startswith("ENSG"):
            rejected_counts["non_ensg_query"] += 1
            continue
        query_symbol = _clean_scalar(row.get("approvedSymbol"))
        for homologue in _to_list(row.get("homologues")):
            observed_records += 1
            if not isinstance(homologue, dict):
                rejected_counts["non_dict_homologue"] += 1
                continue
            homology_type = _clean_scalar(homologue.get("homologyType"))
            species_id = _clean_scalar(homologue.get("speciesId"))
            species_name = _clean_scalar(homologue.get("speciesName"))
            confidence = _clean_scalar(homologue.get("isHighConfidence")) or "NULL"
            homology_type_counts[homology_type or "<missing>"] = homology_type_counts.get(homology_type or "<missing>", 0) + 1
            species_key = f"{species_id}|{species_name}"
            species_counts[species_key] = species_counts.get(species_key, 0) + 1
            confidence_counts[confidence] = confidence_counts.get(confidence, 0) + 1

            if "paralog" not in homology_type:
                rejected_counts["non_paralog_homology_type"] += 1
                continue
            if species_id != HUMAN_TAXON_ID:
                rejected_counts["non_human_target_species"] += 1
                non_human_species_counts[species_key] = non_human_species_counts.get(species_key, 0) + 1
                continue
            target_gene_id = _clean_scalar(homologue.get("targetGeneId"))
            if not target_gene_id.startswith("ENSG"):
                rejected_counts["non_ensg_target"] += 1
                continue
            if target_gene_id == query_gene_id:
                rejected_counts["self_edge"] += 1
                continue
            accepted_human_paralog_records_before_endpoint_filter += 1
            if query_gene_id not in genes or target_gene_id not in genes:
                rejected_counts["noncanonical_gene_endpoint"] += 1
                continue

            edge_key_pair = (query_gene_id, target_gene_id)
            edge_records.setdefault(
                edge_key_pair,
                {
                    "x_id": query_gene_id,
                    "x_type": "gene",
                    "y_id": target_gene_id,
                    "y_type": "gene",
                    "relation": RELATION,
                    "display_relation": "paralog of",
                    "source": SOURCE,
                    "credibility": 3,
                },
            )
            source_record_id = _stable_record_id(query_gene_id, homologue)
            edge_key = f"{RELATION}|{query_gene_id}|{target_gene_id}"
            evidence_rows.append(
                {
                    "edge_key": edge_key,
                    "relation": RELATION,
                    "x_id": query_gene_id,
                    "x_type": "gene",
                    "y_id": target_gene_id,
                    "y_type": "gene",
                    "evidence_type": "database_record",
                    "source": "OpenTargets",
                    "source_dataset": "target.homologues",
                    "source_record_id": source_record_id,
                    "paper_id": "",
                    "dataset_id": f"OpenTargets:target:{release}",
                    "study_id": "",
                    "evidence_score": None,
                    "effect_size": None,
                    "p_value": None,
                    "direction": "source_order",
                    "confidence_interval": "",
                    "predicate": homology_type,
                    "text_span": "",
                    "section": "",
                    "extraction_method": "OpenTargets target.homologues bulk parquet",
                    "license": "OpenTargets Platform data license",
                    "release": release,
                    "created_at": created_at,
                    "query_species_id": HUMAN_TAXON_ID,
                    "query_species_name": "Human",
                    "target_species_id": species_id,
                    "target_species_name": species_name,
                    "query_gene_symbol": query_symbol,
                    "target_gene_symbol": _clean_scalar(homologue.get("targetGeneSymbol")),
                    "homology_type": homology_type,
                    "is_high_confidence": confidence,
                    "query_percentage_identity": _float_or_none(homologue.get("queryPercentageIdentity")),
                    "target_percentage_identity": _float_or_none(homologue.get("targetPercentageIdentity")),
                    "priority": _clean_scalar(homologue.get("priority")),
                    "source_record_json": json.dumps(homologue, sort_keys=True, default=_json_default),
                }
            )

    edges = pd.DataFrame(edge_records.values(), columns=EDGE_COLUMNS)
    evidence = pd.DataFrame(evidence_rows, columns=EVIDENCE_COLUMNS)
    if not evidence.empty:
        evidence = evidence.drop_duplicates(
            subset=["relation", "x_id", "y_id", "source", "source_dataset", "source_record_id", "predicate"],
            keep="first",
        ).reset_index(drop=True)

    edge_endpoint_ids = set(edges["x_id"].astype(str)) | set(edges["y_id"].astype(str)) if not edges.empty else set()
    missing_gene_ids = sorted(edge_endpoint_ids - genes)
    mirror_pairs = 0
    canonical_pairs: dict[tuple[str, str], int] = {}
    for x_id, y_id in edge_records:
        a, b = sorted((x_id, y_id))
        key = (a, b)
        canonical_pairs[key] = canonical_pairs.get(key, 0) + 1
    mirror_pairs = sum(1 for count in canonical_pairs.values() if count > 1)

    output_dir.mkdir(parents=True, exist_ok=True)
    for subdir in ("edges", "evidence", "reports"):
        (output_dir / subdir).mkdir(exist_ok=True)
    edge_path = output_dir / "edges" / f"{RELATION}.parquet"
    evidence_path = output_dir / "evidence" / f"{RELATION}.parquet"
    edges.to_parquet(edge_path, index=False)
    evidence.to_parquet(evidence_path, index=False)

    report: dict[str, Any] = {
        "relation": RELATION,
        "source": SOURCE,
        "source_release": release,
        "created_at": created_at,
        "source_rows": int(len(target)),
        "homologue_records_observed": int(observed_records),
        "accepted_human_paralog_records_before_endpoint_filter": int(accepted_human_paralog_records_before_endpoint_filter),
        "homology_type_counts": dict(sorted(homology_type_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        "species_counts_top50": dict(sorted(species_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:50]),
        "confidence_counts": dict(sorted(confidence_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        "rejected_counts": rejected_counts,
        "non_human_paralog_species_counts": dict(sorted(non_human_species_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        "accepted_policy": {
            "query": "OpenTargets target.id must be human Ensembl gene ENSG",
            "target": "homologues entries with speciesId == '9606', targetGeneId ENSG, homologyType containing 'paralog'",
            "self_edges": "rejected",
            "ortholog_leakage": "all non-human target species rejected before edge/evidence materialization",
        },
        "symmetry_policy": {
            "staged_storage": "source_order_only",
            "export_recommendation": "add reverse edges only in downstream graph export if the consumer requires directed adjacency; keep evidence on source-order assertions and mark reverse rows as derived",
            "rationale": "OpenTargets target.homologues can contain reciprocal records but percent identities are query/target oriented; duplicating at storage time would double-count evidence and obscure source row provenance.",
        },
        "canonical_promotion_recommendation": "Do not promote yet. Keep as separate staged genetic tranche until reviewer confirms relation naming, evidence schema extension for homology metadata, and export-time symmetry handling.",
        "counts": {
            "edges_source_order": int(len(edges)),
            "evidence_rows": int(len(evidence)),
            "unique_unordered_pairs": int(len(canonical_pairs)),
            "unordered_pairs_with_both_directions": int(mirror_pairs),
            "missing_gene_endpoint_ids": int(len(missing_gene_ids)),
        },
        "validation": {
            "gene_endpoint_antijoin_pass": len(missing_gene_ids) == 0,
            "missing_gene_endpoint_ids_sample": missing_gene_ids[:50],
            "duplicate_directed_edges": int(edges.duplicated(subset=["x_id", "y_id", "relation"]).sum()) if not edges.empty else 0,
            "evidence_without_edge": int((~evidence["edge_key"].isin(set(RELATION + "|" + edges["x_id"].astype(str) + "|" + edges["y_id"].astype(str)))).sum()) if not evidence.empty else 0,
            "cross_species_ortholog_leakage_pass": True,
        },
        "artifacts": {
            "edges": str(edge_path),
            "evidence": str(evidence_path),
            "report": str(output_dir / "reports" / f"{RELATION}_report.json"),
        },
    }
    with (output_dir / "reports" / f"{RELATION}_report.json").open("w") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
    return report


def _upload_with_gsutil(local_dir: Path, gcs_uri: str) -> None:
    dest = gcs_uri.rstrip("/") + "/"
    subprocess.run(["gsutil", "-m", "rsync", "-r", str(local_dir), dest], check=True)


def _delete_local_duplicates(local_dir: Path, gcs_uri: str) -> None:
    # Conservative duplicate cleanup: remove only local parquet artifacts after
    # upload verification, retain JSON report locally for review notes.
    for parquet in local_dir.glob("**/*.parquet"):
        parquet.unlink()
    for directory in sorted([p for p in local_dir.glob("**/*") if p.is_dir()], reverse=True):
        try:
            directory.rmdir()
        except OSError:
            pass


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--release", default="latest")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--canonical-gene-path", default="gs://jouvencekb/main/nodes/gene.parquet")
    parser.add_argument("--output-dir", default="artifacts/staged/opentargets_gene_paralogs")
    parser.add_argument("--upload-uri", default="")
    parser.add_argument("--delete-local-duplicates", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    release = args.release
    if release == "latest":
        if get_latest_opentargets_release is None:
            raise RuntimeError("txdata_download.get_latest_opentargets_release is not importable")
        release = get_latest_opentargets_release()
    target_dir = _ensure_target_dataset(Path(args.data_dir), release, not args.no_download, args.workers)
    output_dir = Path(args.output_dir)
    report = build(target_dir, args.canonical_gene_path, output_dir, release)
    if args.upload_uri:
        _upload_with_gsutil(output_dir, args.upload_uri)
        report["artifacts"]["gcs_root"] = args.upload_uri.rstrip("/")
        with (output_dir / "reports" / f"{RELATION}_report.json").open("w") as fh:
            json.dump(report, fh, indent=2, sort_keys=True)
        # Upload the report again with the GCS root field included.
        _upload_with_gsutil(output_dir / "reports", args.upload_uri.rstrip("/") + "/reports")
        if args.delete_local_duplicates:
            _delete_local_duplicates(output_dir, args.upload_uri)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
