"""Local builder for mutation_associated_phenotype from OpenTargets EVA HPO evidence.

This module intentionally writes only to a caller-provided KG root.  It does not
promote or write canonical GCS.  The intended source is the OpenTargets
``evidence_eva`` parquet cache containing ClinVar/EVA-style variant evidence.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.dataset as ds

from manage_db import kg_evidence, kg_storage

RELATION = "mutation_associated_phenotype"
SOURCE = "OpenTargets"
SOURCE_DATASET = "evidence_eva"
DISPLAY_RELATION = "associated phenotype"

_REQUIRED_SOURCE_COLUMNS = ["variantId", "diseaseId", "clinicalSignificances"]
_OPTIONAL_SOURCE_COLUMNS = [
    "id",
    "studyId",
    "score",
    "literature",
    "alleleOrigins",
    "datasourceId",
    "datasourceVersion",
    "variantFunctionalConsequenceId",
    "variantFunctionalConsequenceFromSourceId",
    "releaseDate",
]


def _as_list(value: Any) -> list[Any]:
    if value is None or value is pd.NA:
        return []
    if isinstance(value, float) and pd.isna(value):
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    # pyarrow list values commonly arrive as ndarray-like scalars.
    if hasattr(value, "tolist") and not isinstance(value, str):
        converted = value.tolist()
        return converted if isinstance(converted, list) else [converted]
    return [value]


def _clean_token(value: Any) -> str:
    return str(value).strip().replace("_", " ").lower()


def normalize_hp_id(value: Any) -> str | None:
    """Normalize HPO IDs from HP_0001250/HP:0001250 to HP:0001250."""

    if value is None or value is pd.NA:
        return None
    raw = str(value).strip()
    if raw.startswith("HP:"):
        suffix = raw[3:]
    elif raw.startswith("HP_"):
        suffix = raw[3:]
    else:
        return None
    return f"HP:{suffix}" if suffix.isdigit() and len(suffix) == 7 else None


def clinical_significance_predicate(value: Any) -> str:
    """Return normalized clinical-significance metadata for an HP row."""

    terms = sorted({_clean_token(item) for item in _as_list(value) if str(item).strip()})
    return ";".join(terms)


def normalize_pmid(value: Any) -> str | None:
    text = str(value).strip()
    if not text:
        return None
    if text.upper().startswith("PMID:"):
        suffix = text.split(":", 1)[1].strip()
    else:
        suffix = text
    return f"PMID:{suffix}" if suffix.isdigit() else None


def _source_record_id(row: pd.Series, row_number: int) -> str:
    for col in ("id", "studyId"):
        value = row.get(col)
        if value is not None and value is not pd.NA and str(value).strip():
            return str(value).strip()
    return f"{row.get('variantId')}:{row.get('diseaseId')}:{row_number}"


def build_rows_from_eva_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter EVA rows and return canonical edge/evidence DataFrames.

    Filters:
    - ``diseaseId`` must be an HPO endpoint, normalized to ``HP:nnnnnnn``.
    - all clinical-significance classes are retained as predicate metadata.
    - no disease-to-phenotype mapping is attempted.
    """

    rows: list[dict[str, Any]] = []
    evidence_rows: list[dict[str, Any]] = []
    for row_number, row in frame.reset_index(drop=True).iterrows():
        variant_id = str(row.get("variantId", "")).strip()
        hp_id = normalize_hp_id(row.get("diseaseId"))
        predicate = clinical_significance_predicate(row.get("clinicalSignificances"))
        if not variant_id or hp_id is None:
            continue

        source_record_id = _source_record_id(row, row_number)
        edge_key = f"{RELATION}|{variant_id}|{hp_id}"
        rows.append(
            {
                "x_id": variant_id,
                "x_type": "mutation",
                "y_id": hp_id,
                "y_type": "phenotype",
                "relation": RELATION,
                "display_relation": DISPLAY_RELATION,
                "source": f"{SOURCE}/{SOURCE_DATASET}",
                "credibility": 3,
            }
        )
        release = row.get("datasourceVersion") or row.get("releaseDate") or ""
        study_id = row.get("studyId") or ""
        evidence_rows.append(
            {
                "edge_key": edge_key,
                "relation": RELATION,
                "x_id": variant_id,
                "x_type": "mutation",
                "y_id": hp_id,
                "y_type": "phenotype",
                "evidence_type": "database_record",
                "source": SOURCE,
                "source_dataset": SOURCE_DATASET,
                "source_record_id": source_record_id,
                "paper_id": "",
                "study_id": str(study_id) if str(study_id) != "<NA>" else "",
                "evidence_score": row.get("score"),
                "predicate": predicate,
                "extraction_method": "OpenTargets EVA ClinVar clinical significance",
                "release": str(release) if str(release) != "<NA>" else "",
                "created_at": str(row.get("releaseDate") or ""),
            }
        )
        for pmid in sorted({p for p in (normalize_pmid(v) for v in _as_list(row.get("literature"))) if p}):
            evidence_rows.append(
                {
                    "edge_key": edge_key,
                    "relation": RELATION,
                    "x_id": variant_id,
                    "x_type": "mutation",
                    "y_id": hp_id,
                    "y_type": "phenotype",
                    "evidence_type": "paper",
                    "source": SOURCE,
                    "source_dataset": SOURCE_DATASET,
                    "source_record_id": f"{source_record_id}:{pmid}",
                    "paper_id": pmid,
                    "study_id": str(study_id) if str(study_id) != "<NA>" else "",
                    "evidence_score": row.get("score"),
                    "predicate": predicate,
                    "extraction_method": "OpenTargets EVA ClinVar literature support",
                    "release": str(release) if str(release) != "<NA>" else "",
                    "created_at": str(row.get("releaseDate") or ""),
                }
            )

    edges = pd.DataFrame(rows)
    if not edges.empty:
        edges = edges.drop_duplicates(subset=["x_id", "y_id", "relation"]).reset_index(drop=True)
    evidence = pd.DataFrame(evidence_rows)
    if not evidence.empty:
        evidence = evidence.drop_duplicates(
            subset=["relation", "x_id", "y_id", "evidence_type", "source_record_id", "paper_id", "predicate"],
            keep="last",
        ).reset_index(drop=True)
    return edges, evidence


def _eva_dataset(source_path: str | Path) -> tuple[ds.Dataset, list[str]]:
    source = Path(source_path)
    parquet_paths = sorted(
        p for p in (source.glob("*.parquet") if source.is_dir() else [source]) if p.name.endswith(".parquet")
    )
    if not parquet_paths:
        raise FileNotFoundError(f"no parquet files found under {source}")
    dataset = ds.dataset([str(p) for p in parquet_paths], format="parquet")
    available = set(dataset.schema.names)
    missing = [col for col in _REQUIRED_SOURCE_COLUMNS if col not in available]
    if missing:
        raise ValueError(f"EVA source missing required columns: {missing}")
    columns = [col for col in [*_REQUIRED_SOURCE_COLUMNS, *_OPTIONAL_SOURCE_COLUMNS] if col in available]
    return dataset, columns


def _read_eva_source(source_path: str | Path) -> pd.DataFrame:
    dataset, columns = _eva_dataset(source_path)
    return dataset.to_table(columns=columns).to_pandas()


def _iter_eva_source_frames(source_path: str | Path, *, batch_size: int = 100_000) -> Iterable[pd.DataFrame]:
    """Yield bounded pandas batches from an EVA parquet file or directory."""

    dataset, columns = _eva_dataset(source_path)
    for batch in dataset.to_batches(columns=columns, batch_size=batch_size):
        if batch.num_rows:
            yield batch.to_pandas()


def _endpoint_ids(root: kg_storage.KGRoot, node_type: str) -> set[str] | None:
    try:
        if not root.fs.exists(root._node_internal(node_type)):
            return None
        nodes = kg_storage.read_nodes(root, node_type, columns=["id"])
        return set(nodes["id"].astype(str))
    except FileNotFoundError:
        return None


def _filter_to_existing_endpoints(
    edges: pd.DataFrame,
    evidence: pd.DataFrame,
    endpoint_root: kg_storage.KGRoot,
) -> tuple[pd.DataFrame, pd.DataFrame, int, int]:
    mutation_ids = _endpoint_ids(endpoint_root, "mutation")
    phenotype_ids = _endpoint_ids(endpoint_root, "phenotype")
    if edges.empty:
        return edges, evidence, 0, 0
    keep = pd.Series(True, index=edges.index)
    missing_mutations = 0
    missing_phenotypes = 0
    if mutation_ids is not None:
        missing_mutations = int((~edges["x_id"].astype(str).isin(mutation_ids)).sum())
        keep &= edges["x_id"].astype(str).isin(mutation_ids)
    if phenotype_ids is not None:
        missing_phenotypes = int((~edges["y_id"].astype(str).isin(phenotype_ids)).sum())
        keep &= edges["y_id"].astype(str).isin(phenotype_ids)
    kept_edges = edges.loc[keep].reset_index(drop=True)
    kept_keys = set(kept_edges["relation"] + "|" + kept_edges["x_id"] + "|" + kept_edges["y_id"])
    if evidence.empty:
        kept_evidence = evidence
    else:
        ev_keys = evidence["relation"].astype(str) + "|" + evidence["x_id"].astype(str) + "|" + evidence["y_id"].astype(str)
        kept_evidence = evidence.loc[ev_keys.isin(kept_keys)].reset_index(drop=True)
    return kept_edges, kept_evidence, missing_mutations, missing_phenotypes


def build_local_mutation_associated_phenotype(
    source_path: str | Path,
    output_kg_root: str | Path,
    *,
    endpoint_kg_root: str | Path | None = None,
) -> dict[str, int]:
    """Build local edge/evidence parquets for ``mutation_associated_phenotype``."""

    output_text = str(output_kg_root)
    canonical = "/mnt/gcs/jouvencekb/main"
    if Path(output_text).resolve() == Path(canonical).resolve() or output_text.rstrip("/") == "gs://jouvencekb/main":
        raise ValueError("refusing to write canonical KG root")

    source_rows = 0
    hp_rows = 0
    associated_hp_rows = 0
    edge_chunks: list[pd.DataFrame] = []
    evidence_chunks: list[pd.DataFrame] = []
    for frame in _iter_eva_source_frames(source_path):
        source_rows += len(frame)
        hp_mask = frame["diseaseId"].map(normalize_hp_id).notna()
        hp_rows += int(hp_mask.to_numpy().sum())
        associated_hp_rows += int(hp_mask.to_numpy().sum())
        edges_chunk, evidence_chunk = build_rows_from_eva_frame(frame.loc[hp_mask].reset_index(drop=True))
        if not edges_chunk.empty:
            edge_chunks.append(edges_chunk)
        if not evidence_chunk.empty:
            evidence_chunks.append(evidence_chunk)

    edges = pd.concat(edge_chunks, ignore_index=True) if edge_chunks else pd.DataFrame()
    if not edges.empty:
        edges = edges.drop_duplicates(subset=["x_id", "y_id", "relation"]).reset_index(drop=True)
    evidence = pd.concat(evidence_chunks, ignore_index=True) if evidence_chunks else pd.DataFrame()
    if not evidence.empty:
        evidence = evidence.drop_duplicates(
            subset=["relation", "x_id", "y_id", "evidence_type", "source_record_id", "paper_id", "predicate"],
            keep="last",
        ).reset_index(drop=True)

    endpoint_root = kg_storage.open_kg_root(str(endpoint_kg_root or output_kg_root))
    edges, evidence, missing_mutations, missing_phenotypes = _filter_to_existing_endpoints(edges, evidence, endpoint_root)

    output_root = kg_storage.open_kg_root(str(output_kg_root))
    edge_rows = kg_storage.write_edges(output_root, RELATION, edges)
    evidence_rows = kg_evidence.write_evidence(output_root, RELATION, evidence)
    return {
        "source_rows": int(source_rows),
        "hp_rows": int(hp_rows),
        "associated_hp_rows": int(associated_hp_rows),
        "edge_rows": int(edge_rows),
        "evidence_rows": int(evidence_rows),
        "missing_mutation_endpoints": missing_mutations,
        "missing_phenotype_endpoints": missing_phenotypes,
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="OpenTargets evidence_eva parquet file or directory")
    parser.add_argument("--output-kg-root", required=True, help="Local/temp KG root to write")
    parser.add_argument(
        "--endpoint-kg-root",
        help="Optional read-only KG root containing canonical mutation/phenotype node universes",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    counts = build_local_mutation_associated_phenotype(
        args.source,
        args.output_kg_root,
        endpoint_kg_root=args.endpoint_kg_root,
    )
    print(json.dumps(counts, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
