"""Stage a reproducible ClinicalTrials.gov evidence/text-feature production candidate.

This builder scales the accepted t_f65a077a prototype from a small live-API sample
into a frozen raw-response snapshot under artifacts/staged/<task-id>/raw plus
validated sidecar artifacts. It is staging-only: it never writes canonical KG/GCS
outputs and does not create trial nodes by default.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.feature_extraction.text import HashingVectorizer

try:
    from .kg_evidence import evidence_schema
    from .stage_clinicaltrials_gov_evidence_layer import (
        classify_trial_outcome,
        extract_nct_ids,
        flatten_study,
    )
except ImportError:  # pragma: no cover
    from kg_evidence import evidence_schema  # type: ignore
    from stage_clinicaltrials_gov_evidence_layer import (  # type: ignore
        classify_trial_outcome,
        extract_nct_ids,
        flatten_study,
    )

TASK_ID = "t_c4f67957"
CREATED_AT = datetime.now(timezone.utc).isoformat()
CTGOV_API_BASE = "https://clinicaltrials.gov/api/v2/studies"
SOURCE_DATASET = "ClinicalTrials.gov API v2 raw query.id snapshot"
TEXT_FEATURE_EXTRACTION_VERSION = "clinical_trials_gov_trial_text_features_v2_production_candidate"
TEXT_EMBEDDING_MODEL = "sklearn.feature_extraction.text.HashingVectorizer"
TEXT_EMBEDDING_VERSION = "scikit-learn-hashing-vectorizer@1.8.0+clinical_trials_gov_trial_text_production_candidate_v1"
TEXT_EMBEDDING_DIM = 384
DEFAULT_OT_EVIDENCE = Path(
    "/Users/jkobject/mnt/gcs/jouvencekb/staging/"
    "opentargets-clinical-drug-evidence-20260622-t_ceee5d53/"
    "evidence/molecule_treats_disease.parquet"
)
DEFAULT_KG_ROOT = Path("/Users/jkobject/mnt/gcs/jouvencekb/main")
DEFAULT_OUT_ROOT = Path(f"artifacts/staged/{TASK_ID}_clinical_trials_gov_production_candidate")
DEFAULT_USER_AGENT = "Jouvence-KG-clinical-trials-production-candidate/0.2"
LICENSE_TEXT = "ClinicalTrials.gov/NLM public API; staging-only, verify redistribution/attribution terms before canonical promotion"


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def _chunks(items: list[str], size: int) -> Iterable[list[str]]:
    if size <= 0:
        raise ValueError("batch size must be positive")
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _source_release(raw_manifest_checksum: str) -> str:
    return f"{SOURCE_DATASET}; task={TASK_ID}; fetched_at={CREATED_AT}; raw_manifest_sha256={raw_manifest_checksum}"


def load_source_edge_trials(
    ot_evidence_path: Path,
    *,
    max_edges: int | None = None,
    max_ncts_per_edge: int | None = None,
    max_studies: int | None = None,
) -> pd.DataFrame:
    """Expand OpenTargets-staged molecule_treats_disease clinical evidence to edge-NCT links."""
    limit_sql = "" if not max_edges else " LIMIT ?"
    params: list[Any] = [ot_evidence_path.as_posix()]
    if max_edges:
        params.append(int(max_edges))
    con = duckdb.connect()
    df = con.execute(
        """
        SELECT edge_key, relation, x_id, x_type, y_id, y_type, predicate, study_id, text_span,
               source, source_dataset, source_record_id, evidence_score, release
        FROM read_parquet(?)
        WHERE study_id IS NOT NULL AND study_id <> ''
        ORDER BY predicate, edge_key, study_id
        """
        + limit_sql,
        params,
    ).fetchdf()
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        nct_ids = extract_nct_ids(row["study_id"])
        if max_ncts_per_edge:
            nct_ids = nct_ids[:max_ncts_per_edge]
        for nct_id in nct_ids:
            rec = row.to_dict()
            rec["nct_id"] = nct_id
            rec["mapping_source"] = "OpenTargets staged molecule_treats_disease clinicalReportIds"
            rec["mapping_method"] = "source edge_key + ClinicalTrials.gov NCT extracted from OpenTargets study_id/clinicalReportIds"
            rec["mapping_confidence"] = "source_asserted_edge_nct_reference"
            rec["source_study_id_raw"] = str(row["study_id"])
            rec["source_text_span_hash"] = _sha256_text(str(row.get("text_span") or ""))
            rows.append(rec)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.drop_duplicates(["edge_key", "nct_id"]).sort_values(["edge_key", "nct_id"]).reset_index(drop=True)
    if max_studies:
        selected = list(dict.fromkeys(out["nct_id"].tolist()))[: int(max_studies)]
        out = out[out["nct_id"].isin(selected)].reset_index(drop=True)
    return out


def fetch_study_chunk(
    nct_ids: list[str],
    *,
    user_agent: str = DEFAULT_USER_AGENT,
    timeout: int = 60,
) -> dict[str, Any]:
    query = ",".join(nct_ids)
    params = urllib.parse.urlencode({"query.id": query, "pageSize": str(max(1, len(nct_ids))), "format": "json"})
    url = f"{CTGOV_API_BASE}?{params}"
    request = urllib.request.Request(url, headers={"User-Agent": user_agent})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    payload["_txgnn_request"] = {
        "url": url,
        "nct_ids": nct_ids,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "user_agent": user_agent,
    }
    return payload


def fetch_or_load_raw_snapshot(
    nct_ids: list[str],
    raw_root: Path,
    *,
    batch_size: int = 100,
    sleep_seconds: float = 0.1,
    refresh: bool = False,
) -> tuple[dict[str, dict[str, Any]], dict[str, str], dict[str, Any]]:
    """Fetch/freeze CTGov API v2 responses and return studies by NCT plus manifest."""
    raw_root.mkdir(parents=True, exist_ok=True)
    chunk_root = raw_root / "ctgov_api_v2_query_id_chunks"
    chunk_root.mkdir(parents=True, exist_ok=True)
    studies_by_nct: dict[str, dict[str, Any]] = {}
    fetch_errors: dict[str, str] = {}
    chunk_manifest: list[dict[str, Any]] = []

    for chunk_index, chunk_ids in enumerate(_chunks(nct_ids, batch_size)):
        chunk_path = chunk_root / f"chunk_{chunk_index:05d}.json"
        if chunk_path.exists() and not refresh:
            try:
                payload = json.loads(chunk_path.read_text())
            except json.JSONDecodeError as exc:
                fetch_errors[",".join(chunk_ids)] = f"JSONDecodeError cached {chunk_path}: {exc}"
                continue
        else:
            try:
                payload = fetch_study_chunk(chunk_ids)
                _write_json(chunk_path, payload)
                if sleep_seconds:
                    time.sleep(sleep_seconds)
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
                fetch_errors[",".join(chunk_ids)] = f"{type(exc).__name__}: {exc}"
                continue
        returned_ids: list[str] = []
        for study in payload.get("studies", []) or []:
            nct_id = (
                study.get("protocolSection", {})
                .get("identificationModule", {})
                .get("nctId", "")
            )
            nct_id = str(nct_id).upper()
            if not re.fullmatch(r"NCT\d{8}", nct_id):
                continue
            returned_ids.append(nct_id)
            studies_by_nct[nct_id] = study
        missing = sorted(set(chunk_ids) - set(returned_ids))
        for nct_id in missing:
            fetch_errors[nct_id] = "not returned by ClinicalTrials.gov API v2 query.id chunk"
        chunk_manifest.append(
            {
                "chunk_index": chunk_index,
                "path": str(chunk_path),
                "requested_nct_ids": chunk_ids,
                "returned_nct_ids": sorted(returned_ids),
                "missing_nct_ids": missing,
                "sha256": _sha256_file(chunk_path) if chunk_path.exists() else "",
                "bytes": chunk_path.stat().st_size if chunk_path.exists() else 0,
            }
        )

    manifest_without_checksum: dict[str, Any] = {
        "task": TASK_ID,
        "created_at": CREATED_AT,
        "source_dataset": SOURCE_DATASET,
        "source_url": CTGOV_API_BASE,
        "snapshot_strategy": "ClinicalTrials.gov API v2 query.id chunks frozen as raw JSON because no AACT snapshot URL was selected for this build card",
        "requested_nct_count": len(nct_ids),
        "fetched_nct_count": len(studies_by_nct),
        "fetch_error_count": len(fetch_errors),
        "requested_nct_ids_sha256": _sha256_text("\n".join(nct_ids) + "\n"),
        "chunk_count": len(chunk_manifest),
        "chunks": chunk_manifest,
        "fetch_errors": fetch_errors,
        "license": LICENSE_TEXT,
    }
    manifest_path = raw_root / "ctgov_api_v2_raw_manifest.json"
    # A file cannot contain the SHA-256 of its own final bytes without a fixed-point
    # construction. Record the checksum of the canonical manifest payload excluding
    # the checksum/path fields; chunk-level raw JSON files carry direct file hashes.
    manifest_checksum = _sha256_text(_stable_json(manifest_without_checksum))
    manifest = {
        **manifest_without_checksum,
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_checksum,
        "manifest_sha256_semantics": "sha256 of canonical manifest payload excluding manifest_path/manifest_sha256 fields",
    }
    _write_json(manifest_path, manifest)
    return studies_by_nct, fetch_errors, manifest


def build_trial_index(studies_by_nct: dict[str, dict[str, Any]], nct_ids: list[str], source_release: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for nct_id in nct_ids:
        study = studies_by_nct.get(nct_id)
        if not study:
            continue
        row = flatten_study(study, nct_id)
        row["fetched_at"] = CREATED_AT
        row["source_dataset"] = SOURCE_DATASET
        row["source_release"] = source_release
        row["source_record_id"] = f"ClinicalTrials.gov:{nct_id}"
        row["raw_response_sha256"] = _sha256_text(_stable_json(study))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("nct_id").reset_index(drop=True) if rows else pd.DataFrame()


def _trial_text_feature_row(row: pd.Series, source_release: str) -> dict[str, Any]:
    text_fields = [
        "brief_title",
        "official_title",
        "brief_summary",
        "detailed_description",
        "conditions",
        "intervention_details",
        "primary_outcome_text",
        "secondary_outcome_text",
        "eligibility_criteria",
        "why_stopped",
        "result_summary_text",
    ]
    payload_parts = []
    inventory = []
    for field in text_fields:
        value = "" if pd.isna(row.get(field)) else str(row.get(field) or "").strip()
        if value:
            payload_parts.append(f"{field}: {value}")
            inventory.append(field)
    source_text = "\n".join(payload_parts)
    return {
        "nct_id": str(row["nct_id"]),
        "source_feature_key": str(row["nct_id"]),
        "source_text": source_text,
        "source_field_inventory": _stable_json(inventory),
        "source_text_hash": _sha256_text(source_text),
        "source_text_length": len(source_text),
        "brief_title": str(row.get("brief_title") or ""),
        "official_title": str(row.get("official_title") or ""),
        "brief_summary": str(row.get("brief_summary") or ""),
        "detailed_description": str(row.get("detailed_description") or ""),
        "condition_text": str(row.get("conditions") or ""),
        "intervention_text": str(row.get("intervention_details") or ""),
        "primary_outcome_text": str(row.get("primary_outcome_text") or ""),
        "secondary_outcome_text": str(row.get("secondary_outcome_text") or ""),
        "eligibility_criteria": str(row.get("eligibility_criteria") or ""),
        "why_stopped": str(row.get("why_stopped") or ""),
        "result_summary_text": str(row.get("result_summary_text") or ""),
        "extraction_version": TEXT_FEATURE_EXTRACTION_VERSION,
        "source_dataset": SOURCE_DATASET,
        "source_release": source_release,
        "license": LICENSE_TEXT,
        "created_at": CREATED_AT,
    }


def build_trial_text_features(trial_index: pd.DataFrame, source_release: str) -> pd.DataFrame:
    if trial_index.empty:
        return pd.DataFrame(columns=["nct_id", "source_feature_key", "source_text", "source_field_inventory", "source_text_hash"])
    return pd.DataFrame([_trial_text_feature_row(row, source_release) for _, row in trial_index.sort_values("nct_id").iterrows()])


def _hashing_vectorizer() -> HashingVectorizer:
    return HashingVectorizer(
        n_features=TEXT_EMBEDDING_DIM,
        alternate_sign=False,
        norm="l2",
        lowercase=True,
        analyzer="word",
        ngram_range=(1, 2),
        token_pattern=r"(?u)\b\w\w+\b",
        dtype=np.float32,
    )


def build_trial_text_embeddings(text_features: pd.DataFrame, source_release: str) -> pd.DataFrame:
    columns = [
        "embedding_key",
        "nct_id",
        "source_feature_table",
        "source_feature_key",
        "source_feature_hash",
        "modality",
        "embedding_model",
        "embedding_version",
        "embedding_dim",
        "embedding_dtype",
        "embedding_format",
        "embedding",
        "pooling",
        "normalization",
        "preprocessing",
        "input_length",
        "window_count",
        "created_at",
        "source_feature_release",
        "provenance",
        "license",
        "citation",
    ]
    if text_features.empty:
        return pd.DataFrame(columns=columns)
    payloads = text_features["source_text"].fillna("").astype(str).tolist()
    vectors = _hashing_vectorizer().transform(payloads).astype(np.float32).toarray()
    rows: list[dict[str, Any]] = []
    for (_, row), payload, vector in zip(text_features.iterrows(), payloads, vectors, strict=True):
        source_hash = str(row["source_text_hash"])
        rows.append(
            {
                "embedding_key": f"{row['nct_id']}|{TEXT_EMBEDDING_MODEL}|{TEXT_EMBEDDING_VERSION}|{source_hash}|l2_hashing",
                "nct_id": str(row["nct_id"]),
                "source_feature_table": "features/clinical_trials_gov_trial_text_features.parquet",
                "source_feature_key": str(row["source_feature_key"]),
                "source_feature_hash": source_hash,
                "modality": "clinical_trial_free_text",
                "embedding_model": TEXT_EMBEDDING_MODEL,
                "embedding_version": TEXT_EMBEDDING_VERSION,
                "embedding_dim": TEXT_EMBEDDING_DIM,
                "embedding_dtype": "float32",
                "embedding_format": "list_float32",
                "embedding": vector.astype(np.float32).tolist(),
                "pooling": "hashing_vectorizer_l2",
                "normalization": "l2",
                "preprocessing": _stable_json({"serializer": TEXT_FEATURE_EXTRACTION_VERSION, "fields": json.loads(row["source_field_inventory"])}),
                "input_length": len(payload),
                "window_count": 1,
                "created_at": CREATED_AT,
                "source_feature_release": source_release,
                "provenance": _stable_json({"task_id": TASK_ID, "artifact": "clinical_trials_gov_trial_text_features"}),
                "license": LICENSE_TEXT,
                "citation": "scikit-learn HashingVectorizer documentation; ClinicalTrials.gov API v2",
            }
        )
    return pd.DataFrame(rows, columns=columns)


def build_evidence_rows(edge_trials: pd.DataFrame, trial_index: pd.DataFrame, source_release: str) -> pd.DataFrame:
    joined = edge_trials.merge(trial_index, on="nct_id", how="inner", suffixes=("_seed", ""))
    rows: list[dict[str, Any]] = []
    for _, row in joined.iterrows():
        text_span = (
            f"nct_id={row['nct_id']} | status={row['overall_status']} | phase={row['phase']} | "
            f"trajectory={row['trajectory_class']} | sponsor={row['lead_sponsor']} | "
            f"conditions={row['conditions']} | interventions={row['intervention_names']} | "
            f"primary_endpoints={row['primary_endpoints']} | why_stopped={row['why_stopped']} | "
            f"mapping_confidence={row.get('mapping_confidence', '')}"
        )
        rows.append(
            {
                "edge_key": row["edge_key"],
                "relation": "molecule_treats_disease",
                "x_id": row["x_id"],
                "x_type": "molecule",
                "y_id": row["y_id"],
                "y_type": "disease",
                "evidence_type": "clinical_trial_metadata",
                "source": "ClinicalTrials.gov",
                "source_dataset": SOURCE_DATASET,
                "source_record_id": f"ClinicalTrials.gov:{row['nct_id']}",
                "paper_id": row.get("publication_ids") or "",
                "dataset_id": "ClinicalTrials.gov",
                "study_id": row["nct_id"],
                "evidence_score": None,
                "effect_size": None,
                "p_value": None,
                "direction": row["trajectory_class"],
                "confidence_interval": "",
                "predicate": f"clinical trial metadata; {row['trajectory_class']}",
                "text_span": text_span[:8000],
                "section": "protocolSection/resultsSection when present",
                "extraction_method": "clinicaltrials_gov_api_v2_snapshot_joined_from_opentargets_clinicalReportIds_nct_id",
                "license": LICENSE_TEXT,
                "release": source_release,
                "created_at": CREATED_AT,
            }
        )
    evidence = pd.DataFrame(rows)
    for field in evidence_schema():
        if field.name not in evidence.columns:
            evidence[field.name] = None
    return evidence[[field.name for field in evidence_schema()]]


def _write_parquet(df: pd.DataFrame, path: Path, schema: pa.Schema | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, schema=schema, preserve_index=False) if schema is not None else pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, path)


def validate_artifacts(
    *,
    kg_root: Path,
    trial_index: pd.DataFrame,
    edge_trials: pd.DataFrame,
    evidence: pd.DataFrame,
    text_features: pd.DataFrame,
    text_embeddings: pd.DataFrame,
    source_manifest: dict[str, Any],
    out_root: Path,
) -> dict[str, Any]:
    con = duckdb.connect()
    edge_path = kg_root / "edges" / "molecule_treats_disease.parquet"
    molecule_path = kg_root / "nodes" / "molecule.parquet"
    disease_path = kg_root / "nodes" / "disease.parquet"
    link_path = out_root / "metadata" / "molecule_disease_trial_links.parquet"
    evidence_path = out_root / "evidence" / "molecule_treats_disease.clinical_trials_gov.production_candidate.parquet"

    def scalar(sql: str, params: list[Any] | None = None) -> Any:
        return con.execute(sql, params or []).fetchone()[0]

    validation: dict[str, Any] = {
        "task": TASK_ID,
        "created_at": CREATED_AT,
        "status_label": "staged production candidate",
        "source_snapshot": {
            "manifest_path": source_manifest.get("manifest_path"),
            "manifest_sha256": source_manifest.get("manifest_sha256"),
            "chunk_count": source_manifest.get("chunk_count"),
            "requested_nct_count": source_manifest.get("requested_nct_count"),
            "fetched_nct_count": source_manifest.get("fetched_nct_count"),
            "fetch_error_count": source_manifest.get("fetch_error_count"),
        },
        "counts": {
            "trial_rows": int(len(trial_index)),
            "link_rows": int(len(edge_trials)),
            "evidence_rows": int(len(evidence)),
            "trial_text_feature_rows": int(len(text_features)),
            "trial_text_embedding_rows": int(len(text_embeddings)),
            "distinct_link_edge_keys": int(edge_trials["edge_key"].nunique()) if len(edge_trials) else 0,
            "distinct_link_nct_ids": int(edge_trials["nct_id"].nunique()) if len(edge_trials) else 0,
        },
        "checks": {},
    }
    checks = validation["checks"]
    checks["duplicate_trial_index_nct_ids"] = int(trial_index.duplicated(["nct_id"]).sum()) if len(trial_index) else 0
    checks["duplicate_link_edge_nct_ids"] = int(edge_trials.duplicated(["edge_key", "nct_id"]).sum()) if len(edge_trials) else 0
    checks["duplicate_evidence_edge_nct_ids"] = int(evidence.duplicated(["edge_key", "study_id"]).sum()) if len(evidence) else 0
    checks["links_without_trial_index"] = int(len(set(edge_trials.get("nct_id", [])) - set(trial_index.get("nct_id", []))))
    checks["trial_index_without_link"] = int(len(set(trial_index.get("nct_id", [])) - set(edge_trials.get("nct_id", []))))
    checks["features_without_trial_index"] = int(len(set(text_features.get("nct_id", [])) - set(trial_index.get("nct_id", []))))
    checks["embeddings_without_features"] = int(len(set(text_embeddings.get("nct_id", [])) - set(text_features.get("nct_id", []))))
    checks["feature_rows_match_trial_rows"] = len(text_features) == len(trial_index)
    checks["embedding_rows_match_feature_rows"] = len(text_embeddings) == len(text_features)
    if len(text_embeddings):
        dims = sorted({int(v) for v in text_embeddings["embedding_dim"].dropna().tolist()})
        lengths = sorted({len(v) for v in text_embeddings["embedding"].tolist()})
        nonfinite = 0
        all_zero = 0
        for vector in text_embeddings["embedding"].tolist():
            arr = np.asarray(vector, dtype=np.float32)
            nonfinite += int((~np.isfinite(arr)).sum())
            all_zero += int(bool(len(arr) and np.allclose(arr, 0.0)))
        checks["trial_text_embedding_dim_values"] = dims
        checks["trial_text_embedding_vector_length_values"] = lengths
        checks["trial_text_embedding_nonfinite_values"] = nonfinite
        checks["trial_text_embedding_all_zero_rows"] = all_zero
    else:
        checks["trial_text_embedding_dim_values"] = []
        checks["trial_text_embedding_vector_length_values"] = []
        checks["trial_text_embedding_nonfinite_values"] = 0
        checks["trial_text_embedding_all_zero_rows"] = 0
    if link_path.exists():
        checks["link_x_missing_molecule_nodes"] = int(
            scalar(
                "SELECT count(*) FROM (SELECT DISTINCT CAST(x_id AS VARCHAR) id FROM read_parquet(?)) l "
                "ANTI JOIN read_parquet(?) n ON l.id = CAST(n.id AS VARCHAR)",
                [link_path.as_posix(), molecule_path.as_posix()],
            )
        )
        checks["link_y_missing_disease_nodes"] = int(
            scalar(
                "SELECT count(*) FROM (SELECT DISTINCT CAST(y_id AS VARCHAR) id FROM read_parquet(?)) l "
                "ANTI JOIN read_parquet(?) n ON l.id = CAST(n.id AS VARCHAR)",
                [link_path.as_posix(), disease_path.as_posix()],
            )
        )
        checks["links_without_canonical_edge"] = int(
            scalar(
                "SELECT count(*) FROM (SELECT DISTINCT edge_key FROM read_parquet(?)) l "
                "ANTI JOIN (SELECT DISTINCT relation || '|' || x_id || '|' || y_id AS edge_key FROM read_parquet(?)) e USING(edge_key)",
                [link_path.as_posix(), edge_path.as_posix()],
            )
        )
    if evidence_path.exists():
        checks["evidence_without_link"] = int(
            scalar(
                "SELECT count(*) FROM (SELECT DISTINCT edge_key, study_id AS nct_id FROM read_parquet(?)) e "
                "ANTI JOIN (SELECT DISTINCT edge_key, nct_id FROM read_parquet(?)) l USING(edge_key, nct_id)",
                [evidence_path.as_posix(), link_path.as_posix()],
            )
        )
        checks["links_without_evidence"] = int(
            scalar(
                "SELECT count(*) FROM (SELECT DISTINCT edge_key, nct_id FROM read_parquet(?)) l "
                "ANTI JOIN (SELECT DISTINCT edge_key, study_id AS nct_id FROM read_parquet(?)) e USING(edge_key, nct_id)",
                [link_path.as_posix(), evidence_path.as_posix()],
            )
        )
        checks["wrong_relation_rows"] = int(scalar("SELECT count(*) FROM read_parquet(?) WHERE relation <> 'molecule_treats_disease' OR x_type <> 'molecule' OR y_type <> 'disease'", [evidence_path.as_posix()]))
        checks["blank_nct_study_ids"] = int(scalar("SELECT count(*) FROM read_parquet(?) WHERE study_id IS NULL OR study_id = ''", [evidence_path.as_posix()]))
    blocking = [key for key, value in checks.items() if key.endswith(("missing_molecule_nodes", "missing_disease_nodes", "without_canonical_edge", "without_link", "without_evidence", "wrong_relation_rows", "blank_nct_study_ids")) and value not in (0, False)]
    blocking += [key for key in ("duplicate_trial_index_nct_ids", "duplicate_link_edge_nct_ids", "duplicate_evidence_edge_nct_ids") if checks.get(key) not in (0, False)]
    if checks.get("trial_text_embedding_dim_values") not in ([TEXT_EMBEDDING_DIM], []):
        blocking.append("trial_text_embedding_dim_values")
    if checks.get("trial_text_embedding_vector_length_values") not in ([TEXT_EMBEDDING_DIM], []):
        blocking.append("trial_text_embedding_vector_length_values")
    if checks.get("trial_text_embedding_nonfinite_values") not in (0, False):
        blocking.append("trial_text_embedding_nonfinite_values")
    validation["blocking_check_failures"] = blocking
    validation["passed"] = len(blocking) == 0
    return validation


def run(
    ot_evidence_path: Path,
    out_root: Path,
    *,
    kg_root: Path = DEFAULT_KG_ROOT,
    max_edges: int | None = None,
    max_ncts_per_edge: int | None = None,
    max_studies: int | None = None,
    batch_size: int = 100,
    sleep_seconds: float = 0.1,
    refresh_raw: bool = False,
) -> dict[str, Any]:
    out_root.mkdir(parents=True, exist_ok=True)
    for child in ("raw", "metadata", "evidence", "features", "reports"):
        (out_root / child).mkdir(exist_ok=True)

    edge_trials = load_source_edge_trials(
        ot_evidence_path,
        max_edges=max_edges,
        max_ncts_per_edge=max_ncts_per_edge,
        max_studies=max_studies,
    )
    selected_ncts = sorted(edge_trials["nct_id"].unique().tolist()) if len(edge_trials) else []
    studies_by_nct, fetch_errors, source_manifest = fetch_or_load_raw_snapshot(
        selected_ncts,
        out_root / "raw",
        batch_size=batch_size,
        sleep_seconds=sleep_seconds,
        refresh=refresh_raw,
    )
    source_release = _source_release(source_manifest["manifest_sha256"])
    trial_index = build_trial_index(studies_by_nct, selected_ncts, source_release)
    if len(trial_index):
        edge_trials = edge_trials[edge_trials["nct_id"].isin(trial_index["nct_id"])].reset_index(drop=True)
    else:
        edge_trials = edge_trials.iloc[0:0].copy()
    evidence = build_evidence_rows(edge_trials, trial_index, source_release)
    text_features = build_trial_text_features(trial_index, source_release)
    text_embeddings = build_trial_text_embeddings(text_features, source_release)

    trial_path = out_root / "metadata" / "clinical_trials_gov_trial_index.parquet"
    link_path = out_root / "metadata" / "molecule_disease_trial_links.parquet"
    evidence_path = out_root / "evidence" / "molecule_treats_disease.clinical_trials_gov.production_candidate.parquet"
    feature_path = out_root / "features" / "clinical_trials_gov_trial_text_features.parquet"
    embedding_path = out_root / "features" / "embeddings" / "clinical_trials_gov_trial_text" / "hashing_vectorizer" / "production_candidate_v1" / "part-000.parquet"
    validation_path = out_root / "reports" / "validation_checks.json"
    report_path = out_root / "reports" / "clinical_trials_gov_production_candidate_report.json"

    _write_parquet(trial_index, trial_path)
    _write_parquet(edge_trials, link_path)
    _write_parquet(evidence, evidence_path, schema=evidence_schema())
    _write_parquet(text_features, feature_path)
    _write_parquet(text_embeddings, embedding_path)

    validation = validate_artifacts(
        kg_root=kg_root,
        trial_index=trial_index,
        edge_trials=edge_trials,
        evidence=evidence,
        text_features=text_features,
        text_embeddings=text_embeddings,
        source_manifest=source_manifest,
        out_root=out_root,
    )
    _write_json(validation_path, validation)

    text_lengths = text_features["source_text_length"] if len(text_features) else pd.Series(dtype=int)
    report: dict[str, Any] = {
        "task": TASK_ID,
        "status_label": "staged production candidate",
        "created_at": CREATED_AT,
        "source_snapshot": source_manifest,
        "source_inputs": {
            "opentargets_staged_evidence": str(ot_evidence_path),
            "opentargets_staged_evidence_sha256": _sha256_file(ot_evidence_path) if ot_evidence_path.exists() else "missing",
            "kg_root": str(kg_root),
            "clinicaltrials_gov_api": CTGOV_API_BASE,
        },
        "bounds": {
            "max_edges": max_edges,
            "max_ncts_per_edge": max_ncts_per_edge,
            "max_studies": max_studies,
            "batch_size": batch_size,
            "bounded_reason": "none; all NCT IDs from the selected source evidence were requested" if not max_edges and not max_ncts_per_edge and not max_studies else "operator-specified bound for staging/runtime",
        },
        "counts": {
            "candidate_edge_nct_links": int(len(edge_trials)),
            "requested_nct_ids": int(len(selected_ncts)),
            "fetched_trials": int(len(trial_index)),
            "fetch_errors": int(len(fetch_errors)),
            "evidence_rows": int(len(evidence)),
            "distinct_supported_edge_keys": int(evidence["edge_key"].nunique()) if len(evidence) else 0,
            "trial_text_feature_rows": int(len(text_features)),
            "trial_text_embedding_rows": int(len(text_embeddings)),
            "trial_text_nonempty_rows": int((text_features["source_text_length"] > 0).sum()) if len(text_features) else 0,
        },
        "distributions": {
            "trajectory_class": dict(Counter(trial_index.get("trajectory_class", []))),
            "overall_status": dict(Counter(trial_index.get("overall_status", []))),
            "phase": dict(Counter(trial_index.get("phase", []))),
            "lead_sponsor_class": dict(Counter(trial_index.get("lead_sponsor_class", []))),
        },
        "text_feature_layer": {
            "status": "staged production candidate",
            "feature_table": str(feature_path),
            "embedding_table": str(embedding_path),
            "key": "nct_id/source_feature_key; joins to trial_index.nct_id and molecule_disease_trial_links.nct_id",
            "embedding_model": TEXT_EMBEDDING_MODEL,
            "embedding_version": TEXT_EMBEDDING_VERSION,
            "embedding_dim": TEXT_EMBEDDING_DIM,
            "source_text_length_min": int(text_lengths.min()) if len(text_lengths) else 0,
            "source_text_length_max": int(text_lengths.max()) if len(text_lengths) else 0,
            "source_text_length_median": float(text_lengths.median()) if len(text_lengths) else 0.0,
            "limitations": [
                "Raw source is a frozen CTGov API v2 response cache, not an AACT relational export; adequate for reproducible staging but AACT remains preferred for future full CTGov mirror builds.",
                "Only NCT IDs already referenced by current staged OpenTargets molecule_treats_disease clinical evidence are included; this is production-candidate coverage for that existing clinical-evidence seed, not a global all-CTGov all-drug/disease map.",
                "HashingVectorizer embeddings are deterministic local surrogate text features for key/schema validation, not foundation clinical text embeddings.",
                "CTGov completion/status is not efficacy success; failed/terminated/unknown trajectory classes are evidence features, not naive negative edges.",
            ],
        },
        "artifacts": {
            "raw_manifest": str(out_root / "raw" / "ctgov_api_v2_raw_manifest.json"),
            "trial_index": str(trial_path),
            "edge_trial_links": str(link_path),
            "evidence": str(evidence_path),
            "trial_text_features": str(feature_path),
            "trial_text_embeddings": str(embedding_path),
            "validation": str(validation_path),
            "report": str(report_path),
        },
        "validation_summary": {
            "passed": validation["passed"],
            "blocking_check_failures": validation["blocking_check_failures"],
            "checks": validation["checks"],
        },
        "schema_decision": "Keep ClinicalTrials.gov studies as evidence/metadata/text-feature sidecars keyed by NCT ID and supported molecule_treats_disease edge_key; do not add trial nodes or clinical failure negative edges by default.",
        "training_recommendation": "Use trial phase/status/trajectory/endpoint/sponsor/enrollment/date/text embeddings as leakage-aware evidence features or edge weights; do not convert failed, terminated, withdrawn, or unknown trials into direct negative biomedical graph edges without task-specific policy and held-out-label leakage controls.",
        "rerun_command": (
            "uv run python -m manage_db.stage_clinicaltrials_gov_production_candidate "
            f"--out-root {out_root} --opentargets-evidence {ot_evidence_path} --kg-root {kg_root} --batch-size {batch_size}"
        ),
    }
    _write_json(report_path, report)
    return report


def _none_if_nonpositive(value: int) -> int | None:
    return None if value <= 0 else value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--opentargets-evidence", type=Path, default=DEFAULT_OT_EVIDENCE)
    parser.add_argument("--kg-root", type=Path, default=DEFAULT_KG_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--max-edges", type=int, default=0, help="0 means all source evidence rows")
    parser.add_argument("--max-ncts-per-edge", type=int, default=0, help="0 means all NCTs per edge")
    parser.add_argument("--max-studies", type=int, default=0, help="0 means all selected NCT IDs")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--sleep-seconds", type=float, default=0.1)
    parser.add_argument("--refresh-raw", action="store_true")
    args = parser.parse_args()
    report = run(
        args.opentargets_evidence,
        args.out_root,
        kg_root=args.kg_root,
        max_edges=_none_if_nonpositive(args.max_edges),
        max_ncts_per_edge=_none_if_nonpositive(args.max_ncts_per_edge),
        max_studies=_none_if_nonpositive(args.max_studies),
        batch_size=args.batch_size,
        sleep_seconds=args.sleep_seconds,
        refresh_raw=args.refresh_raw,
    )
    print(json.dumps(report, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
