"""Stage a bounded ClinicalTrials.gov trial-metadata evidence layer.

This prototype enriches existing molecule_treats_disease evidence with trial-level
ClinicalTrials.gov metadata. It is intentionally staging-only: trial records are
kept as evidence/metadata sidecars, not new default graph nodes or topology.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.feature_extraction.text import HashingVectorizer

try:
    from .kg_evidence import evidence_schema
except ImportError:  # pragma: no cover
    from kg_evidence import evidence_schema  # type: ignore

CREATED_AT = datetime.now(timezone.utc).isoformat()
CTGOV_API_BASE = "https://clinicaltrials.gov/api/v2/studies"
SOURCE_DATASET = "ClinicalTrials.gov API v2"
SOURCE_RELEASE = "live API fetch; fetched_at=" + CREATED_AT
TEXT_FEATURE_EXTRACTION_VERSION = "clinical_trials_gov_trial_text_features_v1"
TEXT_EMBEDDING_MODEL = "sklearn.feature_extraction.text.HashingVectorizer"
TEXT_EMBEDDING_VERSION = "scikit-learn-hashing-vectorizer@1.8.0+clinical_trials_gov_trial_text_prototype_v1"
TEXT_EMBEDDING_DIM = 384
DEFAULT_OT_EVIDENCE = Path(
    "/Users/jkobject/mnt/gcs/jouvencekb/staging/"
    "opentargets-clinical-drug-evidence-20260622-t_ceee5d53/"
    "evidence/molecule_treats_disease.parquet"
)
DEFAULT_KG_ROOT = Path("/Users/jkobject/mnt/gcs/jouvencekb/main")

NCT_RE = re.compile(r"NCT\d{8}", re.IGNORECASE)
SAFETY_TERMS = ("safety", "adverse", "toxicity", "toxic", "death", "mortality", "harm")
BUSINESS_TERMS = (
    "business",
    "commercial",
    "strategic",
    "portfolio",
    "sponsor",
    "funding",
    "financial",
    "resource",
    "administrative",
    "recruitment",
    "enrollment",
    "accrual",
    "slow",
    "feasibility",
)
EFFICACY_FAILURE_TERMS = (
    "efficacy",
    "futility",
    "lack of effect",
    "lack of efficacy",
    "ineffective",
    "not effective",
    "no benefit",
    "endpoint",
    "primary endpoint",
    "failed",
)


def extract_nct_ids(value: object) -> list[str]:
    """Return uppercase NCT IDs from a semicolon/text field, preserving order."""
    seen: set[str] = set()
    out: list[str] = []
    for match in NCT_RE.findall("" if value is None else str(value)):
        nct = match.upper()
        if nct not in seen:
            seen.add(nct)
            out.append(nct)
    return out


def classify_trial_outcome(overall_status: str | None, why_stopped: str | None) -> str:
    """Conservative trial trajectory class from CTGov status and stop reason.

    ClinicalTrials.gov does not encode a universal efficacy success/failure flag;
    this classifier only calls explicit safety/business/efficacy-failure cases and
    otherwise leaves completed/recruiting/unknown rows as unknown outcome classes.
    """
    status = (overall_status or "").strip().upper().replace(" ", "_")
    reason = (why_stopped or "").strip().lower()
    if reason and any(term in reason for term in SAFETY_TERMS):
        return "safety_failure_or_harm"
    if reason and any(term in reason for term in BUSINESS_TERMS):
        return "terminated_business_funding_or_feasibility"
    if reason and any(term in reason for term in EFFICACY_FAILURE_TERMS):
        return "failed_efficacy_or_endpoint"
    if status in {"TERMINATED", "SUSPENDED"}:
        return "terminated_unknown_reason"
    if status == "WITHDRAWN":
        return "withdrawn_unknown_reason"
    if status in {"RECRUITING", "ENROLLING_BY_INVITATION", "NOT_YET_RECRUITING", "ACTIVE_NOT_RECRUITING"}:
        return "active_or_recruiting_unknown_outcome"
    if status == "COMPLETED":
        return "completed_outcome_unknown"
    return "unknown_trial_trajectory"


def _first_date(module: dict[str, Any], key: str) -> str:
    value = module.get(key) or {}
    if isinstance(value, dict):
        return str(value.get("date") or "")
    return ""


def _join(values: list[str], limit: int = 12) -> str:
    return "; ".join([v for v in values if v][:limit])


def _clean_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value)
    if text.lower() in {"nan", "none", "<na>"}:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def _pick_interventions(arms_module: dict[str, Any]) -> tuple[str, str, str]:
    interventions = arms_module.get("interventions") or []
    names: list[str] = []
    types: list[str] = []
    details: list[str] = []
    for item in interventions:
        if not isinstance(item, dict):
            continue
        name = _clean_text(item.get("name"))
        description = _clean_text(item.get("description"))
        item_type = _clean_text(item.get("type"))
        names.append(name)
        types.append(item_type)
        parts = []
        if item_type:
            parts.append(f"type={item_type}")
        if name:
            parts.append(f"name={name}")
        if description:
            parts.append(f"description={description}")
        if parts:
            details.append(" | ".join(parts))
    return _join(names, 20), _join(sorted(set(t for t in types if t)), 20), _join(details, 20)


def _pick_design(design_module: dict[str, Any]) -> tuple[str, str, str]:
    phases = design_module.get("phases") or []
    if isinstance(phases, str):
        phases = [phases]
    design = design_module.get("designInfo") or {}
    allocation = design.get("allocation", "") if isinstance(design, dict) else ""
    masking = design.get("maskingInfo", {}).get("masking", "") if isinstance(design, dict) else ""
    return _join([str(p) for p in phases], 8), str(allocation or ""), str(masking or "")


def _outcome_text(outcomes: list[dict[str, Any]]) -> str:
    details: list[str] = []
    for outcome in outcomes:
        if not isinstance(outcome, dict):
            continue
        parts = []
        for key, label in (("measure", "measure"), ("description", "description"), ("timeFrame", "time_frame")):
            value = _clean_text(outcome.get(key))
            if value:
                parts.append(f"{label}={value}")
        if parts:
            details.append(" | ".join(parts))
    return _join(details, 12)


def _pick_outcomes(outcomes_module: dict[str, Any]) -> tuple[str, str, str, str, int, int]:
    primary = outcomes_module.get("primaryOutcomes") or []
    secondary = outcomes_module.get("secondaryOutcomes") or []
    primary_text = _join(
        [str(o.get("measure") or "") for o in primary if isinstance(o, dict)],
        8,
    )
    secondary_text = _join(
        [str(o.get("measure") or "") for o in secondary if isinstance(o, dict)],
        8,
    )
    return primary_text, secondary_text, _outcome_text(primary), _outcome_text(secondary), len(primary), len(secondary)


def _pick_result_summary(results_section: dict[str, Any]) -> str:
    """Best-effort API v2 results free text when results are available.

    ClinicalTrials.gov API v2 result modules are not uniformly populated for the
    bounded OpenTargets-seeded prototype. Preserve available text, while the
    source field inventory records when no results fields were present.
    """
    if not isinstance(results_section, dict) or not results_section:
        return ""
    texts: list[str] = []
    outcome_measures = (results_section.get("outcomeMeasuresModule") or {}).get("outcomeMeasures") or []
    for outcome in outcome_measures:
        if not isinstance(outcome, dict):
            continue
        parts = []
        for key, label in (("title", "title"), ("description", "description"), ("timeFrame", "time_frame"), ("type", "type")):
            value = _clean_text(outcome.get(key))
            if value:
                parts.append(f"{label}={value}")
        if parts:
            texts.append(" | ".join(parts))
    adverse_events = results_section.get("adverseEventsModule") or {}
    for key in ("frequencyThreshold", "timeFrame", "description"):
        value = _clean_text(adverse_events.get(key))
        if value:
            texts.append(f"adverse_events_{key}={value}")
    more_info = results_section.get("moreInfoModule") or {}
    for key in ("limitationsAndCaveats", "certainAgreement"):
        value = _clean_text(more_info.get(key))
        if value:
            texts.append(f"results_{key}={value}")
    return _join(texts, 20)


def _pick_references(refs_module: dict[str, Any]) -> str:
    refs = refs_module.get("references") or []
    pmids = []
    for ref in refs:
        if isinstance(ref, dict) and ref.get("pmid"):
            pmids.append("PMID:" + str(ref["pmid"]))
    return ";".join(pmids[:20])


def fetch_study(nct_id: str, *, sleep_seconds: float = 0.05) -> dict[str, Any]:
    url = f"{CTGOV_API_BASE}/{urllib.parse.quote(nct_id)}?format=json"
    req = urllib.request.Request(url, headers={"User-Agent": "Jouvence-KG-prototype/0.1"})
    with urllib.request.urlopen(req, timeout=30) as response:
        data = json.load(response)
    if sleep_seconds:
        time.sleep(sleep_seconds)
    return data


def flatten_study(study: dict[str, Any], nct_id: str) -> dict[str, Any]:
    protocol = study.get("protocolSection") or {}
    ident = protocol.get("identificationModule") or {}
    status = protocol.get("statusModule") or {}
    sponsor = protocol.get("sponsorCollaboratorsModule") or {}
    design = protocol.get("designModule") or {}
    arms = protocol.get("armsInterventionsModule") or {}
    outcomes = protocol.get("outcomesModule") or {}
    conditions = protocol.get("conditionsModule") or {}
    description = protocol.get("descriptionModule") or {}
    eligibility = protocol.get("eligibilityModule") or {}
    refs = protocol.get("referencesModule") or {}
    results = study.get("resultsSection") or {}

    phases, allocation, masking = _pick_design(design)
    intervention_names, intervention_types, intervention_details = _pick_interventions(arms)
    primary, secondary, primary_details, secondary_details, n_primary, n_secondary = _pick_outcomes(outcomes)
    overall_status = str(status.get("overallStatus") or "")
    why_stopped = str(status.get("whyStopped") or "")
    lead = sponsor.get("leadSponsor") or {}
    collaborators = sponsor.get("collaborators") or []
    enrollment = design.get("enrollmentInfo") or {}

    return {
        "nct_id": nct_id,
        "brief_title": str(ident.get("briefTitle") or ""),
        "official_title": str(ident.get("officialTitle") or ""),
        "overall_status": overall_status,
        "why_stopped": why_stopped,
        "trajectory_class": classify_trial_outcome(overall_status, why_stopped),
        "phase": phases,
        "study_type": str(design.get("studyType") or ""),
        "allocation": allocation,
        "masking": masking,
        "enrollment_count": enrollment.get("count"),
        "enrollment_type": enrollment.get("type"),
        "start_date": _first_date(status, "startDateStruct"),
        "primary_completion_date": _first_date(status, "primaryCompletionDateStruct"),
        "completion_date": _first_date(status, "completionDateStruct"),
        "conditions": _join([str(c) for c in (conditions.get("conditions") or [])], 20),
        "brief_summary": _clean_text(description.get("briefSummary")),
        "detailed_description": _clean_text(description.get("detailedDescription")),
        "intervention_names": intervention_names,
        "intervention_types": intervention_types,
        "intervention_details": intervention_details,
        "primary_endpoints": primary,
        "secondary_endpoints": secondary,
        "primary_outcome_text": primary_details,
        "secondary_outcome_text": secondary_details,
        "primary_endpoint_count": n_primary,
        "secondary_endpoint_count": n_secondary,
        "eligibility_criteria": _clean_text(eligibility.get("eligibilityCriteria")),
        "result_summary_text": _pick_result_summary(results),
        "lead_sponsor": str(lead.get("name") or ""),
        "lead_sponsor_class": str(lead.get("class") or ""),
        "collaborators": _join([str(c.get("name") or "") for c in collaborators if isinstance(c, dict)], 12),
        "publication_ids": _pick_references(refs),
        "source_url": f"https://clinicaltrials.gov/study/{nct_id}",
        "fetched_at": CREATED_AT,
    }


def _load_source_edges(ot_evidence_path: Path, max_edges: int, max_ncts_per_edge: int) -> pd.DataFrame:
    con = duckdb.connect()
    df = con.execute(
        """
        SELECT edge_key, relation, x_id, x_type, y_id, y_type, predicate, study_id, text_span
        FROM read_parquet(?)
        WHERE study_id IS NOT NULL AND study_id <> ''
        ORDER BY predicate, edge_key
        LIMIT ?
        """,
        [ot_evidence_path.as_posix(), max_edges],
    ).fetchdf()
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        for nct_id in extract_nct_ids(row["study_id"])[:max_ncts_per_edge]:
            rec = row.to_dict()
            rec["nct_id"] = nct_id
            rows.append(rec)
    return pd.DataFrame(rows).drop_duplicates(["edge_key", "nct_id"]).reset_index(drop=True)


def _trial_text_feature_row(row: pd.Series) -> dict[str, Any]:
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
        value = _clean_text(row.get(field))
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
        "brief_title": _clean_text(row.get("brief_title")),
        "official_title": _clean_text(row.get("official_title")),
        "brief_summary": _clean_text(row.get("brief_summary")),
        "detailed_description": _clean_text(row.get("detailed_description")),
        "condition_text": _clean_text(row.get("conditions")),
        "intervention_text": _clean_text(row.get("intervention_details")),
        "primary_outcome_text": _clean_text(row.get("primary_outcome_text")),
        "secondary_outcome_text": _clean_text(row.get("secondary_outcome_text")),
        "eligibility_criteria": _clean_text(row.get("eligibility_criteria")),
        "why_stopped": _clean_text(row.get("why_stopped")),
        "result_summary_text": _clean_text(row.get("result_summary_text")),
        "extraction_version": TEXT_FEATURE_EXTRACTION_VERSION,
        "source_dataset": SOURCE_DATASET,
        "source_release": SOURCE_RELEASE,
        "license": "ClinicalTrials.gov/NLM public API; verify redistribution terms before canonical promotion",
        "created_at": CREATED_AT,
    }


def build_trial_text_features(trial_index: pd.DataFrame) -> pd.DataFrame:
    if trial_index.empty:
        return pd.DataFrame(columns=["nct_id", "source_feature_key", "source_text", "source_field_inventory", "source_text_hash"])
    return pd.DataFrame([_trial_text_feature_row(row) for _, row in trial_index.sort_values("nct_id").iterrows()])


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


def build_trial_text_embeddings(text_features: pd.DataFrame) -> pd.DataFrame:
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
                "source_feature_release": SOURCE_RELEASE,
                "provenance": _stable_json({"task_id": "t_f65a077a", "artifact": "clinical_trials_gov_trial_text_features"}),
                "license": "ClinicalTrials.gov/NLM public API; verify redistribution terms before canonical promotion",
                "citation": "scikit-learn HashingVectorizer documentation; ClinicalTrials.gov API v2",
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _to_evidence_rows(edge_trials: pd.DataFrame, trial_index: pd.DataFrame) -> pd.DataFrame:
    joined = edge_trials.merge(trial_index, on="nct_id", how="inner")
    rows: list[dict[str, Any]] = []
    for _, row in joined.iterrows():
        text_span = (
            f"nct_id={row['nct_id']} | status={row['overall_status']} | phase={row['phase']} | "
            f"trajectory={row['trajectory_class']} | sponsor={row['lead_sponsor']} | "
            f"conditions={row['conditions']} | interventions={row['intervention_names']} | "
            f"primary_endpoints={row['primary_endpoints']} | why_stopped={row['why_stopped']}"
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
                "section": "protocolSection",
                "extraction_method": "clinicaltrials_gov_api_v2_joined_from_opentargets_nct_id",
                "license": "ClinicalTrials.gov/NLM public API; verify redistribution terms before canonical promotion",
                "release": SOURCE_RELEASE,
                "created_at": CREATED_AT,
            }
        )
    evidence = pd.DataFrame(rows)
    for field in evidence_schema():
        if field.name not in evidence.columns:
            evidence[field.name] = None
    return evidence[[field.name for field in evidence_schema()]]


def run(
    ot_evidence_path: Path,
    out_root: Path,
    *,
    max_edges: int = 20,
    max_ncts_per_edge: int = 2,
    max_studies: int = 30,
) -> dict[str, Any]:
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "evidence").mkdir(exist_ok=True)
    (out_root / "features").mkdir(exist_ok=True)
    (out_root / "metadata").mkdir(exist_ok=True)
    (out_root / "reports").mkdir(exist_ok=True)

    edge_trials = _load_source_edges(ot_evidence_path, max_edges, max_ncts_per_edge)
    selected_ncts = list(dict.fromkeys(edge_trials["nct_id"].tolist()))[:max_studies]
    edge_trials = edge_trials[edge_trials["nct_id"].isin(selected_ncts)].reset_index(drop=True)

    trial_rows: list[dict[str, Any]] = []
    fetch_errors: dict[str, str] = {}
    for nct_id in selected_ncts:
        try:
            trial_rows.append(flatten_study(fetch_study(nct_id), nct_id))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError, ValueError) as exc:
            fetch_errors[nct_id] = f"{type(exc).__name__}: {exc}"

    trial_index = pd.DataFrame(trial_rows)
    evidence = _to_evidence_rows(edge_trials, trial_index) if not trial_index.empty else pd.DataFrame(columns=[f.name for f in evidence_schema()])
    text_features = build_trial_text_features(trial_index)
    text_embeddings = build_trial_text_embeddings(text_features)

    trial_path = out_root / "metadata" / "clinical_trials_gov_trial_index.parquet"
    edge_trial_path = out_root / "metadata" / "molecule_disease_trial_links.parquet"
    evidence_path = out_root / "evidence" / "molecule_treats_disease.clinical_trials_gov.prototype.parquet"
    text_features_path = out_root / "features" / "clinical_trials_gov_trial_text_features.parquet"
    text_embeddings_path = out_root / "features" / "embeddings" / "clinical_trials_gov_trial_text" / "hashing_vectorizer" / "prototype_v1" / "part-000.parquet"
    report_path = out_root / "reports" / "clinical_trials_gov_evidence_layer_report.json"

    pq.write_table(pa.Table.from_pandas(trial_index, preserve_index=False), trial_path)
    pq.write_table(pa.Table.from_pandas(edge_trials, preserve_index=False), edge_trial_path)
    pq.write_table(pa.Table.from_pandas(evidence, schema=evidence_schema(), preserve_index=False), evidence_path)
    pq.write_table(pa.Table.from_pandas(text_features, preserve_index=False), text_features_path)
    text_embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(text_embeddings, preserve_index=False), text_embeddings_path)

    trajectory_counts = Counter(trial_index.get("trajectory_class", []))
    status_counts = Counter(trial_index.get("overall_status", []))
    phase_counts = Counter(trial_index.get("phase", []))
    sponsor_class_counts = Counter(trial_index.get("lead_sponsor_class", []))
    edge_counts = defaultdict(int)
    for edge_key in edge_trials["edge_key"].tolist() if not edge_trials.empty else []:
        edge_counts[edge_key] += 1

    report = {
        "task": "t_f65a077a",
        "created_at": CREATED_AT,
        "source_inputs": {
            "opentargets_staged_evidence": str(ot_evidence_path),
            "clinicaltrials_gov_api": CTGOV_API_BASE,
        },
        "bounds": {
            "max_edges": max_edges,
            "max_ncts_per_edge": max_ncts_per_edge,
            "max_studies": max_studies,
        },
        "counts": {
            "candidate_edge_nct_links": int(len(edge_trials)),
            "requested_nct_ids": int(len(selected_ncts)),
            "fetched_trials": int(len(trial_index)),
            "fetch_errors": int(len(fetch_errors)),
            "prototype_evidence_rows": int(len(evidence)),
            "distinct_supported_edge_keys": int(evidence["edge_key"].nunique()) if len(evidence) else 0,
            "trial_text_feature_rows": int(len(text_features)),
            "trial_text_embedding_rows": int(len(text_embeddings)),
            "trial_text_nonempty_rows": int((text_features["source_text_length"] > 0).sum()) if len(text_features) else 0,
        },
        "distributions": {
            "trajectory_class": dict(trajectory_counts),
            "overall_status": dict(status_counts),
            "phase": dict(phase_counts),
            "lead_sponsor_class": dict(sponsor_class_counts),
        },
        "fetch_errors_by_nct": fetch_errors,
        "example_trial_rows": trial_index.head(5).to_dict(orient="records"),
        "example_evidence_rows": evidence.head(5).to_dict(orient="records"),
        "artifacts": {
            "trial_index": str(trial_path),
            "edge_trial_links": str(edge_trial_path),
            "evidence": str(evidence_path),
            "trial_text_features": str(text_features_path),
            "trial_text_embeddings": str(text_embeddings_path),
            "report": str(report_path),
        },
        "text_feature_layer": {
            "status": "staged-only prototype",
            "feature_table": str(text_features_path),
            "embedding_table": str(text_embeddings_path),
            "key": "nct_id/source_feature_key; joins to trial_index.nct_id and molecule_disease_trial_links.nct_id",
            "source_fields": [
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
                "result_summary_text when present in API v2",
            ],
            "embedding_model": TEXT_EMBEDDING_MODEL,
            "embedding_version": TEXT_EMBEDDING_VERSION,
            "embedding_dim": TEXT_EMBEDDING_DIM,
            "limitations": [
                "bounded live API prototype, not a frozen AACT snapshot",
                "result_summary_text is populated only when ClinicalTrials.gov API v2 exposes results modules for fetched NCT rows",
                "HashingVectorizer embeddings are deterministic local surrogate scaffold artifacts, not foundation-model biological text embeddings",
            ],
        },
        "schema_decision": "Keep ClinicalTrials.gov studies as evidence/metadata sidecars keyed by NCT ID and supported molecule_treats_disease edge_key; do not add trial nodes or negative disease edges by default.",
        "training_recommendation": "Use trial phase/status/trajectory/endpoint/sponsor fields as evidence features or edge weights with leakage-aware masking for held-out molecule_treats_disease labels; do not convert failed/terminated trials into direct negative biomedical graph edges without task-specific policy.",
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--opentargets-evidence", type=Path, default=DEFAULT_OT_EVIDENCE)
    parser.add_argument("--out-root", type=Path, default=Path("artifacts/staged/t_f65a077a_clinical_trials_evidence_layer"))
    parser.add_argument("--max-edges", type=int, default=20)
    parser.add_argument("--max-ncts-per-edge", type=int, default=2)
    parser.add_argument("--max-studies", type=int, default=30)
    args = parser.parse_args()
    report = run(
        args.opentargets_evidence,
        args.out_root,
        max_edges=args.max_edges,
        max_ncts_per_edge=args.max_ncts_per_edge,
        max_studies=args.max_studies,
    )
    print(json.dumps(report, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
