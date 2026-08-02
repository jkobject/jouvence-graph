"""Stage OpenTargets clinical evidence for legacy TxGNN drug-disease edges.

This is intentionally staging-only: it does not overwrite canonical KG files.
It supports existing canonical molecule_treats_disease edges with positive
clinical indication records from OpenTargets 26.03 clinical_indication and
audits drug_warning as a contraindication candidate without promoting it.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

try:
    from .kg_evidence import evidence_schema
except ImportError:  # pragma: no cover
    from kg_evidence import evidence_schema  # type: ignore

EVIDENCE_COLUMNS = [field.name for field in evidence_schema()]
RELEASE = "OpenTargets Platform 26.03"
SOURCE_URL = "https://ftp.ebi.ac.uk/pub/databases/opentargets/platform/26.03/output"
CREATED_AT = datetime.now(timezone.utc).isoformat()
STAGE_SCORE = {
    "APPROVAL": 1.0,
    "PREAPPROVAL": 0.95,
    "PHASE_3": 0.85,
    "PHASE_2_3": 0.80,
    "PHASE_2": 0.70,
    "PHASE_1_2": 0.55,
    "PHASE_1": 0.45,
    "EARLY_PHASE_1": 0.35,
    "IND": 0.30,
    "PRECLINICAL": 0.20,
    "UNKNOWN": 0.10,
}


def _norm_disease(value: object) -> str:
    text = "" if value is None else str(value).strip()
    if not text:
        return ""
    if text.startswith("EFO_") or text.startswith("MONDO_") or text.startswith("HP_"):
        return text.replace("_", ":", 1)
    return text


def _first_list_text(values: object, *, prefix: str | None = None, limit: int = 12) -> str:
    if values is None:
        return ""
    if not isinstance(values, (list, tuple)):
        return str(values)
    out: list[str] = []
    for v in values:
        s = "" if v is None else str(v)
        if prefix and not s.lower().startswith(prefix.lower()):
            continue
        if s:
            out.append(s)
        if len(out) >= limit:
            break
    return ";".join(out)


def _source_record_id(row: pd.Series) -> str:
    rid = str(row.get("clinical_indication_id") or "").strip()
    return f"OpenTargets:clinical_indication:{rid}" if rid else "OpenTargets:clinical_indication"


def _clinical_text_span(row: pd.Series) -> str:
    parts = [
        f"clinical_stage={row.get('maxClinicalStage') or ''}",
        f"chembl_id={row.get('chembl_id') or ''}",
        f"ot_disease_id={row.get('ot_disease_id') or ''}",
    ]
    ncts = str(row.get("nct_ids") or "")
    reports = str(row.get("clinical_report_ids") or "")
    if ncts:
        parts.append(f"nct_ids={ncts}")
    if reports:
        parts.append(f"clinical_report_ids={reports}")
    return " | ".join(parts)


def _to_evidence_frame(matches: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in matches.iterrows():
        rows.append(
            {
                "edge_key": f"molecule_treats_disease|{row['x_id']}|{row['y_id']}",
                "relation": "molecule_treats_disease",
                "x_id": row["x_id"],
                "x_type": "molecule",
                "y_id": row["y_id"],
                "y_type": "disease",
                "evidence_type": "clinical_indication",
                "source": "OpenTargets",
                "source_dataset": "clinical_indication",
                "source_record_id": _source_record_id(row),
                "paper_id": "",
                "dataset_id": "OpenTargets:26.03:clinical_indication",
                "study_id": str(row.get("nct_ids") or ""),
                "evidence_score": STAGE_SCORE.get(str(row.get("maxClinicalStage") or "UNKNOWN"), 0.1),
                "effect_size": None,
                "p_value": None,
                "direction": "positive_indication",
                "confidence_interval": "",
                "predicate": f"clinical indication; stage={row.get('maxClinicalStage') or 'UNKNOWN'}",
                "text_span": _clinical_text_span(row),
                "section": "clinical_indication",
                "extraction_method": "source_exact_edge_match_via_xref",
                "license": "OpenTargets Platform terms",
                "release": RELEASE,
                "created_at": CREATED_AT,
            }
        )
    df = pd.DataFrame(rows)
    for col in EVIDENCE_COLUMNS:
        if col not in df.columns:
            df[col] = None
    df = df[EVIDENCE_COLUMNS]
    for col in ["evidence_score", "effect_size", "p_value"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.drop_duplicates(
        subset=["relation", "x_id", "y_id", "source_dataset", "source_record_id", "study_id", "predicate"],
        keep="last",
    ).reset_index(drop=True)


def run(raw_root: Path, kg_cache: Path, out_root: Path) -> dict:
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "evidence").mkdir(exist_ok=True)
    (out_root / "reports").mkdir(exist_ok=True)

    con = duckdb.connect()
    con.execute("PRAGMA threads=2")
    con.execute("PRAGMA memory_limit='4GB'")
    con.create_function("norm_disease", _norm_disease, ["VARCHAR"], "VARCHAR")
    con.create_function("first_list_text", _first_list_text, null_handling="special")

    treat_edges = kg_cache / "edges" / "molecule_treats_disease.parquet"
    contraind_edges = kg_cache / "edges" / "molecule_contraindicates_disease.parquet"
    clinical = raw_root / "clinical_indication" / "clinical_indication.parquet"
    drug_molecule = raw_root / "drug_molecule" / "*.parquet"
    drug_warning = raw_root / "drug_warning" / "*.parquet"

    # Materialize compact mapping tables; explode crossReferences.source/ids.
    con.sql(
        f"""
        CREATE TEMP TABLE drug_xref AS
        WITH dm AS (
          SELECT id AS chembl_id, unnest(crossReferences) AS xr
          FROM read_parquet('{drug_molecule.as_posix()}')
        ), exploded AS (
          SELECT chembl_id, xr.source AS source_name, unnest(xr.ids) AS source_id
          FROM dm
        )
        SELECT DISTINCT chembl_id, source_name, source_id FROM exploded WHERE source_id IS NOT NULL
        UNION
        SELECT DISTINCT id AS chembl_id, 'ChEMBL' AS source_name, id AS source_id
        FROM read_parquet('{drug_molecule.as_posix()}')
        """
    )
    con.sql(
        """
        CREATE TEMP TABLE molecule_to_chembl AS
        SELECT DISTINCT source_id AS molecule_id, chembl_id, source_name
        FROM drug_xref
        WHERE source_id LIKE 'DB%' OR source_id LIKE 'CTD:%' OR source_id LIKE 'CHEMBL%'
        """
    )
    con.sql(
        f"""
        CREATE TEMP TABLE clinical_norm AS
        SELECT
          id AS clinical_indication_id,
          drugId AS chembl_id,
          norm_disease(diseaseId) AS disease_id,
          diseaseId AS ot_disease_id,
          maxClinicalStage,
          clinicalReportIds,
          list_filter(clinicalReportIds, x -> starts_with(lower(x), 'nct')) AS nctIds
        FROM read_parquet('{clinical.as_posix()}')
        WHERE drugId IS NOT NULL AND diseaseId IS NOT NULL
        """
    )
    con.sql(
        f"""
        CREATE TEMP TABLE treat_edges AS
        SELECT * FROM read_parquet('{treat_edges.as_posix()}')
        """
    )
    con.sql(
        """
        CREATE TEMP TABLE treat_matches AS
        SELECT
          e.x_id, e.y_id, e.relation, e.display_relation, e.source AS canonical_source,
          m.chembl_id, m.source_name AS mapping_source,
          c.clinical_indication_id, c.maxClinicalStage, c.ot_disease_id,
          array_to_string(c.clinicalReportIds, ';') AS clinical_report_ids,
          array_to_string(c.nctIds, ';') AS nct_ids
        FROM treat_edges e
        JOIN molecule_to_chembl m ON e.x_id = m.molecule_id
        JOIN clinical_norm c ON c.chembl_id = m.chembl_id AND c.disease_id = e.y_id
        """
    )
    matches = con.sql("SELECT * FROM treat_matches").fetchdf()
    evidence = _to_evidence_frame(matches)
    evidence_path = out_root / "evidence" / "molecule_treats_disease.parquet"
    pq.write_table(pa.Table.from_pandas(evidence, schema=evidence_schema(), preserve_index=False), evidence_path)

    # Contraindication candidate audit: OpenTargets drug_warning describes warnings,
    # withdrawals, toxicity classes and warning-class EFOs, not disease-specific
    # contraindication indication rows. Keep as source-decision evidence only.
    warning_summary = con.sql(
        f"""
        SELECT warningType, count(*) AS rows,
               count(efoId) AS rows_with_efoId,
               count(efoIdForWarningClass) AS rows_with_warning_class_efo,
               count(description) AS rows_with_description
        FROM read_parquet('{drug_warning.as_posix()}')
        GROUP BY warningType ORDER BY rows DESC
        """
    ).fetchdf().to_dict(orient="records")

    con.sql(f"CREATE TEMP TABLE contraind_edges AS SELECT * FROM read_parquet('{contraind_edges.as_posix()}')")
    # Potential shape-compatible overlaps by DB->ChEMBL and exact disease only; not promoted.
    contraind_warning_overlap = con.sql(
        f"""
        WITH warnings AS (
          SELECT unnest(chemblIds) AS chembl_id, norm_disease(coalesce(efoId, efoIdForWarningClass)) AS disease_id,
                 id AS warning_id, warningType, toxicityClass, country, description
          FROM read_parquet('{drug_warning.as_posix()}')
        )
        SELECT count(*) AS rows,
               count(DISTINCT e.relation || '|' || e.x_id || '|' || e.y_id) AS distinct_supported_edge_keys
        FROM contraind_edges e
        JOIN molecule_to_chembl m ON e.x_id = m.molecule_id
        JOIN warnings w ON w.chembl_id = m.chembl_id AND w.disease_id = e.y_id
        """
    ).fetchdf().iloc[0].to_dict()

    report = {
        "task": "t_ceee5d53",
        "created_at": CREATED_AT,
        "source_release": RELEASE,
        "source_url": SOURCE_URL,
        "canonical": {
            "molecule_treats_disease_edges": int(con.sql("SELECT count(*) FROM treat_edges").fetchone()[0]),
            "molecule_contraindicates_disease_edges": int(con.sql("SELECT count(*) FROM contraind_edges").fetchone()[0]),
            "existing_treat_evidence_on_gcs": False,
            "existing_contraindication_evidence_on_gcs": False,
        },
        "mapping": {
            "molecule_to_chembl_rows": int(con.sql("SELECT count(*) FROM molecule_to_chembl").fetchone()[0]),
            "clinical_indication_rows": int(con.sql("SELECT count(*) FROM clinical_norm").fetchone()[0]),
        },
        "positive_indication_evidence": {
            "source_decision": "OpenTargets 26.03 clinical_indication is positive clinical indication/trial-stage evidence and is valid only for molecule_treats_disease support.",
            "staged_evidence_rows": int(len(evidence)),
            "staged_distinct_edge_keys": int(evidence["edge_key"].nunique()) if len(evidence) else 0,
            "supported_canonical_edges": int(matches[["x_id", "y_id"]].drop_duplicates().shape[0]) if len(matches) else 0,
            "unsupported_canonical_edges": int(con.execute("SELECT count(*) FROM (SELECT DISTINCT relation||'|'||x_id||'|'||y_id AS edge_key FROM treat_edges) e ANTI JOIN (SELECT DISTINCT edge_key FROM read_parquet(?)) v USING(edge_key)", [evidence_path.as_posix()]).fetchone()[0]),
            "evidence_without_edge": int(con.execute("SELECT count(*) FROM (SELECT DISTINCT edge_key FROM read_parquet(?)) v ANTI JOIN (SELECT DISTINCT relation||'|'||x_id||'|'||y_id AS edge_key FROM treat_edges) e USING(edge_key)", [evidence_path.as_posix()]).fetchone()[0]),
            "clinical_stage_counts": matches["maxClinicalStage"].value_counts(dropna=False).to_dict() if len(matches) else {},
            "mapping_source_counts": matches["mapping_source"].value_counts(dropna=False).to_dict() if len(matches) else {},
        },
        "contraindication_evidence": {
            "source_decision": "Do not use OpenTargets clinical_indication/evidence_clinical_precedence as contraindication evidence. OpenTargets drug_warning is a safety/adverse-warning source (Black Box Warning/Withdrawn, toxicity class and warning-class EFO), not a clean molecule_contraindicates_disease assertion; stage no contraindication evidence without a contraindication-specific source such as DrugBank contraindications/SIDER label contraindication text with disease mapping.",
            "drug_warning_summary": warning_summary,
            "shape_compatible_warning_overlap_not_promoted": {
                "rows": int(contraind_warning_overlap["rows"]),
                "distinct_edge_keys": int(contraind_warning_overlap["distinct_supported_edge_keys"]),
            },
            "staged_evidence_rows": 0,
        },
        "artifacts": {
            "evidence_file": str(evidence_path),
            "report_file": str(out_root / "reports" / "opentargets_clinical_drug_evidence_report.json"),
        },
    }
    report_path = out_root / "reports" / "opentargets_clinical_drug_evidence_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=Path("artifacts/cache/raw/opentargets/26.03"))
    parser.add_argument("--kg-cache", type=Path, default=Path("/Users/jkobject/mnt/gcs/jouvencekb/main"))
    parser.add_argument("--out-root", type=Path, default=Path("artifacts/staged/opentargets-clinical-drug-evidence-20260622-t_ceee5d53"))
    args = parser.parse_args()
    report = run(args.raw_root, args.kg_cache, args.out_root)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
