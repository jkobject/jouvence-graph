"""Build staged disease/tissue/phenotype context relation audit artifacts.

This pilot is deliberately conservative. It stages direct HPA Pathology Atlas / TCGA
cancer-type to tissue manifestation context only when both disease and tissue
endpoints are explicitly mapped. It does **not** infer phenotype tissue context from
HPO disease->phenotype annotations plus phenotype/anatomy labels.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

from manage_db import kg_evidence
from manage_db.kg_schema import EDGE_PARQUET_COLUMNS
from manage_db.kg_storage import open_kg_root, write_edges

HPA_PROTEINATLAS_URL = "https://www.proteinatlas.org/download/proteinatlas.tsv.zip"
HPA_DOWNLOAD_PAGE = "https://www.proteinatlas.org/about/download"
HPO_OBO_URL = "https://purl.obolibrary.org/obo/hp.obo"
HPOA_URL = "https://purl.obolibrary.org/obo/hp/hpoa/phenotype.hpoa"
UBERON_BASIC_URL = "https://purl.obolibrary.org/obo/uberon/basic.obo"

DISEASE_TISSUE_RELATION = "disease_manifests_in_tissue"
PHENOTYPE_TISSUE_RELATION = "phenotype_observed_in_tissue"
COMORBIDITY_RELATION = "disease_comorbid_disease"

EDGE_COLUMNS = [name for name, _ in EDGE_PARQUET_COLUMNS]
EVIDENCE_COLUMNS = [name for name, _ in kg_evidence.EVIDENCE_PARQUET_COLUMNS]

SOURCE_AUDIT_COLUMNS = [
    "source",
    "source_dataset",
    "source_url",
    "candidate_relation",
    "decision",
    "reason",
    "rows_or_terms_checked",
    "release",
    "notes",
]

REJECTED_COLUMNS = [
    "candidate_relation",
    "source",
    "source_dataset",
    "source_record_id",
    "source_predicate",
    "source_url",
    "hpa_cancer_type",
    "disease_id",
    "disease_mapping_confidence",
    "tissue_id",
    "tissue_mapping_confidence",
    "reject_reason",
    "raw_metadata_json",
]


@dataclass(frozen=True)
class CancerMapping:
    hpa_label: str
    disease_id: str
    disease_label: str
    disease_mapping_confidence: str
    tissue_id: str
    tissue_label: str
    tissue_mapping_confidence: str
    source_predicate: str = "Cancer prognostics column denotes TCGA/validation cancer type context"
    accepted: bool = True
    reject_reason: str = ""

    @property
    def column_prefix(self) -> str:
        return f"Cancer prognostics - {self.hpa_label}"


# Explicit reviewed mapping for HPA proteinatlas.tsv Cancer prognostics columns.
# Composite or non-single-tissue labels are kept as rejected candidates below.
HPA_CANCER_MAPPINGS: tuple[CancerMapping, ...] = (
    CancerMapping("Bladder Urothelial Carcinoma", "MONDO:0003890", "infiltrating bladder urothelial carcinoma", "tcga_label_to_specific_node", "UBERON:0001255", "urinary bladder", "exact_organ"),
    CancerMapping("Breast Invasive Carcinoma", "MONDO:0007254", "breast cancer", "broad_parent_for_tcga_label", "UBERON:0000310", "breast", "exact_organ"),
    CancerMapping("Cervical Squamous Cell Carcinoma and Endocervical Adenocarcinoma", "", "", "", "UBERON:0000002", "uterine cervix", "exact_organ", accepted=False, reject_reason="composite TCGA label combines squamous cervical carcinoma and endocervical adenocarcinoma; no single disease endpoint staged"),
    CancerMapping("Colon Adenocarcinoma", "EFO:1001949", "colon adenocarcinoma", "exact_label", "UBERON:0001155", "colon", "exact_organ"),
    CancerMapping("Glioblastoma Multiforme", "EFO:0000519", "glioblastoma multiforme", "exact_label", "UBERON:0000955", "brain", "disease_native_site"),
    CancerMapping("Head and Neck Squamous Cell Carcinoma", "EFO:0000181", "head and neck squamous cell carcinoma", "exact_label", "", "", "", accepted=False, reject_reason="disease endpoint exists but source label spans multiple anatomical sites; no single UBERON tissue endpoint staged"),
    CancerMapping("Kidney Chromophobe", "EFO:0000335", "chromophobe renal cell carcinoma", "tcga_synonym", "UBERON:0002113", "kidney", "exact_organ"),
    CancerMapping("Kidney Renal Clear Cell Carcinoma", "EFO:0000349", "clear cell renal carcinoma", "tcga_synonym", "UBERON:0002113", "kidney", "exact_organ"),
    CancerMapping("Kidney Renal Papillary Cell Carcinoma", "EFO:0000640", "papillary renal cell carcinoma", "tcga_synonym", "UBERON:0002113", "kidney", "exact_organ"),
    CancerMapping("Liver Hepatocellular Carcinoma", "EFO:0000182", "hepatocellular carcinoma", "tcga_synonym", "UBERON:0002107", "liver", "exact_organ"),
    CancerMapping("Lung Adenocarcinoma", "EFO:0000571", "lung adenocarcinoma", "exact_label", "UBERON:0002048", "lung", "exact_organ"),
    CancerMapping("Lung Squamous Cell Carcinoma", "EFO:0000708", "squamous cell lung carcinoma", "tcga_synonym", "UBERON:0002048", "lung", "exact_organ"),
    CancerMapping("Ovary Serous Cystadenocarcinoma", "EFO:1000043", "ovarian serous cystadenocarcinoma", "tcga_synonym", "UBERON:0000992", "ovary", "exact_organ"),
    CancerMapping("Pancreatic Adenocarcinoma", "EFO:1000044", "pancreatic adenocarcinoma", "exact_label", "UBERON:0001264", "pancreas", "exact_organ"),
    CancerMapping("Prostate Adenocarcinoma", "EFO:0000673", "prostate adenocarcinoma", "exact_label", "UBERON:0002367", "prostate gland", "exact_organ"),
    CancerMapping("Rectum Adenocarcinoma", "EFO:0005631", "rectal adenocarcinoma", "tcga_synonym", "UBERON:0001052", "rectum", "exact_organ"),
    CancerMapping("Skin Cutaneous Melanoma", "EFO:0000389", "cutaneous melanoma", "tcga_synonym", "UBERON:0002097", "skin of body", "exact_organ"),
    CancerMapping("Stomach Adenocarcinoma", "EFO:0000503", "gastric adenocarcinoma", "tcga_synonym", "UBERON:0000945", "stomach", "exact_organ"),
    CancerMapping("Testicular Germ Cell Tumor", "EFO:1000566", "Testicular Germ Cell Tumor", "exact_label", "UBERON:0000473", "testis", "exact_organ"),
    CancerMapping("Thyroid Carcinoma", "EFO:0002892", "thyroid carcinoma", "exact_label", "UBERON:0002046", "thyroid gland", "exact_organ"),
    CancerMapping("Uterine Corpus Endometrial Carcinoma", "MONDO:0000553", "uterine corpus endometrial carcinoma", "exact_label", "UBERON:0001295", "endometrium", "disease_subsite"),
)


def _default_output_dir(base: str | Path = "artifacts/staged") -> Path:
    return Path(base) / f"disease-tissue-phenotype-context-{date.today().isoformat()}"


def _fetch_bytes(url: str, *, timeout: int = 120) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "txgnn-source-audit"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # nosec - public source audit URLs
        return resp.read()


def _download_if_needed(url: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return path
    path.write_bytes(_fetch_bytes(url))
    return path


def _read_hpa_table(zip_path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        if not names:
            raise ValueError(f"empty HPA zip: {zip_path}")
        with zf.open(names[0]) as fh:
            text = io.TextIOWrapper(fh, encoding="utf-8", errors="replace")
            return pd.read_csv(text, sep="\t", dtype=str, keep_default_na=False)


def _extract_hpa_release(download_page: str) -> str:
    try:
        text = _fetch_bytes(download_page, timeout=30).decode("utf-8", errors="replace")
    except Exception:
        return "HPA proteinatlas.tsv.zip; release not machine-readable from download page"
    release_patterns = [r"Version\s+([0-9]+(?:\.[0-9]+)?)", r"version\s+([0-9]+(?:\.[0-9]+)?)", r"HPA\s+([0-9]+(?:\.[0-9]+)?)"]
    for pattern in release_patterns:
        match = re.search(pattern, text)
        if match:
            return f"HPA {match.group(1)}"
    return "HPA proteinatlas.tsv.zip; release not machine-readable from download page"


def _node_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return set(pd.read_parquet(path, columns=["id"])["id"].astype(str))


def _empty_edges() -> pd.DataFrame:
    return pd.DataFrame(columns=EDGE_COLUMNS).astype("object")


def _empty_evidence() -> pd.DataFrame:
    return pd.DataFrame(columns=EVIDENCE_COLUMNS).astype("object")


def _evidence_row(mapping: CancerMapping, source_column: str, non_empty_gene_rows: int, prognostic_gene_rows: int, release: str) -> dict[str, object]:
    created_at = datetime.now(timezone.utc).isoformat()
    raw = {
        "hpa_cancer_type": mapping.hpa_label,
        "hpa_column": source_column,
        "source_predicate": mapping.source_predicate,
        "non_empty_gene_rows_in_column": non_empty_gene_rows,
        "prognostic_gene_rows_in_column": prognostic_gene_rows,
        "disease_mapping": {
            "id": mapping.disease_id,
            "label": mapping.disease_label,
            "confidence": mapping.disease_mapping_confidence,
        },
        "tissue_mapping": {
            "id": mapping.tissue_id,
            "label": mapping.tissue_label,
            "confidence": mapping.tissue_mapping_confidence,
        },
    }
    return {
        "edge_key": f"{DISEASE_TISSUE_RELATION}|{mapping.disease_id}|{mapping.tissue_id}",
        "relation": DISEASE_TISSUE_RELATION,
        "x_id": mapping.disease_id,
        "x_type": "disease",
        "y_id": mapping.tissue_id,
        "y_type": "tissue",
        "evidence_type": "database_record",
        "source": "Human Protein Atlas",
        "source_dataset": "proteinatlas.tsv Cancer prognostics columns",
        "source_record_id": source_column,
        "paper_id": "",
        "dataset_id": "",
        "study_id": "TCGA/validation cohort as named in HPA column",
        "evidence_score": float(non_empty_gene_rows),
        "effect_size": None,
        "p_value": None,
        "direction": "",
        "confidence_interval": "",
        "predicate": mapping.source_predicate,
        "text_span": json.dumps(raw, sort_keys=True, ensure_ascii=False),
        "section": "HPA Pathology Atlas / Cancer prognostics",
        "extraction_method": "explicit manual disease+tissue mapping from HPA cancer-type column labels; no phenotype anatomy inference",
        "license": "HPA downloadable data; see https://www.proteinatlas.org/about/download",
        "release": release,
        "created_at": created_at,
    }


def build_hpa_disease_tissue(hpa_df: pd.DataFrame, *, release: str, disease_ids: set[str], tissue_ids: set[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, object]]]:
    edge_records: dict[tuple[str, str], dict[str, object]] = {}
    evidence_rows: list[dict[str, object]] = []
    rejected_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []

    for mapping in HPA_CANCER_MAPPINGS:
        matching_columns = [col for col in hpa_df.columns if col.startswith(mapping.column_prefix)]
        non_empty = 0
        prognostic = 0
        if matching_columns:
            mask = pd.Series(False, index=hpa_df.index)
            prog_mask = pd.Series(False, index=hpa_df.index)
            for col in matching_columns:
                col_values = hpa_df[col].fillna("").astype(str)
                mask |= col_values.str.len() > 0
                prog_mask |= col_values.str.contains("prognostic", case=False, na=False) & ~col_values.str.contains("unprognostic", case=False, na=False)
            non_empty = int(mask.sum())
            prognostic = int(prog_mask.sum())
        source_record = ";".join(matching_columns) if matching_columns else mapping.column_prefix
        audit_rows.append(
            {
                "source": "Human Protein Atlas",
                "source_dataset": "proteinatlas.tsv Cancer prognostics columns",
                "source_url": HPA_PROTEINATLAS_URL,
                "candidate_relation": DISEASE_TISSUE_RELATION,
                "decision": "stage_edge" if mapping.accepted and matching_columns else "reject_candidate",
                "reason": "direct cancer-type/tissue manifestation context with explicit endpoint mapping" if mapping.accepted and matching_columns else (mapping.reject_reason or "HPA cancer column missing from source table"),
                "rows_or_terms_checked": non_empty,
                "release": release,
                "notes": json.dumps({"hpa_label": mapping.hpa_label, "columns": matching_columns, "prognostic_gene_rows": prognostic}, sort_keys=True),
            }
        )
        endpoint_missing: list[str] = []
        if mapping.accepted:
            if mapping.disease_id not in disease_ids:
                endpoint_missing.append(f"disease_id_not_in_nodes:{mapping.disease_id}")
            if mapping.tissue_id not in tissue_ids:
                endpoint_missing.append(f"tissue_id_not_in_nodes:{mapping.tissue_id}")
        reject_reason = mapping.reject_reason or ";".join(endpoint_missing)
        if (not mapping.accepted) or endpoint_missing or not matching_columns:
            rejected_rows.append(
                {
                    "candidate_relation": DISEASE_TISSUE_RELATION,
                    "source": "Human Protein Atlas",
                    "source_dataset": "proteinatlas.tsv Cancer prognostics columns",
                    "source_record_id": source_record,
                    "source_predicate": mapping.source_predicate,
                    "source_url": HPA_PROTEINATLAS_URL,
                    "hpa_cancer_type": mapping.hpa_label,
                    "disease_id": mapping.disease_id,
                    "disease_mapping_confidence": mapping.disease_mapping_confidence,
                    "tissue_id": mapping.tissue_id,
                    "tissue_mapping_confidence": mapping.tissue_mapping_confidence,
                    "reject_reason": reject_reason or "source_column_missing",
                    "raw_metadata_json": json.dumps({"matching_columns": matching_columns, "non_empty_gene_rows": non_empty, "prognostic_gene_rows": prognostic}, sort_keys=True),
                }
            )
            continue
        edge_key = (mapping.disease_id, mapping.tissue_id)
        edge_records[edge_key] = {
            "x_id": mapping.disease_id,
            "x_type": "disease",
            "y_id": mapping.tissue_id,
            "y_type": "tissue",
            "relation": DISEASE_TISSUE_RELATION,
            "display_relation": "manifests in tissue",
            "source": "Human Protein Atlas",
            "credibility": 1,
        }
        for col in matching_columns:
            non_empty_col = int(hpa_df[col].fillna("").astype(str).str.len().gt(0).sum())
            prognostic_col = int(hpa_df[col].fillna("").astype(str).str.contains("prognostic", case=False, na=False).sum())
            evidence_rows.append(_evidence_row(mapping, col, non_empty_col, prognostic_col, release))

    edges = pd.DataFrame(edge_records.values(), columns=EDGE_COLUMNS) if edge_records else _empty_edges()
    evidence = pd.DataFrame(evidence_rows, columns=EVIDENCE_COLUMNS) if evidence_rows else _empty_evidence()
    rejected = pd.DataFrame(rejected_rows, columns=REJECTED_COLUMNS)
    return edges, evidence, rejected, audit_rows


def audit_non_edge_sources(hp_obo: Path | None, hpoa: Path | None, uberon_obo: Path | None, release: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    hp_terms = 0
    hp_uberon_relationships = 0
    hp_uberon_xrefs = 0
    if hp_obo and hp_obo.exists():
        current = []
        for line in hp_obo.read_text(errors="replace").splitlines():
            if line == "[Term]":
                if current:
                    block = "\n".join(current)
                    if block.startswith("id: HP:"):
                        hp_terms += 1
                        hp_uberon_relationships += sum(1 for l in current if l.startswith("relationship:") and "UBERON:" in l)
                        hp_uberon_xrefs += sum(1 for l in current if l.startswith("xref: UBERON:"))
                current = []
            else:
                current.append(line)
        if current:
            block = "\n".join(current)
            if block.startswith("id: HP:"):
                hp_terms += 1
                hp_uberon_relationships += sum(1 for l in current if l.startswith("relationship:") and "UBERON:" in l)
                hp_uberon_xrefs += sum(1 for l in current if l.startswith("xref: UBERON:"))
    rows.append(
        {
            "source": "Human Phenotype Ontology",
            "source_dataset": "hp.obo",
            "source_url": HPO_OBO_URL,
            "candidate_relation": PHENOTYPE_TISSUE_RELATION,
            "decision": "no_edge",
            "reason": "no direct HPO phenotype->UBERON tissue relationship/xref found; anatomy-like phenotype names are not treated as tissue observations",
            "rows_or_terms_checked": hp_terms,
            "release": release,
            "notes": json.dumps({"uberon_relationships": hp_uberon_relationships, "uberon_xrefs": hp_uberon_xrefs}, sort_keys=True),
        }
    )
    hpoa_rows = 0
    if hpoa and hpoa.exists():
        with hpoa.open(errors="replace") as fh:
            for row in fh:
                if not row.startswith("#") and row.strip():
                    hpoa_rows += 1
    rows.append(
        {
            "source": "Human Phenotype Ontology",
            "source_dataset": "phenotype.hpoa",
            "source_url": HPOA_URL,
            "candidate_relation": PHENOTYPE_TISSUE_RELATION,
            "decision": "no_edge",
            "reason": "HPOA directly supports disease_has_phenotype, not phenotype_observed_in_tissue; disease->phenotype plus anatomy inference is explicitly out of scope",
            "rows_or_terms_checked": hpoa_rows,
            "release": release,
            "notes": "disease annotations include PMID/reference/evidence/frequency but no tissue endpoint column",
        }
    )
    rows.append(
        {
            "source": "UBERON",
            "source_dataset": "uberon/basic.obo",
            "source_url": UBERON_BASIC_URL,
            "candidate_relation": PHENOTYPE_TISSUE_RELATION,
            "decision": "supporting_mapping_only",
            "reason": "UBERON supplies tissue endpoint IDs for HPA disease tissue mappings; it does not by itself assert phenotype observation",
            "rows_or_terms_checked": sum(1 for line in uberon_obo.read_text(errors="replace").splitlines() if line.startswith("id: UBERON:")) if uberon_obo and uberon_obo.exists() else 0,
            "release": "UBERON basic OBO fetched for staged audit",
            "notes": "used only for endpoint mapping/anti-join context",
        }
    )
    rows.append(
        {
            "source": "EHR/co-occurrence resources",
            "source_dataset": "candidate public comorbidity datasets",
            "source_url": "",
            "candidate_relation": COMORBIDITY_RELATION,
            "decision": "no_edge",
            "reason": "no clean accessible/licensable EHR co-occurrence source was identified during this staged pilot; do not synthesize comorbidity from shared annotations",
            "rows_or_terms_checked": 0,
            "release": "",
            "notes": "leave disease_comorbid_disease empty pending explicit source approval",
        }
    )
    return rows


def validate_outputs(edges: pd.DataFrame, evidence: pd.DataFrame, disease_ids: set[str], tissue_ids: set[str], source_audit: pd.DataFrame, rejected: pd.DataFrame) -> dict[str, object]:
    edge_keys = set(edges["relation"].astype(str) + "|" + edges["x_id"].astype(str) + "|" + edges["y_id"].astype(str)) if not edges.empty else set()
    evidence_keys = set(evidence["relation"].astype(str) + "|" + evidence["x_id"].astype(str) + "|" + evidence["y_id"].astype(str)) if not evidence.empty else set()
    missing_disease = sorted(set(edges["x_id"].astype(str)) - disease_ids) if not edges.empty else []
    missing_tissue = sorted(set(edges["y_id"].astype(str)) - tissue_ids) if not edges.empty else []
    duplicate_edges = int(edges.duplicated(subset=["relation", "x_id", "y_id"]).sum()) if not edges.empty else 0
    inferred_reasons = rejected["reject_reason"].fillna("").astype(str).str.contains("disease->phenotype|inference", case=False).sum() if not rejected.empty else 0
    checks = {
        "evidence_support": {
            "ok": not (edge_keys - evidence_keys),
            "edges_without_evidence": sorted(edge_keys - evidence_keys),
            "evidence_without_edge": sorted(evidence_keys - edge_keys),
        },
        "endpoint_antijoin": {
            "ok": not missing_disease and not missing_tissue,
            "missing_disease_ids": missing_disease,
            "missing_tissue_ids": missing_tissue,
        },
        "duplicate_edges": {"ok": duplicate_edges == 0, "duplicate_edges": duplicate_edges},
        "source_audit": {
            "ok": {DISEASE_TISSUE_RELATION, PHENOTYPE_TISSUE_RELATION, COMORBIDITY_RELATION}.issubset(set(source_audit["candidate_relation"].astype(str))),
            "decisions": {
                f"{relation}|{decision}": int(count)
                for (relation, decision), count in source_audit.groupby(["candidate_relation", "decision"]).size().items()
            },
        },
        "no_forbidden_phenotype_anatomy_inference": {
            "ok": PHENOTYPE_TISSUE_RELATION not in set(edges.get("relation", pd.Series(dtype=str)).astype(str)),
            "rejected_inference_rows": int(inferred_reasons),
            "note": "No phenotype_observed_in_tissue edges are staged from HPO disease annotations or anatomy-like HP names.",
        },
    }
    return {
        "ok": all(v.get("ok", False) for v in checks.values()),
        "edge_rows": int(len(edges)),
        "evidence_rows": int(len(evidence)),
        "rejected_rows": int(len(rejected)),
        "source_audit_rows": int(len(source_audit)),
        "relations_staged": sorted(edges["relation"].unique().tolist()) if not edges.empty else [],
        "checks": checks,
    }


def write_outputs(output_dir: Path, edges: pd.DataFrame, evidence: pd.DataFrame, rejected: pd.DataFrame, source_audit: pd.DataFrame, validation: dict[str, object], inputs: dict[str, str]) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "edges").mkdir(exist_ok=True)
    (output_dir / "evidence").mkdir(exist_ok=True)
    (output_dir / "reports").mkdir(exist_ok=True)
    root = open_kg_root(str(output_dir))

    if edges.empty:
        _empty_edges().to_parquet(output_dir / "edges" / f"{DISEASE_TISSUE_RELATION}.parquet", index=False)
    else:
        write_edges(root, DISEASE_TISSUE_RELATION, edges)
    # Explicit empty files for audited but non-materialized relations.
    _empty_edges().to_parquet(output_dir / "edges" / f"{PHENOTYPE_TISSUE_RELATION}.parquet", index=False)
    _empty_edges().to_parquet(output_dir / "edges" / f"{COMORBIDITY_RELATION}.parquet", index=False)

    if evidence.empty:
        _empty_evidence().to_parquet(output_dir / "evidence" / f"{DISEASE_TISSUE_RELATION}.parquet", index=False)
    else:
        kg_evidence.write_evidence(root, DISEASE_TISSUE_RELATION, evidence)
    _empty_evidence().to_parquet(output_dir / "evidence" / f"{PHENOTYPE_TISSUE_RELATION}.parquet", index=False)
    _empty_evidence().to_parquet(output_dir / "evidence" / f"{COMORBIDITY_RELATION}.parquet", index=False)

    rejected_path = output_dir / "reports" / "rejected_candidates.parquet"
    audit_path = output_dir / "reports" / "source_audit.parquet"
    validation_path = output_dir / "reports" / "validation.json"
    manifest_path = output_dir / "MANIFEST.json"
    rejected.to_parquet(rejected_path, index=False)
    source_audit.to_parquet(audit_path, index=False)
    validation_path.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n")
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "staging_only": True,
        "canonical_promotion": False,
        "relations": [DISEASE_TISSUE_RELATION, PHENOTYPE_TISSUE_RELATION, COMORBIDITY_RELATION],
        "inputs": inputs,
        "outputs": {
            "edges_dir": str(output_dir / "edges"),
            "evidence_dir": str(output_dir / "evidence"),
            "rejected_candidates": str(rejected_path),
            "source_audit": str(audit_path),
            "validation": str(validation_path),
        },
        "validation": validation,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest["outputs"] | {"manifest": str(manifest_path)}


def build(args: argparse.Namespace) -> dict[str, object]:
    output = Path(args.output_dir) if args.output_dir else _default_output_dir(args.staging_root)
    cache = Path(args.cache_dir)
    hpa_zip = _download_if_needed(HPA_PROTEINATLAS_URL, cache / "hpa-pathology" / "proteinatlas.tsv.zip")
    hp_obo = _download_if_needed(HPO_OBO_URL, cache / "hpo" / "hp.obo") if args.fetch_audit_sources else cache / "hpo" / "hp.obo"
    hpoa = _download_if_needed(HPOA_URL, cache / "hpo" / "phenotype.hpoa") if args.fetch_audit_sources else cache / "hpo" / "phenotype.hpoa"
    uberon = _download_if_needed(UBERON_BASIC_URL, cache / "uberon" / "uberon-basic.obo") if args.fetch_audit_sources else cache / "uberon" / "uberon-basic.obo"

    release = args.hpa_release or _extract_hpa_release(HPA_DOWNLOAD_PAGE)
    disease_ids = _node_ids(Path(args.node_root) / "disease.parquet")
    tissue_ids = _node_ids(Path(args.node_root) / "tissue.parquet")
    hpa_df = _read_hpa_table(hpa_zip)
    edges, evidence, rejected, audit_rows = build_hpa_disease_tissue(hpa_df, release=release, disease_ids=disease_ids, tissue_ids=tissue_ids)
    audit_rows.extend(audit_non_edge_sources(hp_obo, hpoa, uberon, release))
    source_audit = pd.DataFrame(audit_rows, columns=SOURCE_AUDIT_COLUMNS)
    validation = validate_outputs(edges, evidence, disease_ids, tissue_ids, source_audit, rejected)
    inputs = {
        "hpa_proteinatlas_zip": str(hpa_zip),
        "hpa_proteinatlas_url": HPA_PROTEINATLAS_URL,
        "hpo_obo": str(hp_obo),
        "hpoa": str(hpoa),
        "uberon_basic": str(uberon),
        "node_root": str(args.node_root),
    }
    outputs = write_outputs(output, edges, evidence, rejected, source_audit, validation, inputs)
    return {"output_dir": str(output), "outputs": outputs, "validation": validation}


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node-root", default="/Users/jkobject/mnt/gcs/jouvencekb/main/nodes", help="Directory containing disease.parquet and tissue.parquet for endpoint anti-join validation.")
    parser.add_argument("--cache-dir", default="/Users/jkobject/mnt/gcs/jouvencekb/main/raw", help="Local raw source cache directory.")
    parser.add_argument("--staging-root", default="artifacts/staged", help="Base directory for dated staged artifacts.")
    parser.add_argument("--output-dir", default="", help="Explicit output directory; overrides --staging-root/date default.")
    parser.add_argument("--hpa-release", default="", help="Override HPA release label if known.")
    parser.add_argument("--fetch-audit-sources", action="store_true", help="Download HPO/HPOA/UBERON audit sources if missing.")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    result = build(parse_args(argv))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
