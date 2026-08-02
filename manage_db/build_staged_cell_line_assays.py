"""Build staged direct cell-line assay edges/evidence.

This builder intentionally creates staged review artifacts, not canonical KG files.
It supports direct cell-line endpoint sources only:

* DepMap CRISPRGeneDependency -> cell_line_gene_essentiality
* GDSC/Sanger dose response -> cell_line_responds_to_molecule
* DepMap harmonized CCLE/Gygi MS proteomics -> cell_line_expresses_protein

No RNA/mRNA projection is used for protein expression.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import shutil
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

try:  # package execution
    from .kg_schema import EDGE_PARQUET_COLUMNS
except ImportError:  # pragma: no cover
    from kg_schema import EDGE_PARQUET_COLUMNS  # type: ignore[no-redef]

log = logging.getLogger(__name__)

EDGE_COLUMNS = [name for name, _ in EDGE_PARQUET_COLUMNS]
COMMON_EVIDENCE_COLUMNS = [
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
]
EXTRA_EVIDENCE_COLUMNS = [
    "assay",
    "source_cell_line_id",
    "source_cell_line_name",
    "cell_line_mapping_confidence",
    "source_gene_id",
    "source_gene_symbol",
    "gene_mapping_confidence",
    "source_protein_id",
    "protein_mapping_confidence",
    "source_molecule_id",
    "source_molecule_name",
    "molecule_mapping_confidence",
    "dependency_probability",
    "auc",
    "ic50",
    "published_auc",
    "published_ic50",
    "dose_response_r2",
    "protein_abundance",
    "threshold_rule",
    "source_record_json",
]
EVIDENCE_COLUMNS = COMMON_EVIDENCE_COLUMNS + EXTRA_EVIDENCE_COLUMNS

REL_ESSENTIALITY = "cell_line_gene_essentiality"
REL_RESPONSE = "cell_line_responds_to_molecule"
REL_PROTEIN = "cell_line_expresses_protein"

SOURCE_DEPMAP = "DepMap/CRISPRGeneDependency"
SOURCE_GDSC = "DepMap/Sanger GDSC dose response"
SOURCE_PROTEOMICS = "DepMap/Harmonized MS CCLE Gygi"


@dataclass(frozen=True)
class BuildConfig:
    depmap_release: str = "DepMap Public 26Q1"
    gdsc_release: str = "Sanger GDSC1 and GDSC2"
    proteomics_release: str = "Harmonized Public Proteomics 26Q1"
    essentiality_threshold: float = 0.90
    gdsc_auc_threshold: float = 0.70
    gdsc_min_r2: float = 0.80
    proteomics_top_n_per_cell_line: int = 10


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stable_id(*parts: Any) -> str:
    payload = json.dumps(parts, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


def _clean_str(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _norm_float(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def _norm_entrez(value: Any) -> str:
    text = _clean_str(value)
    return re.sub(r"\.0$", "", text)


def _parse_crispr_gene_column(col: str) -> tuple[str, str] | None:
    match = re.match(r"^(?P<symbol>.+) \((?P<entrez>\d+)\)$", col)
    if not match:
        return None
    return match.group("symbol"), match.group("entrez")


def _edge_key(relation: str, x_id: str, y_id: str) -> str:
    return f"{relation}|{x_id}|{y_id}"


def _edge_row(relation: str, x_id: str, x_type: str, y_id: str, y_type: str, display_relation: str, source: str) -> dict[str, Any]:
    return {
        "x_id": x_id,
        "x_type": x_type,
        "y_id": y_id,
        "y_type": y_type,
        "relation": relation,
        "display_relation": display_relation,
        "source": source,
        "credibility": 3,
    }


def _empty_edges() -> pd.DataFrame:
    return pd.DataFrame(columns=EDGE_COLUMNS)


def _empty_evidence() -> pd.DataFrame:
    return pd.DataFrame(columns=EVIDENCE_COLUMNS)


def _coerce_and_write(df: pd.DataFrame, path: Path, columns: list[str]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    for col in columns:
        if col not in df.columns:
            df[col] = pd.NA
    df = df[columns].copy()
    if "credibility" in df.columns:
        df["credibility"] = pd.to_numeric(df["credibility"], errors="coerce").fillna(3).astype("int64")
    for col in df.columns:
        if col == "credibility":
            continue
        if col in {"evidence_score", "effect_size", "p_value", "dependency_probability", "auc", "ic50", "published_auc", "published_ic50", "dose_response_r2", "protein_abundance"}:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
        else:
            df[col] = df[col].fillna("").astype("string[pyarrow]")
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path)
    return len(df)


def load_cell_line_maps(cell_line_path: Path, model_path: Path) -> tuple[set[str], dict[str, str], dict[str, str]]:
    cell = pd.read_parquet(cell_line_path, columns=["id", "name", "ccle_name"])
    canonical_ids = set(cell["id"].astype(str))
    depmap_to_cell = {cid: cid for cid in canonical_ids}
    name_by_id = dict(zip(cell["id"].astype(str), cell["name"].fillna(cell["ccle_name"]).astype(str), strict=False))

    model = pd.read_csv(model_path, low_memory=False)
    if "ModelID" in model.columns:
        for model_id in model["ModelID"].dropna().astype(str):
            if model_id in canonical_ids:
                depmap_to_cell[model_id] = model_id
    cosmic_to_cell: dict[str, str] = {}
    if {"ModelID", "COSMICID"}.issubset(model.columns):
        for _, row in model[["ModelID", "COSMICID"]].dropna().iterrows():
            model_id = str(row["ModelID"])
            if model_id in canonical_ids:
                cosmic_to_cell[_norm_entrez(row["COSMICID"])] = model_id
    return canonical_ids, depmap_to_cell, cosmic_to_cell | name_by_id


def load_gene_map(gene_path: Path) -> dict[str, str]:
    gene = pd.read_parquet(gene_path, columns=["id", "ncbi_gene_id"])
    gene = gene.dropna(subset=["id", "ncbi_gene_id"]).copy()
    gene["entrez"] = gene["ncbi_gene_id"].map(_norm_entrez)
    gene = gene[gene["entrez"] != ""]
    gene = gene.sort_values("id").drop_duplicates("entrez", keep="first")
    return dict(zip(gene["entrez"], gene["id"].astype(str), strict=False))


def load_protein_map(protein_path: Path, uniprot_mapping_path: Path | None = None) -> dict[str, str]:
    protein = pd.read_parquet(protein_path, columns=["id", "uniprot_id"])
    rows = []
    for _, row in protein.dropna(subset=["id"]).iterrows():
        pid = str(row["id"])
        rows.append((pid, pid))
        up = _clean_str(row.get("uniprot_id"))
        if up:
            rows.append((up, pid))
    mapping = dict(sorted(rows, key=lambda pair: pair[1]))
    # The DepMap mapping table is retained for audit; canonical protein nodes already
    # expose UniProt IDs, so no gene/RNA projection is needed here.
    if uniprot_mapping_path and uniprot_mapping_path.exists():
        _ = pd.read_csv(uniprot_mapping_path, nrows=1)
    return mapping


def load_molecule_name_map(molecule_path: Path) -> dict[str, str]:
    mol = pd.read_parquet(molecule_path)
    if "name" not in mol.columns:
        return {}
    tmp = mol[["id", "name"]].dropna().copy()
    tmp["name_norm"] = tmp["name"].astype(str).str.strip().str.lower()
    counts = tmp.groupby("name_norm")["id"].nunique()
    unique_names = set(counts[counts == 1].index)
    tmp = tmp[tmp["name_norm"].isin(unique_names)].sort_values("id")
    return dict(zip(tmp["name_norm"], tmp["id"].astype(str), strict=False))


def _base_evidence(relation: str, x_id: str, x_type: str, y_id: str, y_type: str, source: str, source_dataset: str, release: str, created_at: str) -> dict[str, Any]:
    return {
        "edge_key": _edge_key(relation, x_id, y_id),
        "relation": relation,
        "x_id": x_id,
        "x_type": x_type,
        "y_id": y_id,
        "y_type": y_type,
        "evidence_type": "experiment",
        "source": source,
        "source_dataset": source_dataset,
        "paper_id": "",
        "dataset_id": "",
        "study_id": "",
        "p_value": None,
        "direction": "",
        "confidence_interval": "",
        "text_span": "",
        "section": "",
        "extraction_method": "source_matrix_threshold",
        "license": "DepMap Terms of Use",
        "release": release,
        "created_at": created_at,
    }


def build_essentiality(crispr_path: Path, cell_line_path: Path, model_path: Path, gene_path: Path, output_dir: Path, config: BuildConfig) -> dict[str, Any]:
    created_at = _now()
    canonical_cell_lines, depmap_to_cell, _ = load_cell_line_maps(cell_line_path, model_path)
    gene_map = load_gene_map(gene_path)
    header = pd.read_csv(crispr_path, nrows=0).columns.tolist()
    id_col = header[0]
    gene_cols: list[str] = []
    col_to_gene: dict[str, tuple[str, str, str]] = {}
    for col in header[1:]:
        parsed = _parse_crispr_gene_column(col)
        if not parsed:
            continue
        symbol, entrez = parsed
        gene_id = gene_map.get(entrez)
        if gene_id:
            gene_cols.append(col)
            col_to_gene[col] = (gene_id, entrez, symbol)

    edges: dict[tuple[str, str], dict[str, Any]] = {}
    evidence: list[dict[str, Any]] = []
    rows_seen = rows_with_canonical_cell_line = accepted = 0
    usecols = [id_col] + gene_cols
    for chunk in pd.read_csv(crispr_path, usecols=usecols, chunksize=64, low_memory=False):
        rows_seen += len(chunk)
        chunk[id_col] = chunk[id_col].astype(str)
        chunk = chunk[chunk[id_col].isin(depmap_to_cell)]
        rows_with_canonical_cell_line += len(chunk)
        if chunk.empty:
            continue
        melted = chunk.melt(id_vars=[id_col], var_name="source_gene", value_name="dependency_probability")
        melted["dependency_probability"] = pd.to_numeric(melted["dependency_probability"], errors="coerce")
        melted = melted[melted["dependency_probability"] >= config.essentiality_threshold]
        for _, row in melted.iterrows():
            cell_id = depmap_to_cell[str(row[id_col])]
            gene_id, entrez, symbol = col_to_gene[str(row["source_gene"])]
            dep_prob = float(row["dependency_probability"])
            edges.setdefault((cell_id, gene_id), _edge_row(REL_ESSENTIALITY, cell_id, "cell_line", gene_id, "gene", "gene essentiality", SOURCE_DEPMAP))
            ev = _base_evidence(REL_ESSENTIALITY, cell_id, "cell_line", gene_id, "gene", SOURCE_DEPMAP, "CRISPRGeneDependency.csv", config.depmap_release, created_at)
            ev.update({
                "source_record_id": _stable_id(SOURCE_DEPMAP, cell_id, gene_id, dep_prob),
                "study_id": "Achilles/DepMap CRISPR dependency",
                "evidence_score": dep_prob,
                "effect_size": dep_prob,
                "predicate": "dependency_probability_at_or_above_threshold",
                "assay": "CRISPR gene dependency",
                "source_cell_line_id": str(row[id_col]),
                "source_cell_line_name": str(row[id_col]),
                "cell_line_mapping_confidence": "exact_depmap_model_id",
                "source_gene_id": entrez,
                "source_gene_symbol": symbol,
                "gene_mapping_confidence": "exact_ncbi_entrez_to_canonical_gene",
                "dependency_probability": dep_prob,
                "threshold_rule": f"dependency_probability >= {config.essentiality_threshold}",
                "source_record_json": json.dumps({"source_gene_column": str(row["source_gene"]), "dependency_probability": dep_prob}, sort_keys=True),
            })
            evidence.append(ev)
            accepted += 1

    edge_df = pd.DataFrame(edges.values()) if edges else _empty_edges()
    ev_df = pd.DataFrame(evidence) if evidence else _empty_evidence()
    _coerce_and_write(edge_df.drop_duplicates(), output_dir / "edges" / f"{REL_ESSENTIALITY}.parquet", EDGE_COLUMNS)
    _coerce_and_write(ev_df.drop_duplicates(subset=["source_record_id"]), output_dir / "evidence" / f"{REL_ESSENTIALITY}.parquet", EVIDENCE_COLUMNS)
    return {
        "relation": REL_ESSENTIALITY,
        "source": SOURCE_DEPMAP,
        "raw_rows_seen": rows_seen,
        "rows_with_canonical_cell_line": rows_with_canonical_cell_line,
        "source_gene_columns": len(header) - 1,
        "mapped_gene_columns": len(gene_cols),
        "edge_rows": int(len(edge_df.drop_duplicates())),
        "evidence_rows": int(len(ev_df.drop_duplicates(subset=["source_record_id"]))),
        "threshold_rule": f"dependency_probability >= {config.essentiality_threshold}",
        "mapping_confidence": {"cell_line": "exact DepMap ModelID", "gene": "NCBI Entrez column suffix to canonical gene.ncbi_gene_id"},
    }


def _first_broad_id(value: Any) -> str:
    text = _clean_str(value)
    if not text:
        return ""
    return text.split(",")[0].strip()


def build_gdsc_response(gdsc_path: Path, cell_line_path: Path, model_path: Path, molecule_path: Path, output_dir: Path, config: BuildConfig) -> dict[str, Any]:
    created_at = _now()
    _, _, cosmic_to_cell_or_name = load_cell_line_maps(cell_line_path, model_path)
    molecule_by_name = load_molecule_name_map(molecule_path)
    cols = ["DATASET", "COSMIC_ID", "DRUG_ID", "DRUG_NAME", "BROAD_ID", "IC50_PUBLISHED", "AUC_PUBLISHED", "auc", "log2.ic50", "R2"]
    df = pd.read_csv(gdsc_path, usecols=cols, low_memory=False)
    raw_rows = len(df)
    df["cell_line_id"] = df["COSMIC_ID"].map(lambda v: cosmic_to_cell_or_name.get(_norm_entrez(v), ""))
    df["molecule_id"] = df["DRUG_NAME"].map(lambda v: molecule_by_name.get(_clean_str(v).lower(), ""))
    df["auc_numeric"] = pd.to_numeric(df["auc"], errors="coerce").combine_first(pd.to_numeric(df["AUC_PUBLISHED"], errors="coerce"))
    df["r2_numeric"] = pd.to_numeric(df["R2"], errors="coerce")
    mapped = df[(df["cell_line_id"] != "") & (df["molecule_id"] != "")].copy()
    selected = mapped[(mapped["auc_numeric"] <= config.gdsc_auc_threshold) & (mapped["r2_numeric"] >= config.gdsc_min_r2)].copy()

    edges: dict[tuple[str, str], dict[str, Any]] = {}
    evidence: list[dict[str, Any]] = []
    for _, row in selected.iterrows():
        cell_id = str(row["cell_line_id"])
        molecule_id = str(row["molecule_id"])
        auc = _norm_float(row["auc_numeric"])
        log2ic50 = _norm_float(row.get("log2.ic50"))
        published_ic50 = _norm_float(row.get("IC50_PUBLISHED"))
        edges.setdefault((cell_id, molecule_id), _edge_row(REL_RESPONSE, cell_id, "cell_line", molecule_id, "molecule", "responds to molecule", SOURCE_GDSC))
        ev = _base_evidence(REL_RESPONSE, cell_id, "cell_line", molecule_id, "molecule", SOURCE_GDSC, "sanger-dose-response.csv", config.gdsc_release, created_at)
        ev.update({
            "source_record_id": _stable_id(SOURCE_GDSC, row.get("DATASET"), row.get("COSMIC_ID"), row.get("DRUG_ID"), auc, log2ic50),
            "study_id": _clean_str(row.get("DATASET")),
            "evidence_score": None if auc is None else 1.0 - auc,
            "effect_size": auc,
            "predicate": "viability_response_auc_at_or_below_threshold",
            "assay": "dose response viability",
            "source_cell_line_id": _norm_entrez(row.get("COSMIC_ID")),
            "cell_line_mapping_confidence": "COSMICID_to_DepMap_ModelID_to_canonical_cell_line",
            "source_molecule_id": _first_broad_id(row.get("BROAD_ID")) or _clean_str(row.get("DRUG_ID")),
            "source_molecule_name": _clean_str(row.get("DRUG_NAME")),
            "molecule_mapping_confidence": "unique_case_insensitive_molecule_name",
            "auc": auc,
            "ic50": log2ic50,
            "published_auc": _norm_float(row.get("AUC_PUBLISHED")),
            "published_ic50": published_ic50,
            "dose_response_r2": _norm_float(row.get("R2")),
            "threshold_rule": f"auc <= {config.gdsc_auc_threshold} and R2 >= {config.gdsc_min_r2}",
            "source_record_json": json.dumps({c: _clean_str(row.get(c)) for c in cols}, sort_keys=True),
        })
        evidence.append(ev)

    edge_df = pd.DataFrame(edges.values()) if edges else _empty_edges()
    ev_df = pd.DataFrame(evidence) if evidence else _empty_evidence()
    _coerce_and_write(edge_df.drop_duplicates(), output_dir / "edges" / f"{REL_RESPONSE}.parquet", EDGE_COLUMNS)
    _coerce_and_write(ev_df.drop_duplicates(subset=["source_record_id"]), output_dir / "evidence" / f"{REL_RESPONSE}.parquet", EVIDENCE_COLUMNS)
    return {
        "relation": REL_RESPONSE,
        "source": SOURCE_GDSC,
        "raw_rows_seen": raw_rows,
        "mapped_rows_before_threshold": int(len(mapped)),
        "unique_molecule_names_mapped": int(mapped["molecule_id"].nunique()),
        "edge_rows": int(len(edge_df.drop_duplicates())),
        "evidence_rows": int(len(ev_df.drop_duplicates(subset=["source_record_id"]))),
        "threshold_rule": f"auc <= {config.gdsc_auc_threshold} and R2 >= {config.gdsc_min_r2}",
        "mapping_confidence": {"cell_line": "COSMICID via DepMap Model.csv", "molecule": "unique case-insensitive canonical molecule.name"},
    }


def build_proteomics(proteomics_path: Path, cell_line_path: Path, model_path: Path, protein_path: Path, uniprot_mapping_path: Path, output_dir: Path, config: BuildConfig) -> dict[str, Any]:
    created_at = _now()
    _, depmap_to_cell, _ = load_cell_line_maps(cell_line_path, model_path)
    protein_map = load_protein_map(protein_path, uniprot_mapping_path)
    header = pd.read_csv(proteomics_path, nrows=0).columns.tolist()
    id_col = header[0]
    protein_cols = []
    col_to_protein = {}
    for col in header[1:]:
        base = col.split("-")[0]
        pid = protein_map.get(col) or protein_map.get(base)
        if pid:
            protein_cols.append(col)
            col_to_protein[col] = (pid, base)
    usecols = [id_col] + protein_cols
    df = pd.read_csv(proteomics_path, usecols=usecols, low_memory=False)
    raw_rows = len(df)
    df = df[df[id_col].astype(str).isin(depmap_to_cell)].copy()
    rows_with_canonical_cell_line = len(df)

    edges: dict[tuple[str, str], dict[str, Any]] = {}
    evidence: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        cell_id = depmap_to_cell[str(row[id_col])]
        values = pd.to_numeric(row[protein_cols], errors="coerce").dropna()
        if values.empty:
            continue
        selected = values.sort_values(ascending=False).head(config.proteomics_top_n_per_cell_line)
        for source_protein, abundance in selected.items():
            protein_id, uniprot_base = col_to_protein[str(source_protein)]
            abundance_f = float(abundance)
            edges.setdefault((cell_id, protein_id), _edge_row(REL_PROTEIN, cell_id, "cell_line", protein_id, "protein", "expresses protein", SOURCE_PROTEOMICS))
            ev = _base_evidence(REL_PROTEIN, cell_id, "cell_line", protein_id, "protein", SOURCE_PROTEOMICS, "harmonized_MS_CCLE_Gygi.csv", config.proteomics_release, created_at)
            ev.update({
                "source_record_id": _stable_id(SOURCE_PROTEOMICS, cell_id, source_protein, abundance_f),
                "study_id": "CCLE/Gygi mass-spectrometry proteomics",
                "evidence_score": abundance_f,
                "effect_size": abundance_f,
                "predicate": "direct_ms_protein_abundance_top_n_per_cell_line",
                "assay": "mass spectrometry protein abundance",
                "source_cell_line_id": str(row[id_col]),
                "source_cell_line_name": str(row[id_col]),
                "cell_line_mapping_confidence": "exact_depmap_model_id",
                "source_protein_id": str(source_protein),
                "protein_mapping_confidence": "exact_uniprot_id_to_canonical_protein",
                "protein_abundance": abundance_f,
                "threshold_rule": f"top {config.proteomics_top_n_per_cell_line} non-null protein abundance values per cell line; direct MS only, no RNA projection",
                "source_record_json": json.dumps({"source_protein_column": str(source_protein), "uniprot_base": uniprot_base, "protein_abundance": abundance_f}, sort_keys=True),
            })
            evidence.append(ev)

    edge_df = pd.DataFrame(edges.values()) if edges else _empty_edges()
    ev_df = pd.DataFrame(evidence) if evidence else _empty_evidence()
    _coerce_and_write(edge_df.drop_duplicates(), output_dir / "edges" / f"{REL_PROTEIN}.parquet", EDGE_COLUMNS)
    _coerce_and_write(ev_df.drop_duplicates(subset=["source_record_id"]), output_dir / "evidence" / f"{REL_PROTEIN}.parquet", EVIDENCE_COLUMNS)
    return {
        "relation": REL_PROTEIN,
        "source": SOURCE_PROTEOMICS,
        "raw_rows_seen": raw_rows,
        "rows_with_canonical_cell_line": rows_with_canonical_cell_line,
        "source_protein_columns": len(header) - 1,
        "mapped_protein_columns": len(protein_cols),
        "edge_rows": int(len(edge_df.drop_duplicates())),
        "evidence_rows": int(len(ev_df.drop_duplicates(subset=["source_record_id"]))),
        "threshold_rule": f"top {config.proteomics_top_n_per_cell_line} direct MS abundance values per cell line",
        "mapping_confidence": {"cell_line": "exact DepMap ModelID", "protein": "UniProt column to canonical protein.uniprot_id/id"},
    }


def validate_outputs(output_dir: Path, node_root: Path, reports: list[dict[str, Any]]) -> dict[str, Any]:
    node_ids = {
        "cell_line": set(pd.read_parquet(node_root / "cell_line.parquet", columns=["id"])["id"].astype(str)),
        "gene": set(pd.read_parquet(node_root / "gene.parquet", columns=["id"])["id"].astype(str)),
        "protein": set(pd.read_parquet(node_root / "protein.parquet", columns=["id"])["id"].astype(str)),
        "molecule": set(pd.read_parquet(node_root / "molecule.parquet", columns=["id"])["id"].astype(str)),
    }
    validation: dict[str, Any] = {"relations": {}, "all_passed": True}
    relation_targets = {REL_ESSENTIALITY: "gene", REL_RESPONSE: "molecule", REL_PROTEIN: "protein"}
    for report in reports:
        rel = report["relation"]
        edge_path = output_dir / "edges" / f"{rel}.parquet"
        ev_path = output_dir / "evidence" / f"{rel}.parquet"
        edges = pd.read_parquet(edge_path)
        ev = pd.read_parquet(ev_path)
        y_type = relation_targets[rel]
        missing_x = sorted(set(edges["x_id"].astype(str)).difference(node_ids["cell_line"]))
        missing_y = sorted(set(edges["y_id"].astype(str)).difference(node_ids[y_type]))
        edge_keys = set((edges["relation"].astype(str) + "|" + edges["x_id"].astype(str) + "|" + edges["y_id"].astype(str)).tolist())
        evidence_keys = set(ev["edge_key"].astype(str).tolist())
        unsupported_edges = sorted(edge_keys.difference(evidence_keys))
        orphan_evidence = sorted(evidence_keys.difference(edge_keys))
        passed = not missing_x and not missing_y and not unsupported_edges and not orphan_evidence and len(edges) > 0 and len(ev) > 0
        validation["relations"][rel] = {
            "edge_rows": int(len(edges)),
            "evidence_rows": int(len(ev)),
            "missing_cell_line_endpoint_count": len(missing_x),
            "missing_y_endpoint_count": len(missing_y),
            "unsupported_edge_count": len(unsupported_edges),
            "orphan_evidence_count": len(orphan_evidence),
            "passed": passed,
            "sample_missing_cell_line_ids": missing_x[:10],
            "sample_missing_y_ids": missing_y[:10],
        }
        validation["all_passed"] = validation["all_passed"] and passed
    return validation


def write_audit(output_dir: Path, source_manifest: Path | None, reports: list[dict[str, Any]], validation: dict[str, Any], config: BuildConfig) -> None:
    source_audit = {
        "audited_sources": [
            {"source": "DepMap/Project Achilles CRISPRGeneDependency", "decision": "staged", "reason": "direct DepMap ModelID cell-line endpoint and Entrez gene columns; dependency probability preserved"},
            {"source": "Sanger/Project Score", "decision": "not_staged", "reason": "not downloaded in this pilot; compatible future source for the same relation if model/gene mappings are supplied"},
            {"source": "GDSC/Sanger dose response", "decision": "staged", "reason": "direct COSMIC cell-line endpoint mapped through DepMap Model.csv; AUC/IC50/R2 preserved"},
            {"source": "PRISM Repurposing", "decision": "audited_not_staged", "reason": "public files available via DepMap metadata API, but this pilot used GDSC because canonical molecule mapping by name was cleaner"},
            {"source": "CTRP", "decision": "audited_not_staged", "reason": "public harmonized files available via DepMap metadata API; defer to a future tranche for broader molecule mapping"},
            {"source": "CCLE/Gygi MS proteomics", "decision": "staged", "reason": "direct mass-spectrometry protein abundance with DepMap ModelID cell-line endpoint and UniProt protein columns"},
            {"source": "mRNA/RNA-seq expression", "decision": "rejected_for_protein_expression", "reason": "protein expression relation must not be populated from RNA projection"},
        ],
        "config": config.__dict__,
        "source_manifest": json.loads(source_manifest.read_text()) if source_manifest and source_manifest.exists() else [],
        "build_reports": reports,
        "validation": validation,
        "promotion_recommendation": "Promote as staged-only review artifacts first. Essentiality and proteomics endpoint support is strong; GDSC response is useful but molecule mapping should be expanded beyond unique canonical names before canonical promotion.",
    }
    (output_dir / "source_audit.json").write_text(json.dumps(source_audit, indent=2))
    (output_dir / "validation.json").write_text(json.dumps(validation, indent=2))
    (output_dir / "manifest.json").write_text(json.dumps({"relations": reports, "validation": validation, "created_at": _now()}, indent=2))


def build_all(args: argparse.Namespace) -> dict[str, Any]:
    config = BuildConfig(
        essentiality_threshold=args.essentiality_threshold,
        gdsc_auc_threshold=args.gdsc_auc_threshold,
        gdsc_min_r2=args.gdsc_min_r2,
        proteomics_top_n_per_cell_line=args.proteomics_top_n_per_cell_line,
    )
    output_dir = Path(args.output_dir)
    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reports = [
        build_essentiality(Path(args.crispr_dependency), Path(args.cell_line_nodes), Path(args.model), Path(args.gene_nodes), output_dir, config),
        build_gdsc_response(Path(args.gdsc_dose_response), Path(args.cell_line_nodes), Path(args.model), Path(args.molecule_nodes), output_dir, config),
        build_proteomics(Path(args.proteomics), Path(args.cell_line_nodes), Path(args.model), Path(args.protein_nodes), Path(args.uniprot_mapping), output_dir, config),
    ]
    validation = validate_outputs(output_dir, Path(args.node_root), reports)
    write_audit(output_dir, Path(args.source_manifest) if args.source_manifest else None, reports, validation, config)
    return {"reports": reports, "validation": validation, "output_dir": str(output_dir)}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--node-root", default="/Users/jkobject/mnt/gcs/jouvencekb/main/nodes")
    p.add_argument("--cell-line-nodes", default="/Users/jkobject/mnt/gcs/jouvencekb/main/nodes/cell_line.parquet")
    p.add_argument("--gene-nodes", default="/Users/jkobject/mnt/gcs/jouvencekb/main/nodes/gene.parquet")
    p.add_argument("--protein-nodes", default="/Users/jkobject/mnt/gcs/jouvencekb/main/nodes/protein.parquet")
    p.add_argument("--molecule-nodes", default="/Users/jkobject/mnt/gcs/jouvencekb/main/nodes/molecule.parquet")
    p.add_argument("--model", default="artifacts/cache/raw/cell_line_assays/downloads/Model.csv")
    p.add_argument("--crispr-dependency", default="artifacts/cache/raw/cell_line_assays/downloads/CRISPRGeneDependency.csv")
    p.add_argument("--gdsc-dose-response", default="artifacts/cache/raw/cell_line_assays/downloads/sanger-dose-response.csv")
    p.add_argument("--proteomics", default="artifacts/cache/raw/cell_line_assays/downloads/harmonized_MS_CCLE_Gygi.csv")
    p.add_argument("--uniprot-mapping", default="artifacts/cache/raw/cell_line_assays/downloads/uniprot_hugo_entrez_id_mapping.csv")
    p.add_argument("--source-manifest", default="artifacts/cache/raw/cell_line_assays/downloads/manifest.json")
    p.add_argument("--essentiality-threshold", type=float, default=0.90)
    p.add_argument("--gdsc-auc-threshold", type=float, default=0.70)
    p.add_argument("--gdsc-min-r2", type=float, default=0.80)
    p.add_argument("--proteomics-top-n-per-cell-line", type=int, default=10)
    p.add_argument("--overwrite", action="store_true")
    return p


def main(argv: Iterable[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parser().parse_args(argv)
    result = build_all(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
