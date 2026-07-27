"""Build an immutable staged PRISM 20Q2 response feature and edge candidate.

The producer maps cell lines only by exact DepMap ACH identifier and compounds
only by a unique structure-derived InChIKey. Names are retained as evidence but
are never mapping keys. Outputs are staged review artifacts; this module never
writes the canonical KG or LaminDB.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem

try:
    from .kg_schema import EDGE_PARQUET_COLUMNS
except ImportError:  # pragma: no cover
    from kg_schema import EDGE_PARQUET_COLUMNS  # type: ignore[no-redef]


RELEASE = "PRISM Repurposing 20Q2 Secondary"
DOI = "10.6084/m9.figshare.20564034.v1"
LICENSE = "CC BY 4.0"
SOURCE = "Broad PRISM Repurposing"
RELATION = "cell_line_responds_to_molecule"
FEATURE_NAME = "cell_line_molecule_viability_response"
EDGE_COLUMNS = [name for name, _ in EDGE_PARQUET_COLUMNS]

FEATURE_COLUMNS = [
    "record_id", "record_type", "cell_line_id", "molecule_id", "depmap_id", "source_row_name",
    "broad_id", "source_molecule_name", "source_smiles", "source_inchikey",
    "mapping_status", "screen_id", "is_preferred_screen", "dose_um",
    "exposure_time_hours", "compound_plate", "pool_id", "minipool_id",
    "detection_pool", "culture", "assay", "passed_str_profiling",
    "upper_limit", "lower_limit", "slope", "r2", "auc", "ec50", "ic50",
    "logfold_change", "viability", "viability_derivation", "moa", "target",
    "disease_area", "indication", "phase", "source_record_json", "source",
    "source_dataset", "release", "doi", "license",
]

EVIDENCE_COLUMNS = [
    "edge_key", "relation", "x_id", "x_type", "y_id", "y_type",
    "evidence_type", "source", "source_dataset", "source_record_id",
    "dataset_id", "study_id", "evidence_score", "effect_size", "direction",
    "predicate", "extraction_method", "license", "release", "doi", "assay",
    "source_cell_line_id", "source_molecule_id", "source_molecule_name",
    "molecule_mapping_confidence", "screen_id", "preferred_screen_policy",
    "passed_str_profiling", "auc", "ec50", "ic50", "dose_response_r2",
    "threshold_rule", "source_record_json",
]

FEATURE_FLOAT_COLUMNS = {
    "dose_um", "exposure_time_hours", "upper_limit", "lower_limit", "slope",
    "r2", "auc", "ec50", "ic50", "logfold_change", "viability",
}
FEATURE_BOOL_COLUMNS = {"is_preferred_screen", "passed_str_profiling"}
FEATURE_SCHEMA = pa.schema([
    pa.field(column, pa.bool_() if column in FEATURE_BOOL_COLUMNS else pa.float64() if column in FEATURE_FLOAT_COLUMNS else pa.string())
    for column in FEATURE_COLUMNS
])


@dataclass(frozen=True)
class PrismConfig:
    auc_threshold: float = 0.70
    min_r2: float = 0.80
    dose_batch_size: int = 10_000


def _stable_id(*parts: Any) -> str:
    payload = json.dumps(parts, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clean(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _bool(value: Any) -> bool:
    return _clean(value).lower() in {"true", "1", "yes"}


def _inchikeys(smiles_value: Any) -> list[str]:
    keys: set[str] = set()
    for smiles in _clean(smiles_value).split(","):
        smiles = smiles.strip()
        if not smiles:
            continue
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is not None:
            keys.add(Chem.MolToInchiKey(molecule))
    return sorted(keys)


def _source_json(row: pd.Series, columns: Iterable[str]) -> str:
    return json.dumps({column: _clean(row.get(column)) for column in columns}, sort_keys=True, separators=(",", ":"))


def _write_parquet(frame: pd.DataFrame, path: Path, columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    for column in columns:
        if column not in frame:
            frame[column] = pd.NA
    frame = frame[columns].reset_index(drop=True)
    frame.to_parquet(path, index=False, compression="zstd")


def _feature_table(frame: pd.DataFrame) -> pa.Table:
    """Coerce a bounded feature batch to the stable staged feature schema."""
    frame = frame.copy()
    for column in FEATURE_COLUMNS:
        if column not in frame:
            frame[column] = pd.NA
        if column in FEATURE_FLOAT_COLUMNS:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float64")
        elif column in FEATURE_BOOL_COLUMNS:
            frame[column] = frame[column].astype("boolean")
        else:
            frame[column] = frame[column].map(_clean)
    return pa.Table.from_pandas(frame[FEATURE_COLUMNS], schema=FEATURE_SCHEMA, preserve_index=False)


def build_crosswalk(treatment: pd.DataFrame, molecule_nodes: pd.DataFrame) -> pd.DataFrame:
    """Map exact Broad formulation IDs through unique molecular structure only."""
    canonical: dict[str, list[str]] = {}
    for row in molecule_nodes[["id", "inchikey"]].dropna(subset=["id", "inchikey"]).itertuples(index=False):
        key = _clean(row.inchikey)
        if key:
            canonical.setdefault(key, []).append(str(row.id))

    rows: list[dict[str, str]] = []
    for broad_id, group in treatment.groupby("broad_id", sort=True, dropna=False):
        source_keys = sorted({key for value in group["smiles"] for key in _inchikeys(value)})
        candidates = sorted({candidate for key in source_keys for candidate in canonical.get(key, [])})
        if not source_keys:
            status = "missing_structure"
        elif len(source_keys) > 1:
            status = "conflicting_source_structures"
        elif len(candidates) == 0:
            status = "unmatched_inchikey"
        elif len(candidates) > 1:
            status = "ambiguous_inchikey"
        else:
            status = "mapped_unique_inchikey"
        rows.append({
            "broad_id": _clean(broad_id),
            "source_name": next((_clean(v) for v in group.get("name", []) if _clean(v)), ""),
            "source_smiles": next((_clean(v) for v in group["smiles"] if _clean(v)), ""),
            "source_inchikey": source_keys[0] if len(source_keys) == 1 else "",
            "molecule_id": candidates[0] if status == "mapped_unique_inchikey" else "",
            "mapping_status": status,
            "candidate_molecule_ids": "|".join(candidates),
            "mapping_method": "RDKit InChIKey exact match to canonical molecule.inchikey; names prohibited",
        })
    return pd.DataFrame(rows).sort_values("broad_id", kind="stable").reset_index(drop=True)


def _pool_lookup(pool_path: Path | None) -> pd.DataFrame:
    columns = ["row_name", "screen_id", "pool_id", "minipool_id", "detection_pool", "culture"]
    if pool_path is None:
        return pd.DataFrame(columns=columns)
    pool = pd.read_csv(pool_path, low_memory=False)
    missing_keys = {"row_name", "screen_id"} - set(pool.columns)
    if missing_keys:
        raise ValueError(f"pool metadata lacks source-grain keys: {sorted(missing_keys)}")
    for column in columns:
        if column not in pool:
            pool[column] = ""
        pool[column] = pool[column].map(_clean)
    context_columns = ["pool_id", "minipool_id", "detection_pool", "culture"]
    context_counts = pool.groupby(["row_name", "screen_id"], dropna=False)[context_columns].nunique(dropna=False)
    if bool((context_counts > 1).any(axis=None)):
        raise ValueError("pool metadata has conflicting context for a source row_name/screen_id key")
    return pool[columns].sort_values(columns, kind="stable").drop_duplicates(["row_name", "screen_id"], keep="first")


def _readback_pool_context(
    feature_path: Path,
    pool: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    """Stream the completed feature Parquet and compare every pool field to source."""
    context_columns = ["pool_id", "minipool_id", "detection_pool", "culture"]
    joined_rows = 0
    missing_join_count = 0
    mismatch_counts = {column: 0 for column in context_columns}
    columns = ["source_row_name", "screen_id", *context_columns]
    for batch in pq.ParquetFile(feature_path).iter_batches(columns=columns):
        values = batch.to_pydict()
        for index, (row_name, screen_id) in enumerate(
            zip(values["source_row_name"], values["screen_id"], strict=True)
        ):
            source_context = pool.get((_clean(row_name), _clean(screen_id)))
            if source_context is None:
                missing_join_count += 1
                continue
            joined_rows += 1
            for column in context_columns:
                if _clean(values[column][index]) != _clean(source_context.get(column)):
                    mismatch_counts[column] += 1
    return {
        "pool_context_joined_feature_rows": joined_rows,
        "pool_context_missing_join_count": missing_join_count,
        "pool_context_mismatch_counts": mismatch_counts,
    }


def _base_feature(row: pd.Series, crosswalk: dict[str, dict[str, Any]], pool: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
    broad_id = _clean(row.get("broad_id"))
    mapping = crosswalk[broad_id]
    depmap_id = _clean(row.get("depmap_id"))
    source_row_name = _clean(row.get("row_name"))
    screen_id = _clean(row.get("screen_id"))
    context = pool.get((source_row_name, screen_id), {})
    return {
        "cell_line_id": depmap_id,
        "molecule_id": mapping["molecule_id"],
        "depmap_id": depmap_id,
        "source_row_name": source_row_name,
        "broad_id": broad_id,
        "source_molecule_name": _clean(row.get("name")) or mapping["source_name"],
        "source_smiles": _clean(row.get("smiles")) or mapping["source_smiles"],
        "source_inchikey": mapping["source_inchikey"],
        "mapping_status": mapping["mapping_status"],
        "screen_id": screen_id,
        "compound_plate": _clean(row.get("compound_plate")),
        "pool_id": _clean(context.get("pool_id")),
        "minipool_id": _clean(context.get("minipool_id")),
        "detection_pool": _clean(context.get("detection_pool")),
        "culture": _clean(context.get("culture")),
        "moa": _clean(row.get("moa")),
        "target": _clean(row.get("target")),
        "disease_area": _clean(row.get("disease.area")),
        "indication": _clean(row.get("indication")),
        "phase": _clean(row.get("phase")),
        "source": SOURCE,
        "source_dataset": RELEASE,
        "release": RELEASE,
        "doi": DOI,
        "license": LICENSE,
    }


def build_prism_20q2(
    *, curve_path: Path, treatment_path: Path, lfc_path: Path,
    cell_line_path: Path, molecule_path: Path, output_dir: Path,
    pool_path: Path | None = None, source_manifest_path: Path | None = None,
    config: PrismConfig = PrismConfig(),
) -> dict[str, Any]:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    cells = set(pd.read_parquet(cell_line_path, columns=["id"])["id"].astype(str))
    molecules = pd.read_parquet(molecule_path, columns=["id", "inchikey"])
    treatment = pd.read_csv(treatment_path, low_memory=False).fillna("")
    curve = pd.read_csv(curve_path, low_memory=False)
    raw_curve_rows = len(curve)
    crosswalk = build_crosswalk(treatment, molecules)
    mapped_crosswalk = crosswalk[crosswalk["mapping_status"] == "mapped_unique_inchikey"]
    crosswalk_records = crosswalk.set_index("broad_id").to_dict("index")
    mapped_broad_ids = set(mapped_crosswalk["broad_id"])
    pool_frame = _pool_lookup(pool_path)
    pool_records = pool_frame.set_index(["row_name", "screen_id"]).to_dict("index") if not pool_frame.empty else {}

    _write_parquet(crosswalk, output_dir / "mapping" / "broad_id_to_molecule.parquet", list(crosswalk.columns))
    quarantine = crosswalk[crosswalk["mapping_status"] != "mapped_unique_inchikey"].copy()
    _write_parquet(quarantine, output_dir / "mapping" / "quarantine.parquet", list(crosswalk.columns))

    curve = curve[curve["depmap_id"].astype(str).isin(cells) & curve["broad_id"].astype(str).isin(mapped_broad_ids)].copy()
    curve_rows: list[dict[str, Any]] = []
    curve_source_columns = list(curve.columns)
    for _, row in curve.iterrows():
        feature = _base_feature(row, crosswalk_records, pool_records)
        feature.update({
            "record_type": "curve_fit", "dose_um": None, "exposure_time_hours": None,
            "assay": "pooled cell-line dose-response viability; four-parameter log-logistic fit",
            "passed_str_profiling": _bool(row.get("passed_str_profiling")),
            "upper_limit": pd.to_numeric(row.get("upper_limit"), errors="coerce"),
            "lower_limit": pd.to_numeric(row.get("lower_limit"), errors="coerce"),
            "slope": pd.to_numeric(row.get("slope"), errors="coerce"),
            "r2": pd.to_numeric(row.get("r2"), errors="coerce"),
            "auc": pd.to_numeric(row.get("auc"), errors="coerce"),
            "ec50": pd.to_numeric(row.get("ec50"), errors="coerce"),
            "ic50": pd.to_numeric(row.get("ic50"), errors="coerce"),
            "logfold_change": None, "viability": None, "viability_derivation": "",
            "source_record_json": _source_json(row, curve_source_columns),
        })
        feature["record_id"] = _stable_id(RELEASE, "curve_fit", feature["source_row_name"], feature["depmap_id"], feature["broad_id"], feature["screen_id"])
        curve_rows.append(feature)
    curve_features = pd.DataFrame(curve_rows)

    preferred_pairs = set(
        curve_features.loc[curve_features["screen_id"] == "MTS010", ["cell_line_id", "molecule_id"]].itertuples(index=False, name=None)
    ) if not curve_features.empty else set()
    if not curve_features.empty:
        curve_features["is_preferred_screen"] = [
            screen == "MTS010" or (cell, molecule) not in preferred_pairs
            for cell, molecule, screen in curve_features[["cell_line_id", "molecule_id", "screen_id"]].itertuples(index=False, name=None)
        ]

    lfc = pd.read_csv(lfc_path, low_memory=False)
    row_column = lfc.columns[0]
    treatment_mapped = treatment[treatment["broad_id"].isin(mapped_broad_ids)].drop_duplicates("column_name").copy()
    treatment_mapped["_source_record_json"] = treatment_mapped.apply(lambda row: _source_json(row, treatment.columns), axis=1)
    treatment_by_column = treatment_mapped.set_index("column_name")
    dose_columns = [column for column in lfc.columns[1:] if column in treatment_by_column.index]
    lfc = lfc[lfc[row_column].astype(str).str.replace("PR500_", "", regex=False).isin(cells)]
    molecule_ids = set(molecules["id"].astype(str))
    feature_path = output_dir / "features" / f"{FEATURE_NAME}.parquet"
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    feature_writer = pq.ParquetWriter(feature_path, FEATURE_SCHEMA, compression="zstd")
    feature_rows = 0
    feature_endpoint_antijoin_count = 0
    feature_record_ids: set[str] = set()
    duplicate_feature_record_count = 0

    def write_feature_batch(frame: pd.DataFrame) -> None:
        nonlocal feature_rows, feature_endpoint_antijoin_count, duplicate_feature_record_count
        if frame.empty:
            return
        record_ids = frame["record_id"].astype(str)
        duplicate_feature_record_count += int(record_ids.duplicated().sum())
        duplicate_feature_record_count += sum(record_id in feature_record_ids for record_id in record_ids.drop_duplicates())
        feature_record_ids.update(record_ids)
        feature_endpoint_antijoin_count += int(
            (~frame["cell_line_id"].astype(str).isin(cells) | ~frame["molecule_id"].astype(str).isin(molecule_ids)).sum()
        )
        feature_writer.write_table(_feature_table(frame))
        feature_rows += len(frame)

    write_feature_batch(curve_features)
    try:
        for _, matrix_row in lfc.iterrows():
            source_row_name = _clean(matrix_row[row_column])
            depmap_id = source_row_name.replace("PR500_", "")
            values = pd.to_numeric(matrix_row[dose_columns], errors="coerce").dropna()
            for start in range(0, len(values), config.dose_batch_size):
                value_chunk = values.iloc[start:start + config.dose_batch_size]
                metadata = treatment_by_column.loc[value_chunk.index].reset_index()
                broad_ids = metadata["broad_id"].map(_clean)
                screens = metadata["screen_id"].map(_clean)
                molecule_id_values = broad_ids.map(lambda value: crosswalk_records[value]["molecule_id"])
                contexts = [pool_records.get((source_row_name, screen), {}) for screen in screens]
                names = metadata["name"].map(_clean) if "name" in metadata else pd.Series("", index=metadata.index)
                smiles = metadata["smiles"].map(_clean) if "smiles" in metadata else pd.Series("", index=metadata.index)
                feature_batch = pd.DataFrame({
                    "record_id": [_stable_id(RELEASE, "dose_observation", source_row_name, depmap_id, column) for column in value_chunk.index],
                    "record_type": "dose_observation",
                    "cell_line_id": depmap_id,
                    "molecule_id": molecule_id_values,
                    "depmap_id": depmap_id,
                    "source_row_name": source_row_name,
                    "broad_id": broad_ids,
                    "source_molecule_name": [name or crosswalk_records[broad_id]["source_name"] for name, broad_id in zip(names, broad_ids, strict=False)],
                    "source_smiles": [value or crosswalk_records[broad_id]["source_smiles"] for value, broad_id in zip(smiles, broad_ids, strict=False)],
                    "source_inchikey": broad_ids.map(lambda value: crosswalk_records[value]["source_inchikey"]),
                    "mapping_status": broad_ids.map(lambda value: crosswalk_records[value]["mapping_status"]),
                    "screen_id": screens,
                    "is_preferred_screen": [screen == "MTS010" or (depmap_id, molecule_id) not in preferred_pairs for screen, molecule_id in zip(screens, molecule_id_values, strict=False)],
                    "dose_um": pd.to_numeric(metadata["dose"], errors="coerce"),
                    "compound_plate": metadata["compound_plate"].map(_clean) if "compound_plate" in metadata else "",
                    "pool_id": [_clean(context.get("pool_id")) for context in contexts],
                    "minipool_id": [_clean(context.get("minipool_id")) for context in contexts],
                    "detection_pool": [_clean(context.get("detection_pool")) for context in contexts],
                    "culture": [_clean(context.get("culture")) for context in contexts],
                    "assay": "replicate-collapsed pooled cell-line viability",
                    "passed_str_profiling": None,
                    "logfold_change": value_chunk.to_numpy(dtype=float),
                    "viability": np.exp2(value_chunk.to_numpy(dtype=float)),
                    "viability_derivation": "2 ** replicate-collapsed log2 fold change relative to DMSO",
                    "moa": metadata["moa"].map(_clean) if "moa" in metadata else "",
                    "target": metadata["target"].map(_clean) if "target" in metadata else "",
                    "disease_area": metadata["disease.area"].map(_clean) if "disease.area" in metadata else "",
                    "indication": metadata["indication"].map(_clean) if "indication" in metadata else "",
                    "phase": metadata["phase"].map(_clean) if "phase" in metadata else "",
                    "source_record_json": metadata["_source_record_json"],
                    "source": SOURCE,
                    "source_dataset": RELEASE,
                    "release": RELEASE,
                    "doi": DOI,
                    "license": LICENSE,
                })
                write_feature_batch(feature_batch)
    finally:
        feature_writer.close()

    pool_readback = _readback_pool_context(feature_path, pool_records) if pool_path else {
        "pool_context_joined_feature_rows": 0,
        "pool_context_missing_join_count": 0,
        "pool_context_mismatch_counts": {
            "pool_id": 0, "minipool_id": 0, "detection_pool": 0, "culture": 0,
        },
    }
    pool_context_pass = (
        not pool_path
        or (
            pool_readback["pool_context_joined_feature_rows"] == feature_rows
            and pool_readback["pool_context_missing_join_count"] == 0
            and not any(pool_readback["pool_context_mismatch_counts"].values())
        )
    )

    selected = curve_features[
        curve_features["is_preferred_screen"]
        & curve_features["passed_str_profiling"]
        & (pd.to_numeric(curve_features["auc"], errors="coerce") <= config.auc_threshold)
        & (pd.to_numeric(curve_features["r2"], errors="coerce") >= config.min_r2)
    ].sort_values(
        ["cell_line_id", "molecule_id", "screen_id", "broad_id", "source_row_name", "record_id"],
        kind="stable",
    )
    edges: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    threshold_rule = f"preferred screen (MTS010 when available), passed STR, AUC <= {config.auc_threshold}, R2 >= {config.min_r2}"
    selected_pairs = selected.drop_duplicates(["cell_line_id", "molecule_id"], keep="first")
    for _, row in selected_pairs.iterrows():
        x_id, y_id = str(row["cell_line_id"]), str(row["molecule_id"])
        edges.append({
            "x_id": x_id, "x_type": "cell_line", "y_id": y_id, "y_type": "molecule",
            "relation": RELATION, "display_relation": "responds to molecule",
            "source": SOURCE, "credibility": 3,
        })
    for _, row in selected.iterrows():
        x_id, y_id = str(row["cell_line_id"]), str(row["molecule_id"])
        edge_key = f"{RELATION}|{x_id}|{y_id}"
        evidence.append({
            "edge_key": edge_key, "relation": RELATION, "x_id": x_id, "x_type": "cell_line",
            "y_id": y_id, "y_type": "molecule", "evidence_type": "experiment", "source": SOURCE,
            "source_dataset": RELEASE, "source_record_id": row["record_id"], "dataset_id": DOI,
            "study_id": "PRISM Repurposing Secondary", "evidence_score": 1.0 - float(row["auc"]),
            "effect_size": float(row["auc"]), "direction": "decreased_viability",
            "predicate": "viability_response_auc_at_or_below_qc_threshold",
            "extraction_method": "source_curve_qc_threshold", "license": LICENSE, "release": RELEASE,
            "doi": DOI, "assay": row["assay"], "source_cell_line_id": row["depmap_id"],
            "source_molecule_id": row["broad_id"], "source_molecule_name": row["source_molecule_name"],
            "molecule_mapping_confidence": "exact_unique_structure_inchikey",
            "screen_id": row["screen_id"], "preferred_screen_policy": "MTS010 preferred at canonical cell-line/molecule pair when available",
            "passed_str_profiling": str(bool(row["passed_str_profiling"])).lower(), "auc": row["auc"],
            "ec50": row["ec50"], "ic50": row["ic50"], "dose_response_r2": row["r2"],
            "threshold_rule": threshold_rule, "source_record_json": row["source_record_json"],
        })
    edge_frame = pd.DataFrame(edges)
    evidence_frame = pd.DataFrame(evidence)
    _write_parquet(edge_frame, output_dir / "edges" / f"{RELATION}.parquet", EDGE_COLUMNS)
    _write_parquet(evidence_frame, output_dir / "evidence" / f"{RELATION}.parquet", EVIDENCE_COLUMNS)

    evidence_edge_keys = set(evidence_frame.get("edge_key", pd.Series(dtype=str)).astype(str))
    candidate_edge_keys = {
        f"{RELATION}|{x_id}|{y_id}"
        for x_id, y_id in edge_frame.get("x_id", pd.Series(dtype=str)).astype(str).to_frame().join(
            edge_frame.get("y_id", pd.Series(dtype=str)).astype(str)
        ).itertuples(index=False, name=None)
    }
    endpoint_pass = set(edge_frame.get("x_id", pd.Series(dtype=str))).issubset(cells) and set(edge_frame.get("y_id", pd.Series(dtype=str))).issubset(molecule_ids)
    duplicate_edge_count = int(edge_frame.duplicated(["x_id", "y_id", "relation"]).sum())
    duplicate_evidence_record_count = int(evidence_frame.duplicated(["source_record_id"]).sum())
    eligible_pair_sizes = selected.groupby(["cell_line_id", "molecule_id"], sort=False).size()
    eligible_multi_record_pair_count = int((eligible_pair_sizes > 1).sum())
    selected_record_ids = set(selected["record_id"].astype(str))
    evidence_record_ids = set(evidence_frame.get("source_record_id", pd.Series(dtype=str)).astype(str))
    qualifying_evidence_parity = selected_record_ids == evidence_record_ids and len(evidence_frame) == len(selected)
    unsupported_edge_count = len(candidate_edge_keys - evidence_edge_keys)
    evidence_without_edge_count = len(evidence_edge_keys - candidate_edge_keys)
    curve_feature_parity = len(curve_features) == len(curve)
    validation = {
        "all_passed": bool(
            endpoint_pass and feature_endpoint_antijoin_count == 0
            and len(edge_frame) > 0 and len(evidence_frame) > 0
            and duplicate_feature_record_count == 0
            and duplicate_edge_count == 0 and duplicate_evidence_record_count == 0
            and unsupported_edge_count == 0 and evidence_without_edge_count == 0
            and curve_feature_parity and qualifying_evidence_parity and pool_context_pass
        ),
        "endpoint_antijoin_count": 0 if endpoint_pass else 1,
        "edge_rows": len(edge_frame), "evidence_rows": len(evidence_frame),
        "duplicate_feature_record_count": duplicate_feature_record_count,
        "duplicate_edge_count": duplicate_edge_count,
        "duplicate_evidence_record_count": duplicate_evidence_record_count,
        "eligible_curve_record_count": len(selected),
        "eligible_multi_record_pair_count": eligible_multi_record_pair_count,
        "qualifying_evidence_parity": qualifying_evidence_parity,
        "pool_context_gate_applied": pool_path is not None,
        "pool_context_pass": pool_context_pass,
        **pool_readback,
        "unsupported_edge_count": unsupported_edge_count,
        "evidence_without_edge_count": evidence_without_edge_count,
        "feature_endpoint_antijoin_count": feature_endpoint_antijoin_count,
        "curve_feature_parity": curve_feature_parity,
    }
    inputs = [curve_path, treatment_path, lfc_path, cell_line_path, molecule_path] + ([pool_path] if pool_path else [])
    inventory = [{"path": path.name, "bytes": path.stat().st_size, "sha256": _sha256(path)} for path in inputs]
    if source_manifest_path:
        inventory.extend(json.loads(source_manifest_path.read_text()))
    report = {
        "release": RELEASE, "doi": DOI, "license": LICENSE, "config": asdict(config),
        "source_inventory": inventory, "raw_curve_rows": raw_curve_rows,
        "mapped_curve_rows": len(curve_features), "feature_rows": feature_rows,
        "mapped_broad_ids": len(mapped_crosswalk), "quarantined_broad_ids": len(quarantine),
        "edge_rows": len(edge_frame), "evidence_rows": len(evidence_frame),
        "threshold_rule": threshold_rule,
        "pool_context_key": ["source_row_name", "screen_id"],
        "evidence_multiplicity_policy": "one evidence row per qualifying source curve; canonical edges deduplicated by cell-line/molecule pair",
        "validation": validation,
        "scope": "staged-only; no canonical KG or LaminDB write; independent of CRISPR essentiality",
        "source_limitations": ["20Q2 secondary only; 24Q2 mono-dose excluded", "source does not publish exposure time in these files", "convergence field is documented but absent from the published curve-parameter CSV", "viability is derived from published log2 fold change as documented by the release"],
    }
    (output_dir / "manifest.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (output_dir / "validation.json").write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n")
    return report


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curve", required=True)
    parser.add_argument("--treatment", required=True)
    parser.add_argument("--lfc", required=True)
    parser.add_argument("--pool")
    parser.add_argument("--cell-lines", required=True)
    parser.add_argument("--molecules", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-manifest")
    parser.add_argument("--auc-threshold", type=float, default=0.70)
    parser.add_argument("--min-r2", type=float, default=0.80)
    parser.add_argument("--dose-batch-size", type=int, default=10_000)
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = parser().parse_args(argv)
    report = build_prism_20q2(
        curve_path=Path(args.curve), treatment_path=Path(args.treatment), lfc_path=Path(args.lfc),
        pool_path=Path(args.pool) if args.pool else None, cell_line_path=Path(args.cell_lines),
        molecule_path=Path(args.molecules), output_dir=Path(args.output_dir),
        source_manifest_path=Path(args.source_manifest) if args.source_manifest else None,
        config=PrismConfig(auc_threshold=args.auc_threshold, min_r2=args.min_r2, dose_batch_size=args.dose_batch_size),
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()