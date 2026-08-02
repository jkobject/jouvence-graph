"""Build bounded PyG/HeteroData-style artifacts from canonical KG Parquet files.

The canonical KG under ``kg/v2`` stays the source of truth.  This builder creates a
reproducible derived layer under ``ml/pyg`` (or a caller supplied output path):
node maps, edge-index tensors/Parquet sidecars, relation metadata, feature row
maps, and validation reports.  It is intentionally safe for pilot exports: select
node types/relations and row caps before attempting full-KG materialization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import posixpath
import pickle
import zlib
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import fsspec
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from fsspec.core import url_to_fs

from .kg_schema import (
    GRAPH_DISCONNECTED_RELATIONS,
    NODE_TYPES,
    RELATION_BY_NAME,
    TRAINING_GRAPH_EXCLUDED_NODE_TYPES,
    RelationStatus,
)
from .kg_storage import open_kg_root

try:  # optional in the lightweight KG management environment
    import torch  # type: ignore
except Exception:  # pragma: no cover - exercised when torch is unavailable
    torch = None  # type: ignore

try:  # optional; the artifact format remains valid without importing PyG
    from torch_geometric.data import HeteroData  # type: ignore
except Exception:  # pragma: no cover - exercised when torch_geometric is unavailable
    HeteroData = None  # type: ignore


DEFAULT_KG_ROOT = "/mnt/gcs/jouvencekb/main"
DEFAULT_PILOT_NODE_TYPES = ("gene", "disease", "molecule")
DEFAULT_PILOT_RELATIONS = ("disease_associated_gene", "molecule_targets_gene")
ARTIFACT_MODES = ("auto", "sidecar", "heterodata", "both")

DEFERRED_NODE_FEATURES: dict[str, list[dict[str, str]]] = {
    "protein": [{"field": "protein_sequence_esm2", "status": "deferred", "reason": "production/full sequence embedding build pending"}],
    "transcript": [{"field": "transcript_sequence_embedding", "status": "deferred", "reason": "nucleotide transformer production build pending"}],
    "molecule": [{"field": "chemical_encoder_embedding", "status": "deferred", "reason": "learned chemical encoder production build pending"}],
    "gene": [{"field": "dna_sequence_embedding", "status": "deferred", "reason": "DNA/foundation sequence feature production build pending"}],
    "enhancer": [{"field": "dna_sequence_embedding", "status": "deferred", "reason": "DNA/foundation sequence feature production build pending"}],
    "mutation": [{"field": "dna_variant_embedding", "status": "deferred", "reason": "variant/DNA feature production build pending"}],
}
TEXTUAL_SUMMARY_NODE_TYPES_WITH_STAGED_EMBEDDINGS = {
    "cell_line", "cell_type", "disease", "gene", "molecule", "pathway", "phenotype", "protein", "tissue"
}


@dataclass(frozen=True)
class BuildConfig:
    kg_root: str = DEFAULT_KG_ROOT
    output_root: str | None = None
    node_types: tuple[str, ...] = DEFAULT_PILOT_NODE_TYPES
    relations: tuple[str, ...] = DEFAULT_PILOT_RELATIONS
    max_nodes_per_type: int | None = 50_000
    max_edges_per_relation: int | None = 100_000
    include_reverse_edges: bool = True
    feature_tables: tuple[str, ...] = ()
    strict: bool = True
    build_name: str = "pilot"
    sort_node_ids: bool = False
    embedding_features_root: str | None = None
    learned_fallback_config_path: str | None = None
    fallback_seed: int = 20260623
    include_provenance_node_types: bool = False
    plan_only: bool = False
    remote_output_root: str | None = None
    remote_runner: str = "gcp"
    artifact_mode: str = "auto"


@dataclass
class ValidationIssue:
    severity: str
    check: str
    message: str
    relation: str | None = None
    node_type: str | None = None
    count: int | None = None


@dataclass
class BuildResult:
    output_root: str
    manifest_path: str
    validation_report_path: str
    node_counts: dict[str, int]
    edge_counts: dict[str, int]
    issues: list[ValidationIssue] = field(default_factory=list)


def build_pyg_export(config: BuildConfig) -> BuildResult:
    """Build a bounded canonical KG -> PyG artifact export.

    The function reads only canonical ``nodes/``, ``edges/`` and optional
    ``features/`` Parquets from ``config.kg_root``.  It never reads ``.omoc`` and
    never mutates canonical KG files.
    """

    _validate_artifact_mode(config.artifact_mode)
    kg = open_kg_root(config.kg_root)
    output_root = config.output_root or posixpath.join(config.kg_root.rstrip("/"), "ml", "pyg")
    out = _OutputRoot(output_root)
    out.makedirs("")

    node_types = _resolve_node_types(config, kg.list_nodes())
    relations = _resolve_relations(config, kg.list_edges())
    excluded_node_values = {node_type.value for node_type in TRAINING_GRAPH_EXCLUDED_NODE_TYPES}
    excluded_requested_node_types = sorted(set(config.node_types).intersection(excluded_node_values))
    excluded_requested_relations = sorted(set(config.relations).intersection(GRAPH_DISCONNECTED_RELATIONS))

    if config.plan_only:
        manifest = _write_production_plan_manifest(
            config=config,
            kg=kg,
            out=out,
            output_root=output_root,
            node_types=node_types,
            relations=relations,
            excluded_node_values=excluded_node_values,
            excluded_requested_node_types=excluded_requested_node_types,
            excluded_requested_relations=excluded_requested_relations,
        )
        validation_report = {
            "build_name": config.build_name,
            "generated_at": manifest["generated_at"],
            "kg_root": config.kg_root,
            "output_root": output_root,
            "status": "pass",
            "error_count": 0,
            "warning_count": 0,
            "node_counts": manifest["node_rows"],
            "edge_counts": manifest["edge_rows"],
            "checks": {
                "metadata_only_no_table_scan": True,
                "remote_command_present": True,
                "training_graph_exclusion_policy_present": True,
            },
            "issues": [],
        }
        out.write_json("validation_report.json", validation_report)
        out.write_text("validation_report.md", _validation_markdown(validation_report))
        return BuildResult(
            output_root=output_root,
            manifest_path=f"{output_root.rstrip('/')}/production_plan_manifest.json",
            validation_report_path=f"{output_root.rstrip('/')}/validation_report.json",
            node_counts={k: int(v) for k, v in manifest["node_rows"].items()},
            edge_counts={k: int(v) for k, v in manifest["edge_rows"].items()},
            issues=[],
        )

    issues: list[ValidationIssue] = []
    node_maps: dict[str, pd.DataFrame] = {}
    node_counts: dict[str, int] = {}
    node_stats: dict[str, dict[str, Any]] = {}

    for node_type in node_types:
        df = _read_parquet_limited(kg._node_internal(node_type), kg.fs, ["id"], config.max_nodes_per_type)
        if "id" not in df.columns:
            raise ValueError(f"nodes/{node_type}.parquet missing id column")
        ids = df["id"].astype("string").rename("id")
        null_count = int(ids.isna().sum())
        duplicate_count = int(ids.duplicated(keep=False).sum())
        if config.strict and (null_count or duplicate_count):
            raise ValueError(
                f"nodes/{node_type}.parquet has {null_count} null IDs and {duplicate_count} duplicate IDs in selected rows"
            )
        if null_count:
            issues.append(ValidationIssue("error", "node_id_not_null", "Null node ids in selected rows", node_type=node_type, count=null_count))
        if duplicate_count:
            issues.append(ValidationIssue("error", "node_id_unique", "Duplicate node ids in selected rows", node_type=node_type, count=duplicate_count))
        clean = ids.dropna().drop_duplicates().reset_index(drop=True)
        if config.sort_node_ids:
            clean = clean.sort_values(kind="mergesort").reset_index(drop=True)
        node_map = pd.DataFrame(
            {
                "id": clean.astype(str),
                "node_type": node_type,
                "node_index": np.arange(len(clean), dtype=np.int64),
            }
        )
        inverse = node_map[["node_index", "node_type", "id"]]
        _write_parquet(out, f"node_maps/{node_type}.id_to_index.parquet", node_map)
        _write_parquet(out, f"node_maps/{node_type}.index_to_id.parquet", inverse)
        stats = {
            "node_type": node_type,
            "selected_rows": int(len(df)),
            "node_count": int(len(node_map)),
            "null_id_count": null_count,
            "duplicate_id_count": duplicate_count,
            "id_sequence_sha256": _hash_sequence(node_map["id"]),
            "source_node_path": kg.node_path(node_type),
            "bounded": config.max_nodes_per_type is not None,
            "max_nodes_per_type": config.max_nodes_per_type,
            "sort_node_ids": config.sort_node_ids,
        }
        out.write_json(f"node_maps/{node_type}.stats.json", stats)
        node_maps[node_type] = node_map
        node_counts[node_type] = int(len(node_map))
        node_stats[node_type] = stats

    edge_counts: dict[str, int] = {}
    edge_type_rows: list[dict[str, Any]] = []
    heterodata_payload: dict[str, Any] = {"node_types": {}, "edge_types": {}}
    artifact_mode = _resolve_artifact_mode(config)

    for relation in relations:
        rel = RELATION_BY_NAME[relation]
        src_type = rel.source.value
        dst_type = rel.target.value
        if src_type not in node_maps or dst_type not in node_maps:
            msg = f"Skipping {relation}: endpoint maps missing for {src_type}->{dst_type}"
            if config.strict:
                raise ValueError(msg)
            issues.append(ValidationIssue("warning", "endpoint_maps_available", msg, relation=relation))
            continue

        edge_df = _read_parquet_limited(
            kg._edge_internal(relation),
            kg.fs,
            ["x_id", "x_type", "y_id", "y_type", "relation", "source", "credibility"],
            config.max_edges_per_relation,
        )
        _validate_edge_columns(edge_df, relation)
        rel_mismatch = int((edge_df["relation"].astype(str) != relation).sum())
        endpoint_mismatch = int(
            ((edge_df["x_type"].astype(str) != src_type) | (edge_df["y_type"].astype(str) != dst_type)).sum()
        )
        if rel_mismatch:
            issues.append(ValidationIssue("error", "relation_column_matches_file", "Relation column mismatch", relation=relation, count=rel_mismatch))
        if endpoint_mismatch:
            issues.append(ValidationIssue("error", "endpoint_type_matches_schema", "Endpoint type mismatch", relation=relation, count=endpoint_mismatch))
        if config.strict and (rel_mismatch or endpoint_mismatch):
            raise ValueError(f"edges/{relation}.parquet failed relation/endpoint validation")

        src_map = node_maps[src_type][["id", "node_index"]].rename(columns={"id": "x_id", "node_index": "src_index"})
        dst_map = node_maps[dst_type][["id", "node_index"]].rename(columns={"id": "y_id", "node_index": "dst_index"})
        mapped = edge_df.astype({"x_id": "string", "y_id": "string"}).merge(src_map, on="x_id", how="left").merge(dst_map, on="y_id", how="left")
        missing_src = int(mapped["src_index"].isna().sum())
        missing_dst = int(mapped["dst_index"].isna().sum())
        if missing_src:
            issues.append(ValidationIssue("error", "x_endpoint_in_node_map", "Edges with source not present in selected node map", relation=relation, count=missing_src))
        if missing_dst:
            issues.append(ValidationIssue("error", "y_endpoint_in_node_map", "Edges with target not present in selected node map", relation=relation, count=missing_dst))
        if config.strict and (missing_src or missing_dst):
            raise ValueError(
                f"edges/{relation}.parquet has selected endpoints outside node maps: missing_src={missing_src}, missing_dst={missing_dst}"
            )
        mapped = mapped.dropna(subset=["src_index", "dst_index"]).reset_index(drop=True)
        mapped["src_index"] = mapped["src_index"].astype("int64")
        mapped["dst_index"] = mapped["dst_index"].astype("int64")
        mapped["edge_pos"] = np.arange(len(mapped), dtype=np.int64)
        mapped["edge_key"] = mapped["relation"].astype(str) + "|" + mapped["x_id"].astype(str) + "|" + mapped["y_id"].astype(str)

        edge_type_name = _edge_type_dir(src_type, relation, dst_type)
        edge_index = np.vstack(
            [mapped["src_index"].to_numpy(dtype=np.int64), mapped["dst_index"].to_numpy(dtype=np.int64)]
        )
        _write_tensor(out, f"edges/{edge_type_name}/edge_index", edge_index)
        _write_parquet(out, f"edges/{edge_type_name}/edge_index.parquet", mapped[["src_index", "dst_index"]])
        row_map = mapped[["edge_pos", "relation", "x_id", "y_id", "edge_key", "source", "credibility"]]
        _write_parquet(out, f"edges/{edge_type_name}/edge_row_map.parquet", row_map)
        edge_attr = mapped[["edge_pos", "credibility"]].copy()
        edge_attr["credibility"] = pd.to_numeric(edge_attr["credibility"], errors="coerce").fillna(0).astype("int64")
        _write_parquet(out, f"edges/{edge_type_name}/edge_attr.parquet", edge_attr)
        stats = {
            "relation": relation,
            "edge_type": [src_type, relation, dst_type],
            "selected_rows": int(len(edge_df)),
            "edge_count": int(edge_index.shape[1]),
            "edge_index_shape": list(edge_index.shape),
            "missing_src_endpoint_count": missing_src,
            "missing_dst_endpoint_count": missing_dst,
            "relation_mismatch_count": rel_mismatch,
            "endpoint_type_mismatch_count": endpoint_mismatch,
            "source_edge_path": kg.edge_path(relation),
            "bounded": config.max_edges_per_relation is not None,
            "max_edges_per_relation": config.max_edges_per_relation,
            "edge_key_sha256": _hash_sequence(mapped["edge_key"]),
        }
        out.write_json(f"edges/{edge_type_name}/edge_stats.json", stats)
        out.write_json(
            f"edges/{edge_type_name}/reverse_edge_type.json",
            {"forward_edge_type": [src_type, relation, dst_type], "reverse_edge_type": [dst_type, f"rev_{relation}", src_type]},
        )

        if config.include_reverse_edges:
            rev_type_name = _edge_type_dir(dst_type, f"rev_{relation}", src_type)
            rev_edge_index = edge_index[[1, 0], :]
            _write_tensor(out, f"reverse_edges/{rev_type_name}/edge_index", rev_edge_index)
            rev_row_map = row_map[["edge_pos", "relation", "x_id", "y_id", "edge_key"]].rename(columns={"edge_pos": "forward_edge_pos"})
            rev_row_map.insert(0, "edge_pos", np.arange(len(rev_row_map), dtype=np.int64))
            _write_parquet(out, f"reverse_edges/{rev_type_name}/edge_row_map.parquet", rev_row_map)
            out.write_json(
                f"reverse_edges/{rev_type_name}/reverse_of.json",
                {"reverse_edge_type": [dst_type, f"rev_{relation}", src_type], "forward_edge_type": [src_type, relation, dst_type]},
            )
            heterodata_payload["edge_types"][str((dst_type, f"rev_{relation}", src_type))] = {
                "edge_type": [dst_type, f"rev_{relation}", src_type],
                "edge_count": int(rev_edge_index.shape[1]),
                "edge_index_npy": f"reverse_edges/{rev_type_name}/edge_index.npy",
                "edge_index_pt": f"reverse_edges/{rev_type_name}/edge_index.pt" if torch is not None else None,
                "edge_index_shape": list(rev_edge_index.shape),
                "edge_row_map": f"reverse_edges/{rev_type_name}/edge_row_map.parquet",
                "reverse_of": [src_type, relation, dst_type],
            }

        edge_counts[relation] = int(edge_index.shape[1])
        edge_type_rows.append(
            {
                "relation": relation,
                "x_type": src_type,
                "y_type": dst_type,
                "pyg_src_type": src_type,
                "pyg_relation": relation,
                "pyg_dst_type": dst_type,
                "kind": rel.kind.value,
                "direct": bool(rel.direct),
                "canonical_edge_path": kg.edge_path(relation),
                "canonical_evidence_path": f"{config.kg_root.rstrip('/')}/evidence/{relation}.parquet",
                "edge_count": int(edge_index.shape[1]),
                "has_reverse": bool(config.include_reverse_edges),
                "reverse_relation_name": f"rev_{relation}",
                "include_in_txgnn_legacy": relation in {"molecule_treats_disease", "molecule_contraindicated_for_disease", "disease_associated_gene", "molecule_targets_gene"},
            }
        )
        heterodata_payload["edge_types"][str((src_type, relation, dst_type))] = {
            "edge_type": [src_type, relation, dst_type],
            "edge_count": int(edge_index.shape[1]),
            "edge_index_npy": f"edges/{edge_type_name}/edge_index.npy",
            "edge_index_pt": f"edges/{edge_type_name}/edge_index.pt" if torch is not None else None,
            "edge_index_shape": list(edge_index.shape),
            "edge_row_map": f"edges/{edge_type_name}/edge_row_map.parquet",
        }

    for node_type, count in node_counts.items():
        heterodata_payload["node_types"][node_type] = {
            "num_nodes": count,
            "id_to_index": f"node_maps/{node_type}.id_to_index.parquet",
            "index_to_id": f"node_maps/{node_type}.index_to_id.parquet",
        }

    _write_schema_artifacts(out, node_types, edge_type_rows)
    feature_metadata = _write_feature_maps(config, kg, out, node_maps, issues)
    feature_policy = _build_feature_policy_manifest(
        config=config,
        node_types=node_types,
        relations=tuple(edge_counts),
        feature_metadata=feature_metadata,
        node_counts=node_counts,
        edge_counts=edge_counts,
        artifact_mode=artifact_mode,
    )
    heterodata_payload["feature_policy"] = feature_policy
    if artifact_mode in {"heterodata", "both"}:
        _write_heterodata_artifact(out, heterodata_payload, node_counts, node_maps, config)
    else:
        _write_sidecar_artifact_metadata(out, heterodata_payload, config)

    error_count = sum(1 for issue in issues if issue.severity == "error")
    warning_count = sum(1 for issue in issues if issue.severity == "warning")
    validation_report = {
        "build_name": config.build_name,
        "generated_at": _now_iso(),
        "kg_root": config.kg_root,
        "output_root": output_root,
        "status": "pass" if error_count == 0 else "fail",
        "error_count": error_count,
        "warning_count": warning_count,
        "node_counts": node_counts,
        "edge_counts": edge_counts,
        "checks": {
            "node_id_uniqueness": error_count == 0,
            "edge_index_shape": all(v >= 0 for v in edge_counts.values()),
            "endpoint_consistency": not any(issue.check.endswith("endpoint_in_node_map") for issue in issues),
            "reproducibility_hashes_present": True,
            "sidecar_artifact_present": True,
            "heterodata_pickle_required": artifact_mode in {"heterodata", "both"},
        },
        "issues": [asdict(issue) for issue in issues],
    }
    out.write_json("validation_report.json", validation_report)
    out.write_text("validation_report.md", _validation_markdown(validation_report))


    manifest = {
        "artifact_format": "jouvencekb-kg-pyg-v1",
        "build_name": config.build_name,
        "generated_at": _now_iso(),
        "kg_root": config.kg_root,
        "output_root": output_root,
        "bounded": config.max_nodes_per_type is not None and config.max_edges_per_relation is not None,
        "artifact_mode": artifact_mode,
        "requested_artifact_mode": config.artifact_mode,
        "limits": {
            "max_nodes_per_type": config.max_nodes_per_type,
            "max_edges_per_relation": config.max_edges_per_relation,
        },
        "training_graph_exclusion_policy": {
            "excluded_node_types_by_default": sorted(excluded_node_values),
            "graph_disconnected_relations_by_default": sorted(GRAPH_DISCONNECTED_RELATIONS),
            "include_provenance_node_types": config.include_provenance_node_types,
            "excluded_requested_node_types": excluded_requested_node_types if not config.include_provenance_node_types else [],
            "excluded_requested_relations": excluded_requested_relations if not config.include_provenance_node_types else [],
            "policy_label": "metadata-only/non-training; retain canonical files with exporter exclusions unless explicitly audited",
            "rationale": "paper and dataset entities are provenance/catalog metadata only, not message-passing graph nodes",
        },
        "node_types": list(node_types),
        "relations": list(edge_counts),
        "node_stats": node_stats,
        "edge_counts": edge_counts,
        "feature_tables": feature_metadata,
        "node_embeddings": feature_policy["node_embeddings"],
        "edge_embeddings": feature_policy["edge_embeddings"],
        "missing_feature_policy": feature_policy["missing_feature_policy"],
        "tensor_formats": {
            "edge_index.npy": "always written, shape [2, num_edges], int64",
            "edge_index.pt": "written when torch is importable; otherwise omitted",
            "heterodata/full_graph.pt": "written only for artifact_mode=heterodata|both (or auto on bounded exports) when torch_geometric is importable; production sidecar mode does not require this pickle",
            "heterodata node x": "real embeddings where available; learned torch.nn.Embedding fallback rows otherwise",
            "heterodata edge_attr": "real edge embeddings where available; learned torch.nn.Embedding fallback rows otherwise",
        },
        "validation_report": "validation_report.json",
    }
    out.write_json("manifest.json", manifest)
    out.write_text("README.md", _readme_text(config, manifest))
    out.write_json("build_config.json", asdict(config))

    return BuildResult(
        output_root=output_root,
        manifest_path=f"{output_root.rstrip('/')}/manifest.json",
        validation_report_path=f"{output_root.rstrip('/')}/validation_report.json",
        node_counts=node_counts,
        edge_counts=edge_counts,
        issues=issues,
    )


def _resolve_node_types(config: BuildConfig, available_nodes: Iterable[str]) -> tuple[str, ...]:
    available = set(available_nodes)
    selected = tuple(dict.fromkeys(config.node_types))
    if not config.include_provenance_node_types:
        excluded = {node_type.value for node_type in TRAINING_GRAPH_EXCLUDED_NODE_TYPES}
        selected = tuple(node_type for node_type in selected if node_type not in excluded)
    missing = [node_type for node_type in selected if node_type not in available]
    unknown = [node_type for node_type in selected if node_type not in {nt.value for nt in NODE_TYPES}]
    if unknown:
        raise ValueError(f"Unknown node types: {unknown}")
    if missing:
        raise FileNotFoundError(f"Selected node files not present under canonical KG root: {missing}")
    return selected


def _resolve_relations(config: BuildConfig, available_edges: Iterable[str]) -> tuple[str, ...]:
    available = set(available_edges)
    selected = tuple(dict.fromkeys(config.relations))
    unknown = [relation for relation in selected if relation not in RELATION_BY_NAME]
    if not config.include_provenance_node_types:
        selected = tuple(relation for relation in selected if relation not in GRAPH_DISCONNECTED_RELATIONS)
    missing = [relation for relation in selected if relation not in available]
    inactive = [relation for relation in selected if RELATION_BY_NAME[relation].status != RelationStatus.ACTIVE]
    if unknown:
        raise ValueError(f"Unknown relations: {unknown}")
    if inactive:
        raise ValueError(f"Selected non-active relations: {inactive}")
    if missing:
        raise FileNotFoundError(f"Selected edge files not present under canonical KG root: {missing}")
    return selected


def _validate_artifact_mode(mode: str) -> None:
    if mode not in ARTIFACT_MODES:
        raise ValueError(f"Unknown artifact_mode={mode!r}; expected one of {ARTIFACT_MODES}")


def _resolve_artifact_mode(config: BuildConfig) -> str:
    if config.artifact_mode != "auto":
        return config.artifact_mode
    if config.max_edges_per_relation is None:
        return "sidecar"
    return "heterodata"


def _read_parquet_limited(path: str, fs: fsspec.AbstractFileSystem, columns: list[str], limit: int | None) -> pd.DataFrame:
    parquet = pq.ParquetFile(path, filesystem=fs)
    if limit is None:
        return parquet.read(columns=columns).to_pandas()
    batches: list[pa.RecordBatch] = []
    remaining = limit
    for batch in parquet.iter_batches(batch_size=min(remaining, 65_536), columns=columns):
        if remaining <= 0:
            break
        if batch.num_rows > remaining:
            batch = batch.slice(0, remaining)
        batches.append(batch)
        remaining -= batch.num_rows
    if not batches:
        schema = parquet.schema_arrow
        selected = pa.schema([schema.field(name) for name in columns if name in schema.names])
        return pa.Table.from_batches([], schema=selected).to_pandas()
    return pa.Table.from_batches(batches).to_pandas()


def _parquet_num_rows(path: str, fs: fsspec.AbstractFileSystem) -> int:
    return int(pq.ParquetFile(path, filesystem=fs).metadata.num_rows)


def _write_production_plan_manifest(
    *,
    config: BuildConfig,
    kg: Any,
    out: "_OutputRoot",
    output_root: str,
    node_types: tuple[str, ...],
    relations: tuple[str, ...],
    excluded_node_values: set[str],
    excluded_requested_node_types: list[str],
    excluded_requested_relations: list[str],
) -> dict[str, Any]:
    node_rows = {node_type: _parquet_num_rows(kg._node_internal(node_type), kg.fs) for node_type in node_types}
    edge_rows = {relation: _parquet_num_rows(kg._edge_internal(relation), kg.fs) for relation in relations}
    remote_output_root = config.remote_output_root or posixpath.join("gs://jouvencekb/staging/ml/pyg", config.build_name)
    remote_command = _remote_build_command(config, node_types, relations, remote_output_root)
    edge_index_bytes = {relation: int(rows) * 2 * 8 for relation, rows in edge_rows.items()}
    edge_row_map_bytes_estimate = {relation: int(rows) * 128 for relation, rows in edge_rows.items()}
    node_map_bytes_estimate = {node_type: int(rows) * 96 for node_type, rows in node_rows.items()}
    manifest = {
        "artifact_format": "jouvencekb-kg-pyg-production-plan-v1",
        "build_name": config.build_name,
        "generated_at": _now_iso(),
        "kg_root": config.kg_root,
        "output_root": output_root,
        "plan_only": True,
        "metadata_source": "Parquet footer metadata only; no table materialization or full GCS-FUSE scan",
        "node_types": list(node_types),
        "relations": list(relations),
        "node_rows": node_rows,
        "edge_rows": edge_rows,
        "total_node_rows": int(sum(node_rows.values())),
        "total_edge_rows": int(sum(edge_rows.values())),
        "estimated_bytes": {
            "edge_index_int64_forward_only": edge_index_bytes,
            "edge_row_map_rough": edge_row_map_bytes_estimate,
            "node_maps_rough": node_map_bytes_estimate,
            "total_rough_without_heterodata_or_embeddings": int(sum(edge_index_bytes.values()) + sum(edge_row_map_bytes_estimate.values()) + sum(node_map_bytes_estimate.values())),
        },
        "training_graph_exclusion_policy": {
            "excluded_node_types_by_default": sorted(excluded_node_values),
            "graph_disconnected_relations_by_default": sorted(GRAPH_DISCONNECTED_RELATIONS),
            "include_provenance_node_types": config.include_provenance_node_types,
            "excluded_requested_node_types": excluded_requested_node_types if not config.include_provenance_node_types else [],
            "excluded_requested_relations": excluded_requested_relations if not config.include_provenance_node_types else [],
            "policy_label": "metadata-only/non-training; retain canonical files with exporter exclusions unless explicitly audited",
            "rationale": "paper and dataset entities are provenance/catalog metadata only, not message-passing graph nodes",
        },
        "remote_runner": config.remote_runner,
        "remote_output_root": remote_output_root,
        "remote_command": remote_command,
        "artifact_mode": _resolve_artifact_mode(config),
        "next_gate": "Run the remote command on a bucket-local/GCP worker in sidecar mode, then run manage_db.run_pyg_gnn_smoke against one selected relation/sidecar sample instead of requiring a full HeteroData pickle.",
    }
    out.write_json("production_plan_manifest.json", manifest)
    out.write_text("production_plan_manifest.md", _production_plan_markdown(manifest))
    out.write_json("build_config.json", asdict(config))
    return manifest


def _remote_build_command(config: BuildConfig, node_types: tuple[str, ...], relations: tuple[str, ...], remote_output_root: str) -> str:
    args = [
        "uv run python -m manage_db.build_pyg_export",
        f"--kg-root {config.kg_root}",
        f"--output-root {remote_output_root}",
        "--node-types " + " ".join(node_types),
        "--relations " + " ".join(relations),
        f"--max-nodes-per-type {config.max_nodes_per_type or 0}",
        f"--max-edges-per-relation {config.max_edges_per_relation or 0}",
        f"--build-name {config.build_name}",
        f"--artifact-mode {_resolve_artifact_mode(config)}",
    ]
    if config.feature_tables:
        args.append("--feature-tables " + " ".join(config.feature_tables))
    if not config.include_reverse_edges:
        args.append("--no-reverse-edges")
    if not config.strict:
        args.append("--no-strict")
    if config.sort_node_ids:
        args.append("--sort-node-ids")
    if config.embedding_features_root:
        args.append(f"--embedding-features-root {config.embedding_features_root}")
    if config.learned_fallback_config_path:
        args.append(f"--learned-fallback-config-path {config.learned_fallback_config_path}")
    if config.include_provenance_node_types:
        args.append("--include-provenance-node-types")
    return " \\n  ".join(args)


def _production_plan_markdown(manifest: dict[str, Any]) -> str:
    lines = [
        "# PyG production-scale export plan",
        "",
        f"Build name: `{manifest['build_name']}`",
        f"Generated: `{manifest['generated_at']}`",
        f"KG root: `{manifest['kg_root']}`",
        f"Plan artifact root: `{manifest['output_root']}`",
        "",
        "This is a metadata-only planning/dry-run manifest. It reads Parquet footers for row counts and does not materialize full GCS-FUSE tables on the Mac.",
        "",
        f"Total node rows selected: `{manifest['total_node_rows']}`",
        f"Total edge rows selected: `{manifest['total_edge_rows']}`",
        f"Rough bytes before HeteroData/embeddings: `{manifest['estimated_bytes']['total_rough_without_heterodata_or_embeddings']}`",
        "",
        "## Remote bucket-local build command",
        "",
        "```bash",
        manifest["remote_command"],
        "```",
        "",
        "## Training-graph policy",
        "",
        f"Provenance node types included: `{manifest['training_graph_exclusion_policy']['include_provenance_node_types']}`",
        f"Default excluded node types: `{', '.join(manifest['training_graph_exclusion_policy']['excluded_node_types_by_default'])}`",
        "",
        f"Next gate: {manifest['next_gate']}",
    ]
    return "\n".join(lines) + "\n"


def _validate_edge_columns(df: pd.DataFrame, relation: str) -> None:
    required = {"x_id", "x_type", "y_id", "y_type", "relation"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"edges/{relation}.parquet missing columns: {missing}")


def _write_schema_artifacts(out: "_OutputRoot", node_types: tuple[str, ...], edge_type_rows: list[dict[str, Any]]) -> None:
    out.write_json("schema/node_types.json", {"node_types": list(node_types)})
    out.write_json("schema/edge_types.json", {"edge_types": [[r["x_type"], r["relation"], r["y_type"]] for r in edge_type_rows]})
    _write_parquet(out, "schema/relation_to_edge_type.parquet", pd.DataFrame(edge_type_rows))
    out.write_json(
        "schema/feature_tables.json",
        {"feature_table_contract": "features/*.parquet mapped by node_id/node_type or inferred node type prefix"},
    )


def _write_feature_maps(
    config: BuildConfig,
    kg: Any,
    out: "_OutputRoot",
    node_maps: dict[str, pd.DataFrame],
    issues: list[ValidationIssue],
) -> dict[str, Any]:
    feature_root = kg._join("features")
    selected = list(config.feature_tables)
    if not selected and kg.fs.exists(feature_root):
        selected = []  # explicit by default; avoid accidental large feature scans
    metadata: dict[str, Any] = {}
    for table_name in selected:
        rel_path = f"features/{table_name}.parquet"
        internal = kg._join(rel_path)
        if not kg.fs.exists(internal):
            issues.append(ValidationIssue("warning", "feature_table_exists", f"Feature table missing: {rel_path}"))
            continue
        table = pq.read_table(internal, filesystem=kg.fs)
        df = table.to_pandas()
        if "node_id" in df.columns:
            node_id_col = "node_id"
        elif "id" in df.columns:
            node_id_col = "id"
        else:
            issues.append(ValidationIssue("warning", "feature_node_id_column", f"Feature table has no node_id/id column: {table_name}"))
            continue
        if "node_type" in df.columns:
            candidate_types = sorted(set(df["node_type"].dropna().astype(str)) & set(node_maps))
        else:
            prefix = table_name.split("_")[0]
            candidate_types = [prefix] if prefix in node_maps else []
        for node_type in candidate_types:
            sub = df if "node_type" not in df.columns else df[df["node_type"].astype(str) == node_type]
            node_map = node_maps[node_type][["id", "node_index"]].rename(columns={"id": node_id_col})
            row_map = sub[[node_id_col]].astype({node_id_col: "string"}).merge(node_map, on=node_id_col, how="inner")
            row_map = row_map.rename(columns={node_id_col: "node_id"}).drop_duplicates("node_index").reset_index(drop=True)
            row_map.insert(0, "feature_row", np.arange(len(row_map), dtype=np.int64))
            out_path = f"node_features/{node_type}/{table_name}.row_map.parquet"
            _write_parquet(out, out_path, row_map)
            manifest = {
                "feature_table": table_name,
                "node_type": node_type,
                "source_path": f"{config.kg_root.rstrip('/')}/{rel_path}",
                "source_columns": list(df.columns),
                "mapped_rows": int(len(row_map)),
                "node_count": int(len(node_maps[node_type])),
                "coverage_fraction": float(len(row_map) / max(len(node_maps[node_type]), 1)),
                "row_map": out_path,
                "tensor_policy": "raw text/sequence/object values remain sidecars; numeric/vector tensors can be added beside this map",
            }
            out.write_json(f"node_features/{node_type}/{table_name}.feature_manifest.json", manifest)
            metadata[f"{node_type}/{table_name}"] = manifest
    return metadata


def _build_feature_policy_manifest(
    *,
    config: BuildConfig,
    node_types: tuple[str, ...],
    relations: tuple[str, ...],
    feature_metadata: dict[str, Any],
    node_counts: dict[str, int],
    edge_counts: dict[str, int],
    artifact_mode: str,
) -> dict[str, Any]:
    """Describe embedding/feature availability without materializing full tensors."""

    node_embeddings = _collect_node_embedding_metadata(config.embedding_features_root)
    edge_embeddings = _collect_edge_embedding_metadata(config.embedding_features_root)
    node_available = {node_type: node_embeddings.get(node_type, []) for node_type in node_types}
    edge_available = {relation: edge_embeddings.get(relation, []) for relation in relations}
    policy: dict[str, Any] = {
        "scope": "selected PyG export node/edge types",
        "source_of_truth": "artifact audit t_e4f08d5a plus current exporter metadata scan",
        "materialization_policy": (
            "Manifest generation records Parquet footer/schema metadata and sidecar mapping contracts only; "
            "it must not require a full HeteroData pickle or full 100M-edge tensor materialization."
        ),
        "node_types": {},
        "edge_types": {},
    }
    feature_tables_by_node: dict[str, list[dict[str, Any]]] = {node_type: [] for node_type in node_types}
    for entry in feature_metadata.values():
        node_type = entry.get("node_type")
        if node_type in feature_tables_by_node:
            feature_tables_by_node[node_type].append({
                "feature_table": entry.get("feature_table"),
                "status": "available",
                "value_kind": _feature_value_kind(entry.get("source_columns", [])),
                "mapped_rows": int(entry.get("mapped_rows", 0)),
                "node_count": int(entry.get("node_count", 0)),
                "row_map": entry.get("row_map"),
                "source_path": entry.get("source_path"),
            })

    for node_type in node_types:
        available = node_available[node_type]
        absent = [] if available else [{
            "embedding_family": "textual_summary" if node_type in TEXTUAL_SUMMARY_NODE_TYPES_WITH_STAGED_EMBEDDINGS else "foundation_embedding",
            "status": "absent",
            "reason": "no manifest-visible embedding sidecar found for selected node type",
            "fallback": "model-side learned torch.nn.Embedding rows",
        }]
        policy["node_types"][node_type] = {
            "node_count": int(node_counts.get(node_type, 0)),
            "embedding_status": "available" if available else "absent",
            "available_embeddings": available,
            "absent_embeddings": absent,
            "available_feature_values": sorted(feature_tables_by_node[node_type], key=lambda item: item["feature_table"] or ""),
            "intentionally_deferred": DEFERRED_NODE_FEATURES.get(node_type, []),
            "fallback_policy": "use real sidecar rows when present; fill missing node rows with learned fallback embeddings at model materialization time",
            "sidecar_node_id_mapping": {
                "node_map": f"node_maps/{node_type}.id_to_index.parquet",
                "join_key": "embedding.node_id == node_maps.id",
                "pyg_index_column": "node_index",
            },
        }

    for relation in relations:
        available = edge_available[relation]
        policy["edge_types"][relation] = {
            "edge_count": int(edge_counts.get(relation, 0)),
            "embedding_status": "available" if available else "absent",
            "available_embeddings": available,
            "absent_embeddings": [] if available else [{
                "embedding_family": "relation_value_evidence_mlp",
                "status": "absent",
                "reason": "no manifest-visible edge embedding sidecar found for selected relation",
                "fallback": "model-side learned torch.nn.Embedding rows",
            }],
            "available_feature_values": [
                {"field": "credibility", "status": "available", "value_kind": "numeric", "source": "edges/{relation}.parquet and edge_attr.parquet"},
                {"field": "source", "status": "available", "value_kind": "categorical", "source": "edge_row_map.parquet"},
            ],
            "intentionally_deferred": [{"field": "all_evidence_payload_tensor", "status": "deferred", "reason": "rich evidence remains sidecar/provenance, not dense tensor hot path"}],
            "fallback_policy": "use real edge embedding rows where edge_key matches; fill missing edge rows with learned fallback embeddings at model materialization time",
            "sidecar_edge_id_mapping": {
                "edge_row_map": "edges/{x_type}__{relation}__{y_type}/edge_row_map.parquet",
                "join_key": "embedding.edge_key == edge_row_map.edge_key",
                "pyg_index_column": "edge_pos",
                "reverse_edge_mapping": "reverse_edges/*/edge_row_map.parquet carries forward_edge_pos for reverse stores",
            },
        }

    return {
        "artifact_mode": artifact_mode,
        "embedding_features_root": config.embedding_features_root,
        "node_embeddings": node_available,
        "edge_embeddings": edge_available,
        "missing_feature_policy": policy,
    }


def _collect_node_embedding_metadata(features_root: str | None) -> dict[str, list[dict[str, Any]]]:
    if not features_root:
        return {}
    fs, root_path = url_to_fs(features_root.rstrip("/"))
    patterns = (
        posixpath.join(root_path.rstrip("/"), "embeddings", "*.parquet"),
        posixpath.join(root_path.rstrip("/"), "embeddings", "**", "*.parquet"),
    )
    metadata: dict[str, list[dict[str, Any]]] = {}
    paths = sorted({path for pattern in patterns for path in fs.glob(pattern)})
    for path in paths:
        if "/reports/" in path:
            continue
        parquet = pq.ParquetFile(path, filesystem=fs)
        names = set(parquet.schema_arrow.names)
        if not {"node_id", "node_type", "embedding"}.issubset(names):
            continue
        embedding_dim = _first_parquet_scalar(parquet, "embedding_dim")
        embedding_dtype = _first_parquet_scalar(parquet, "embedding_dtype")
        node_type = str(_first_parquet_scalar(parquet, "node_type") or "unknown")
        modality = str(_first_parquet_scalar(parquet, "modality") or "unknown")
        model = str(_first_parquet_scalar(parquet, "embedding_model") or "unknown")
        version = str(_first_parquet_scalar(parquet, "embedding_version") or "unknown")
        if model == "unknown" or version == "unknown" or modality == "unknown":
            parts = path.split("/")
            try:
                text_index = parts.index("text")
                modality = modality if modality != "unknown" else "text"
                node_type = node_type if node_type != "unknown" else parts[text_index + 1]
                model = model if model != "unknown" else parts[text_index + 2]
                version = version if version != "unknown" else parts[text_index + 3]
            except (ValueError, IndexError):
                pass
        metadata.setdefault(node_type, []).append({
            "status": "available",
            "modality": modality,
            "embedding_model": model,
            "embedding_version": version,
            "path": _display_path(path, features_root, fs),
            "rows": int(parquet.metadata.num_rows),
            "embedding_dim": _coerce_optional_int(embedding_dim),
            "embedding_dtype": str(embedding_dtype) if embedding_dtype is not None else None,
            "schema_columns": parquet.schema_arrow.names,
            "embedding_dim_column": "embedding_dim" if "embedding_dim" in names else None,
            "embedding_dtype_column": "embedding_dtype" if "embedding_dtype" in names else None,
            "sidecar_node_id_mapping": {"embedding_join_key": "node_id", "node_map_join_key": "id", "pyg_index_column": "node_index"},
        })
    return metadata


def _collect_edge_embedding_metadata(features_root: str | None) -> dict[str, list[dict[str, Any]]]:
    if not features_root:
        return {}
    fs, root_path = url_to_fs(features_root.rstrip("/"))
    pattern = posixpath.join(root_path.rstrip("/"), "edge_embeddings", "**", "*.parquet")
    metadata: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(fs.glob(pattern)):
        parquet = pq.ParquetFile(path, filesystem=fs)
        names = set(parquet.schema_arrow.names)
        if not {"relation", "edge_key", "embedding"}.issubset(names):
            continue
        embedding_dim = _first_parquet_scalar(parquet, "embedding_dim")
        embedding_dtype = _first_parquet_scalar(parquet, "embedding_dtype")
        parts = path.split("/")
        try:
            relation_index = parts.index("by_relation") + 1
            relation = parts[relation_index]
            model = parts[relation_index + 1]
            version = parts[relation_index + 2]
        except (ValueError, IndexError):
            relation, model, version = "unknown", "unknown", "unknown"
        metadata.setdefault(relation, []).append({
            "status": "available",
            "modality": "edge_evidence",
            "embedding_model": model,
            "embedding_version": version,
            "path": _display_path(path, features_root, fs),
            "rows": int(parquet.metadata.num_rows),
            "embedding_dim": _coerce_optional_int(embedding_dim),
            "embedding_dtype": str(embedding_dtype) if embedding_dtype is not None else None,
            "schema_columns": parquet.schema_arrow.names,
            "embedding_dim_column": "embedding_dim" if "embedding_dim" in names else None,
            "embedding_dtype_column": "embedding_dtype" if "embedding_dtype" in names else None,
            "sidecar_edge_id_mapping": {"embedding_join_key": "edge_key", "edge_row_map_join_key": "edge_key", "pyg_index_column": "edge_pos"},
        })
    return metadata


def _first_parquet_scalar(parquet: pq.ParquetFile, column: str) -> Any | None:
    """Read one scalar sidecar metadata value without loading embeddings/full tables."""

    if column not in parquet.schema_arrow.names or parquet.metadata.num_rows == 0:
        return None
    for batch in parquet.iter_batches(batch_size=1, columns=[column]):
        values = batch.column(0).to_pylist()
        return values[0] if values else None
    return None


def _coerce_optional_int(value: Any | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _feature_value_kind(columns: list[str]) -> str:
    lower = {str(column).lower() for column in columns}
    if any(marker in column for column in lower for marker in {"embedding", "fingerprint", "on_bits"}):
        return "sparse_vector"
    if any(marker in column for column in lower for marker in {"score", "count", "length", "n_bits", "radius", "credibility"}):
        return "numeric"
    if any(marker in column for column in lower for marker in {"sequence", "protein", "transcript"}):
        return "sequence/raw"
    if any(marker in column for column in lower for marker in {"summary", "description", "text"}):
        return "text/raw"
    return "categorical"


def _display_path(path: str, features_root: str, fs: fsspec.AbstractFileSystem) -> str:
    if features_root.startswith(("gs://", "s3://")):
        protocol = fs.protocol[0] if isinstance(fs.protocol, tuple) else fs.protocol
        return f"{protocol}://{path}" if not path.startswith(f"{protocol}://") else path
    return path


def _write_heterodata_artifact(
    out: "_OutputRoot",
    payload: dict[str, Any],
    node_counts: dict[str, int],
    node_maps: dict[str, pd.DataFrame],
    config: BuildConfig,
) -> None:
    if HeteroData is None or torch is None:
        out.write_json("heterodata/full_graph.metadata.json", payload)
        return

    fallback_config = _load_learned_fallback_config(config)
    node_embeddings = _load_node_embedding_tables(config.embedding_features_root)
    edge_embeddings = _load_edge_embedding_tables(config.embedding_features_root)
    data = HeteroData()
    node_policy: dict[str, Any] = {}
    edge_policy: dict[str, Any] = {}
    node_target_dim = _global_embedding_dim(node_embeddings, int(fallback_config.get("node_fallback", {}).get("dim", 256)), "node")

    for node_type, count in node_counts.items():
        data[node_type].num_nodes = count
        x, policy = _node_feature_tensor(node_type, node_maps[node_type], node_embeddings, fallback_config, config.fallback_seed, node_target_dim)
        data[node_type].x = x
        node_policy[node_type] = policy
        with out.open(f"heterodata/node_x/{node_type}.npy", "wb") as fh:
            np.save(fh, x.detach().cpu().numpy())

    for edge_payload in payload["edge_types"].values():
        edge_type = tuple(edge_payload["edge_type"])
        with out.open(edge_payload["edge_index_npy"], "rb") as fh:
            edge_index = np.load(fh)
        data[edge_type].edge_index = torch.as_tensor(edge_index, dtype=torch.long)
        edge_attr, policy = _edge_attr_tensor(out, edge_payload, edge_embeddings, fallback_config, config.fallback_seed)
        data[edge_type].edge_attr = edge_attr
        edge_policy[str(edge_type)] = policy
        edge_type_name = _edge_type_dir(*edge_type)
        with out.open(f"heterodata/edge_attr/{edge_type_name}.npy", "wb") as fh:
            np.save(fh, edge_attr.detach().cpu().numpy())

    payload["node_embedding_policy"] = node_policy
    payload["edge_embedding_policy"] = edge_policy
    payload["learned_fallback_config"] = fallback_config
    out.write_json("heterodata/full_graph.metadata.json", payload)
    _write_pickle(out, "heterodata/full_graph.pt", data)


def _write_sidecar_artifact_metadata(out: "_OutputRoot", payload: dict[str, Any], config: BuildConfig) -> None:
    payload = dict(payload)
    payload["artifact_mode"] = "sidecar"
    payload["heterodata_pickle"] = None
    payload["load_policy"] = (
        "Relation-wise sidecars are the production artifact. Load edge_index.npy with mmap_mode='r' "
        "and join node_maps/edge_row_map Parquets for one relation or bounded subgraph; do not "
        "materialize a full HeteroData pickle for no-cap exports."
    )
    payload["smoke_loader"] = "manage_db.run_pyg_gnn_smoke can load a selected relation from sidecars when heterodata/full_graph.pt is absent"
    out.write_json("sidecar_artifact.metadata.json", payload)


def _load_learned_fallback_config(config: BuildConfig) -> dict[str, Any]:
    default = {
        "node_fallback": {"module": "torch.nn.Embedding", "dim": 256},
        "edge_fallback": {"module": "torch.nn.Embedding", "dim": 256},
        "policy": "model-side learned fallback only; not a structural placeholder",
    }
    path = config.learned_fallback_config_path
    if not path:
        root = config.embedding_features_root
        if root:
            candidate = posixpath.join(root.rstrip("/"), "embeddings", "reports", "learned_fallback_config.json")
            fs, fs_path = url_to_fs(candidate)
            if fs.exists(fs_path):
                path = candidate
    if not path:
        return default
    fs, fs_path = url_to_fs(path)
    with fs.open(fs_path, "r") as fh:
        payload = json.load(fh)
    for key in ("node_fallback", "edge_fallback"):
        payload.setdefault(key, {})
        payload[key].setdefault("module", "torch.nn.Embedding")
        payload[key].setdefault("dim", default[key]["dim"])
    payload.setdefault("policy", default["policy"])
    return payload


def _load_node_embedding_tables(features_root: str | None) -> dict[str, dict[str, np.ndarray]]:
    if not features_root:
        return {}
    fs, root_path = url_to_fs(features_root.rstrip("/"))
    patterns = (
        posixpath.join(root_path.rstrip("/"), "embeddings", "*.parquet"),
        posixpath.join(root_path.rstrip("/"), "embeddings", "**", "*.parquet"),
    )
    tables: dict[str, dict[str, np.ndarray]] = {}
    paths = sorted({path for pattern in patterns for path in fs.glob(pattern)})
    for path in paths:
        if "/reports/" in path:
            continue
        parquet = pq.ParquetFile(path, filesystem=fs)
        if not {"node_id", "node_type", "embedding"}.issubset(parquet.schema_arrow.names):
            continue
        table = pq.read_table(path, columns=["node_id", "node_type", "embedding"], filesystem=fs)
        df = table.to_pandas()
        for row in df.itertuples(index=False):
            node_type = str(row.node_type)
            node_id = str(row.node_id)
            tables.setdefault(node_type, {})[node_id] = _embedding_array(row.embedding, f"{path}:{node_id}")
    return tables


def _load_edge_embedding_tables(features_root: str | None) -> dict[str, dict[str, np.ndarray]]:
    if not features_root:
        return {}
    fs, root_path = url_to_fs(features_root.rstrip("/"))
    pattern = posixpath.join(root_path.rstrip("/"), "edge_embeddings", "**", "*.parquet")
    tables: dict[str, dict[str, np.ndarray]] = {}
    for path in sorted(fs.glob(pattern)):
        table = pq.read_table(path, columns=["relation", "edge_key", "embedding"], filesystem=fs)
        df = table.to_pandas()
        for row in df.itertuples(index=False):
            relation = str(row.relation)
            edge_key = str(row.edge_key)
            tables.setdefault(relation, {})[edge_key] = _embedding_array(row.embedding, f"{path}:{edge_key}")
    return tables



def _global_embedding_dim(embeddings: dict[str, dict[str, np.ndarray]], fallback_dim: int, label: str) -> int:
    dims = {int(vector.shape[0]) for by_key in embeddings.values() for vector in by_key.values()}
    if not dims:
        return fallback_dim
    if len(dims) != 1:
        raise ValueError(f"Mixed {label} embedding dimensions across loaded tables: {sorted(dims)}")
    return next(iter(dims))

def _embedding_array(value: Any, label: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"Embedding must be a non-empty 1D vector for {label}")
    if not np.isfinite(arr).all():
        raise ValueError(f"Embedding contains non-finite values for {label}")
    return arr


def _node_feature_tensor(
    node_type: str,
    node_map: pd.DataFrame,
    node_embeddings: dict[str, dict[str, np.ndarray]],
    fallback_config: dict[str, Any],
    seed: int,
    target_dim: int | None = None,
) -> tuple["torch.Tensor", dict[str, Any]]:
    real = node_embeddings.get(node_type, {})
    real_dims = {int(v.shape[0]) for v in real.values()}
    if len(real_dims) > 1:
        raise ValueError(f"Mixed node embedding dimensions for {node_type}: {sorted(real_dims)}")
    fallback_dim = int(fallback_config.get("node_fallback", {}).get("dim", 256))
    dim = next(iter(real_dims), target_dim or fallback_dim)
    fallback = _learned_embedding(int(len(node_map)), dim, seed, f"node:{node_type}")
    rows: list[torch.Tensor] = []
    real_rows = 0
    for i, node_id in enumerate(node_map["id"].astype(str).tolist()):
        vector = real.get(node_id)
        if vector is None:
            rows.append(fallback.weight[i].detach().clone())
        else:
            if int(vector.shape[0]) != dim:
                raise ValueError(f"Node embedding dimension mismatch for {node_type}:{node_id}")
            rows.append(torch.as_tensor(vector, dtype=torch.float32))
            real_rows += 1
    tensor = torch.stack(rows, dim=0) if rows else torch.empty((0, dim), dtype=torch.float32)
    return tensor, {
        "dim": dim,
        "real_rows": real_rows,
        "fallback_rows": int(len(node_map) - real_rows),
        "fallback_module": fallback_config.get("node_fallback", {}).get("module", "torch.nn.Embedding"),
        "fallback_config_dim": fallback_dim,
        "policy": fallback_config.get("policy"),
    }


def _edge_attr_tensor(
    out: "_OutputRoot",
    edge_payload: dict[str, Any],
    edge_embeddings: dict[str, dict[str, np.ndarray]],
    fallback_config: dict[str, Any],
    seed: int,
) -> tuple["torch.Tensor", dict[str, Any]]:
    edge_type = tuple(edge_payload["edge_type"])
    relation = edge_type[1][4:] if edge_type[1].startswith("rev_") else edge_type[1]
    with out.open(edge_payload["edge_row_map"], "rb") as fh:
        row_map = pq.read_table(fh).to_pandas()
    real = edge_embeddings.get(relation, {})
    real_dims = {int(v.shape[0]) for v in real.values()}
    if len(real_dims) > 1:
        raise ValueError(f"Mixed edge embedding dimensions for {relation}: {sorted(real_dims)}")
    fallback_dim = int(fallback_config.get("edge_fallback", {}).get("dim", 256))
    dim = next(iter(real_dims), fallback_dim)
    fallback = _learned_embedding(int(len(row_map)), dim, seed, f"edge:{edge_type}")
    rows: list[torch.Tensor] = []
    real_rows = 0
    for i, edge_key in enumerate(row_map["edge_key"].astype(str).tolist()):
        vector = real.get(edge_key)
        if vector is None:
            rows.append(fallback.weight[i].detach().clone())
        else:
            if int(vector.shape[0]) != dim:
                raise ValueError(f"Edge embedding dimension mismatch for {edge_key}")
            rows.append(torch.as_tensor(vector, dtype=torch.float32))
            real_rows += 1
    tensor = torch.stack(rows, dim=0) if rows else torch.empty((0, dim), dtype=torch.float32)
    return tensor, {
        "dim": dim,
        "real_rows": real_rows,
        "fallback_rows": int(len(row_map) - real_rows),
        "fallback_module": fallback_config.get("edge_fallback", {}).get("module", "torch.nn.Embedding"),
        "policy": fallback_config.get("policy"),
    }


def _learned_embedding(count: int, dim: int, seed: int, namespace: str) -> "torch.nn.Embedding":
    module = torch.nn.Embedding(count, dim)
    generator = torch.Generator().manual_seed((int(seed) + zlib.crc32(namespace.encode("utf-8"))) % (2**31))
    with torch.no_grad():
        module.weight.normal_(mean=0.0, std=0.02, generator=generator)
    return module


def _write_tensor(out: "_OutputRoot", stem: str, array: np.ndarray) -> None:
    out.makedirs(posixpath.dirname(stem))
    with out.open(f"{stem}.npy", "wb") as fh:
        np.save(fh, array)
    if torch is not None:
        tensor = torch.as_tensor(array, dtype=torch.long)
        with out.open(f"{stem}.pt", "wb") as fh:
            torch.save(tensor, fh)


def _write_pickle(out: "_OutputRoot", rel_path: str, obj: Any) -> None:
    out.makedirs(posixpath.dirname(rel_path))
    with out.open(rel_path, "wb") as fh:
        pickle.dump(obj, fh)


def _write_parquet(out: "_OutputRoot", rel_path: str, df: pd.DataFrame) -> None:
    out.makedirs(posixpath.dirname(rel_path))
    table = pa.Table.from_pandas(df, preserve_index=False)
    with out.open(rel_path, "wb") as fh:
        pq.write_table(table, fh, compression="snappy", write_statistics=True)


def _edge_type_dir(src_type: str, relation: str, dst_type: str) -> str:
    return f"{src_type}__{relation}__{dst_type}"


def _hash_sequence(values: Iterable[Any]) -> str:
    h = hashlib.sha256()
    for value in values:
        h.update(str(value).encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validation_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# PyG export validation report",
        "",
        f"Status: `{report['status']}`",
        f"Generated: {report['generated_at']}",
        f"KG root: `{report['kg_root']}`",
        f"Output root: `{report['output_root']}`",
        "",
        "## Counts",
        "",
        "### Nodes",
    ]
    for name, count in sorted(report["node_counts"].items()):
        lines.append(f"- {name}: {count}")
    lines.extend(["", "### Edges"])
    for name, count in sorted(report["edge_counts"].items()):
        lines.append(f"- {name}: {count}")
    lines.extend(["", "## Issues"])
    if not report["issues"]:
        lines.append("- None")
    for issue in report["issues"]:
        lines.append(f"- {issue['severity']}: {issue['check']} {issue.get('relation') or issue.get('node_type') or ''} count={issue.get('count')} — {issue['message']}")
    return "\n".join(lines) + "\n"


def _readme_text(config: BuildConfig, manifest: dict[str, Any]) -> str:
    return f"""# Canonical KG PyG export ({config.build_name})

Derived from `{config.kg_root}` without reading `.omoc` staging state.

This export is bounded by design:

- max nodes per type: `{config.max_nodes_per_type}`
- max edges per relation: `{config.max_edges_per_relation}`
- node types: `{', '.join(config.node_types)}`
- relations: `{', '.join(config.relations)}`

Training graph exclusion policy: `paper` and `dataset` are provenance/catalog
node types and are excluded, along with any relation touching them, unless this
export was explicitly built with `--include-provenance-node-types` for audit/debug.

Main artifacts:

- `node_maps/*.id_to_index.parquet` and `*.index_to_id.parquet`
- `edges/*/edge_index.npy` plus `edge_index.pt` when PyTorch is installed
- `edges/*/edge_index.parquet` and `edge_row_map.parquet`
- `schema/relation_to_edge_type.parquet`
- `node_features/*/*.feature_manifest.json` for requested feature tables
- `heterodata/full_graph.metadata.json` and optional `heterodata/full_graph.pt`
- `validation_report.json`

Rebuild example:

```bash
uv run python -m manage_db.build_pyg_export \
  --kg-root {config.kg_root} \
  --output-root {manifest['output_root']} \
  --node-types {' '.join(config.node_types)} \
  --relations {' '.join(config.relations)} \
  --max-nodes-per-type {config.max_nodes_per_type or 0} \
  --max-edges-per-relation {config.max_edges_per_relation or 0}
```

Use `--no-strict` only for exploratory pilots where endpoint misses should be
reported and dropped instead of failing the build.
"""


class _OutputRoot:
    def __init__(self, uri: str) -> None:
        self.uri = uri.rstrip("/")
        self.fs, self.path = url_to_fs(self.uri)
        self.path = self.path.rstrip("/")

    def _join(self, rel_path: str) -> str:
        rel_path = rel_path.strip("/")
        if not rel_path:
            return self.path
        if not self.path:
            return rel_path
        return posixpath.join(self.path, rel_path)

    def makedirs(self, rel_path: str) -> None:
        path = self._join(rel_path)
        if path:
            self.fs.makedirs(path, exist_ok=True)

    def open(self, rel_path: str, mode: str):
        parent = posixpath.dirname(rel_path)
        if parent:
            self.makedirs(parent)
        return self.fs.open(self._join(rel_path), mode)

    def write_json(self, rel_path: str, payload: Any) -> None:
        with self.open(rel_path, "w") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.write("\n")

    def write_text(self, rel_path: str, text: str) -> None:
        with self.open(rel_path, "w") as fh:
            fh.write(text)


def _positive_int_or_none(raw: str) -> int | None:
    value = int(raw)
    return None if value <= 0 else value


def parse_args(argv: list[str] | None = None) -> BuildConfig:
    parser = argparse.ArgumentParser(description="Build bounded PyG artifacts from canonical KG Parquets")
    parser.add_argument("--kg-root", default=DEFAULT_KG_ROOT)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--node-types", nargs="+", default=list(DEFAULT_PILOT_NODE_TYPES))
    parser.add_argument("--relations", nargs="+", default=list(DEFAULT_PILOT_RELATIONS))
    parser.add_argument("--max-nodes-per-type", type=_positive_int_or_none, default=50_000, help="0 disables cap")
    parser.add_argument("--max-edges-per-relation", type=_positive_int_or_none, default=100_000, help="0 disables cap")
    parser.add_argument("--feature-tables", nargs="*", default=[])
    parser.add_argument("--build-name", default="pilot")
    parser.add_argument("--no-reverse-edges", action="store_true")
    parser.add_argument("--no-strict", action="store_true")
    parser.add_argument("--sort-node-ids", action="store_true")
    parser.add_argument("--embedding-features-root", default=None, help="KG root containing embeddings/*.parquet (legacy nested embedding layouts are also readable)")
    parser.add_argument("--learned-fallback-config-path", default=None, help="JSON policy declaring torch.nn.Embedding fallback dimensions")
    parser.add_argument("--fallback-seed", type=int, default=20260623)
    parser.add_argument(
        "--include-provenance-node-types",
        action="store_true",
        help="Opt in to exporting provenance/catalog node types such as paper and dataset. Default excludes them from training graph adjacency.",
    )
    parser.add_argument("--plan-only", action="store_true", help="Write a metadata-only production plan/row-count manifest without materializing tensors")
    parser.add_argument("--remote-output-root", default=None, help="Bucket-local output root to embed in the plan-only remote build command")
    parser.add_argument("--remote-runner", default="gcp", help="Runner label recorded in plan-only manifests")
    parser.add_argument("--artifact-mode", choices=ARTIFACT_MODES, default="auto", help="Artifact materialization mode: auto keeps bounded pilots as HeteroData but no-cap production exports as sidecars")
    ns = parser.parse_args(argv)
    return BuildConfig(
        kg_root=ns.kg_root,
        output_root=ns.output_root,
        node_types=tuple(ns.node_types),
        relations=tuple(ns.relations),
        max_nodes_per_type=ns.max_nodes_per_type,
        max_edges_per_relation=ns.max_edges_per_relation,
        include_reverse_edges=not ns.no_reverse_edges,
        feature_tables=tuple(ns.feature_tables),
        strict=not ns.no_strict,
        build_name=ns.build_name,
        sort_node_ids=ns.sort_node_ids,
        embedding_features_root=ns.embedding_features_root,
        learned_fallback_config_path=ns.learned_fallback_config_path,
        fallback_seed=ns.fallback_seed,
        include_provenance_node_types=ns.include_provenance_node_types,
        plan_only=ns.plan_only,
        remote_output_root=ns.remote_output_root,
        remote_runner=ns.remote_runner,
        artifact_mode=ns.artifact_mode,
    )


def main(argv: list[str] | None = None) -> int:
    config = parse_args(argv)
    result = build_pyg_export(config)
    print(json.dumps({"output_root": result.output_root, "manifest": result.manifest_path, "validation_report": result.validation_report_path, "node_counts": result.node_counts, "edge_counts": result.edge_counts}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
