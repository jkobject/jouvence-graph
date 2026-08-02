from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

try:  # Optional: loaded only for real text model runs.
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - exercised in base env without optional dep
    SentenceTransformer = None  # type: ignore[assignment]

try:  # PyTorch is the local learned edge/fallback encoder runtime.
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]

TASK_ID = os.environ.get("HERMES_KANBAN_TASK", "t_8892763b")
RUN_ID = f"foundation_embedding_scaffold_20260624_{TASK_ID}"
POLICY_VERSION = "foundation_embedding_policy_v1+edge_embedding_policy_v1"
TEXT_MODEL_NAME = "pritamdeka/S-BioBERT-snli-multinli-stsb"
TEXT_MODEL_VERSION = "pritamdeka/S-BioBERT-snli-multinli-stsb@huggingface-main+policy_v1"
TEXT_MODEL_DIM = 768
EDGE_ENCODER_NAME = "local_pytorch_relation_value_evidence_mlp"
EDGE_ENCODER_VERSION = "edge_policy_v1+torch_seed_20260624+untrained_projection_smoke"
EDGE_EMBEDDING_DIM = 256
CREATED_AT = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
TEXTUAL_TABLES = [
    "cell_line_textual_summary.parquet",
    "cell_type_textual_summary.parquet",
    "disease_textual_summary.parquet",
    "gene_textual_summary.parquet",
    "molecule_textual_summary.parquet",
    "pathway_textual_summary.parquet",
    "phenotype_textual_summary.parquet",
    "protein_textual_summary.parquet",
    "tissue_textual_summary.parquet",
]
TEXT_FIELDS = ["node_type", "node_id", "summary_kind", "summary_text", "source", "source_dataset", "source_record_id", "release"]
EDGE_HEADER_FIELDS = ["relation", "display_relation", "x_type", "x_id", "y_type", "y_id", "source", "credibility"]
EVIDENCE_FIELDS = [
    "evidence_type",
    "source",
    "source_dataset",
    "source_record_id",
    "predicate",
    "direction",
    "evidence_score",
    "score",
    "effect_size",
    "p_value",
    "paper_id",
    "dataset_id",
    "study_id",
    "text_span",
    "section",
    "extraction_method",
    "release",
    "source_release",
    "experimental_system",
    "experimental_system_type",
    "throughput",
]


def stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def table_metadata_hash(path: Path) -> str:
    pf = pq.ParquetFile(path)
    meta = {
        "path_name": path.name,
        "rows": pf.metadata.num_rows,
        "row_groups": pf.metadata.num_row_groups,
        "schema": pf.schema.names,
        "serialized_size": getattr(pf.metadata, "serialized_size", None),
    }
    try:
        stat = path.stat()
        meta["stat_size"] = stat.st_size
        meta["stat_mtime_ns"] = stat.st_mtime_ns
    except OSError:
        pass
    return sha256_text(stable_json(meta))


def clean_value(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return f"{value:.8g}"
    return str(value).strip()


def read_head(path: Path, limit: int, columns: list[str] | None = None) -> pd.DataFrame:
    if limit <= 0:
        return pd.DataFrame()
    pf = pq.ParquetFile(path)
    batches = []
    remaining = limit
    for batch in pf.iter_batches(batch_size=min(max(limit, 1), 1024), columns=columns):
        table = pa.Table.from_batches([batch])
        batches.append(table.slice(0, remaining))
        remaining -= table.num_rows
        if remaining <= 0:
            break
    if not batches:
        return pd.DataFrame()
    return pa.concat_tables(batches).to_pandas()


class DeterministicTestEncoder:
    def __init__(self, dim: int = TEXT_MODEL_DIM) -> None:
        self.dim = dim

    def encode(self, texts: list[str], batch_size: int = 32, normalize_embeddings: bool = True, show_progress_bar: bool = False) -> np.ndarray:
        rows = []
        for text in texts:
            seed = int(sha256_text(text)[:16], 16) % (2**32)
            rng = np.random.default_rng(seed)
            vec = rng.normal(size=self.dim).astype(np.float32)
            if normalize_embeddings:
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
            rows.append(vec)
        return np.vstack(rows).astype(np.float32) if rows else np.zeros((0, self.dim), dtype=np.float32)


@dataclass
class TextEncoder:
    model_name: str
    test_deterministic: bool = False

    def __post_init__(self) -> None:
        if self.test_deterministic:
            self.model = DeterministicTestEncoder()
            self.embedding_dim = TEXT_MODEL_DIM
            self.model_version = "deterministic_test_encoder@unit-test-only"
            return
        if SentenceTransformer is None:
            raise RuntimeError(
                "sentence-transformers is not installed. Run with: "
                "uv run --with sentence-transformers python -m manage_db.build_real_embeddings ..."
            )
        self.model = SentenceTransformer(self.model_name)
        dim = int(self.model.get_sentence_embedding_dimension() or TEXT_MODEL_DIM)
        self.embedding_dim = dim
        self.model_version = TEXT_MODEL_VERSION

    def encode(self, texts: list[str], batch_size: int) -> np.ndarray:
        vectors = self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vectors, dtype=np.float32)


def text_payload(row: pd.Series) -> str:
    parts = []
    for field in TEXT_FIELDS:
        value = clean_value(row.get(field))
        if value:
            parts.append(f"{field}: {value}")
    return "\n".join(parts)


def write_parquet(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path, compression="zstd")


def gcs_join(root: str, *parts: str) -> str:
    return "/".join([root.rstrip("/"), *(part.strip("/") for part in parts)])


def gcloud_object_exists(uri: str) -> bool:
    result = subprocess.run(["gcloud", "storage", "ls", uri], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return result.returncode == 0


def copy_gcs_object_if_exists(uri: str, destination: Path, required: bool) -> bool:
    if not gcloud_object_exists(uri):
        if required:
            raise FileNotFoundError(f"Required GCS input not found: {uri}")
        # A reusable cache must represent the current GCS snapshot. Keeping an
        # older optional file here would silently feed stale features/evidence
        # into the embedding run after the source object disappears.
        destination.unlink(missing_ok=True)
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["gcloud", "storage", "cp", "--quiet", uri, str(destination)], check=True)
    return True


def prepare_local_cache_from_gcs(gcs_kg_root: str, local_cache_dir: Path, edge_relations: list[str]) -> dict[str, Any]:
    copied: list[str] = []
    skipped_optional: list[str] = []
    for table_name in TEXTUAL_TABLES:
        uri = gcs_join(gcs_kg_root, "features", table_name)
        destination = local_cache_dir / "features" / table_name
        if copy_gcs_object_if_exists(uri, destination, required=False):
            copied.append(str(destination))
        else:
            skipped_optional.append(uri)
    for relation in edge_relations:
        for subdir, required in [("edges", True), ("evidence", False)]:
            uri = gcs_join(gcs_kg_root, subdir, f"{relation}.parquet")
            destination = local_cache_dir / subdir / f"{relation}.parquet"
            if copy_gcs_object_if_exists(uri, destination, required=required):
                copied.append(str(destination))
            else:
                skipped_optional.append(uri)
    return {
        "source_gcs_root": gcs_kg_root.rstrip("/"),
        "local_cache_dir": str(local_cache_dir),
        "copied_files": copied,
        "skipped_optional_inputs": skipped_optional,
        "command_family": "gcloud storage cp local cache prep; avoids brittle macOS FUSE Path.exists() calls",
    }


def build_text_embeddings(kg_root: Path, output_dir: Path, encoder: TextEncoder, limit_per_table: int, batch_size: int) -> dict[str, Any]:
    outputs: dict[str, Any] = {}
    for table_name in TEXTUAL_TABLES:
        input_path = kg_root / "features" / table_name
        if not input_path.exists():
            outputs[table_name] = {"status": "missing_input", "input_uri": str(input_path)}
            continue
        df = read_head(input_path, limit_per_table)
        rows: list[dict[str, Any]] = []
        payloads = [text_payload(row) for _, row in df.iterrows()]
        vectors = encoder.encode(payloads, batch_size=batch_size)
        for (_, row), payload, vector in zip(df.iterrows(), payloads, vectors, strict=True):
            payload_hash = sha256_text(payload)
            node_type = clean_value(row.get("node_type")) or table_name.replace("_textual_summary.parquet", "")
            node_id = clean_value(row.get("node_id"))
            feature_key = clean_value(row.get("feature_key")) or node_id
            rows.append(
                {
                    "embedding_key": f"{node_id}|{node_type}|{encoder.model_name}|{encoder.model_version}|{feature_key}|{payload_hash}|mean_pool_l2",
                    "node_id": node_id,
                    "node_type": node_type,
                    "source_feature_table": table_name,
                    "source_feature_key": feature_key,
                    "source_feature_hash": payload_hash,
                    "modality": "text",
                    "embedding_model": encoder.model_name,
                    "embedding_version": encoder.model_version,
                    "embedding_dim": int(vector.shape[0]),
                    "embedding_dtype": "float32",
                    "embedding_format": "list_float32",
                    "embedding": vector.astype(np.float32).tolist(),
                    "pooling": "sentence_transformer_model_default_mean_pooling",
                    "normalization": "l2",
                    "preprocessing": stable_json({"serializer": "official_textual_summary_payload_v1", "fields": TEXT_FIELDS}),
                    "input_length": len(payload),
                    "window_count": 1,
                    "created_at": CREATED_AT,
                    "source_feature_release": clean_value(row.get("release")) or clean_value(row.get("source_release")),
                    "provenance": stable_json({"task_id": TASK_ID, "run_id": RUN_ID, "kg_root": str(kg_root), "input_path": str(input_path), "payload_preview": payload[:512]}),
                    "license": clean_value(row.get("license")) or "inherits source feature table license",
                    "citation": clean_value(row.get("citation")) or "source feature table citation; S-BioBERT model card",
                }
            )
        node_type_slug = table_name.replace("_textual_summary.parquet", "")
        out_path = output_dir / "features" / "embeddings" / "text" / node_type_slug / "sbiobert_snli_multinli_stsb" / "policy_v1" / "part-000.parquet"
        write_parquet(rows, out_path)
        outputs[table_name] = {
            "status": "embedded",
            "input_uri": str(input_path),
            "input_rows_total": pq.ParquetFile(input_path).metadata.num_rows,
            "embedded_rows": len(rows),
            "embedding_dim": encoder.embedding_dim,
            "output_path": str(out_path),
            "source_table_metadata_hash": table_metadata_hash(input_path),
        }
    return outputs


def edge_key_from_row(row: pd.Series) -> str:
    return f"{clean_value(row.get('relation'))}|{clean_value(row.get('x_id'))}|{clean_value(row.get('y_id'))}"


def edge_header(row: pd.Series) -> str:
    parts = [f"edge_key: {edge_key_from_row(row)}"]
    for field in EDGE_HEADER_FIELDS:
        value = clean_value(row.get(field))
        if value:
            parts.append(f"{field}: {value}")
    extra = []
    for field in sorted(set(row.index) - set(EDGE_HEADER_FIELDS)):
        value = clean_value(row.get(field))
        if value and field not in {"created_at"}:
            extra.append((field, value))
    if extra:
        parts.append("edge_metadata: " + stable_json(dict(extra)))
    return "\n".join(parts)


def evidence_block(row: pd.Series) -> str:
    parts = []
    for field in EVIDENCE_FIELDS:
        if field in row.index:
            value = clean_value(row.get(field))
            if value:
                parts.append(f"{field}: {value}")
    return "\n".join(parts)


def numeric_features(edge: pd.Series, evidence: pd.Series | None) -> np.ndarray:
    vals = []
    for field in ["credibility", "score", "e2g_score"]:
        try:
            vals.append(float(edge.get(field, 0) or 0))
        except Exception:
            vals.append(0.0)
    if evidence is None:
        vals.extend([0.0, 0.0, 1.0])
    else:
        for field in ["evidence_score", "score", "effect_size"]:
            try:
                v = float(evidence.get(field, 0) or 0)
                vals.append(v if math.isfinite(v) else 0.0)
            except Exception:
                vals.append(0.0)
    return np.asarray(vals, dtype=np.float32)


def edge_weight(edge: pd.Series, evidence: pd.Series | None) -> float:
    try:
        credibility = int(edge.get("credibility", 1) or 1)
    except Exception:
        credibility = 1
    cred = {1: 1.0, 2: 1.25, 3: 1.5}.get(credibility, 1.0)
    if evidence is None:
        return cred
    for field in ["evidence_score", "score"]:
        try:
            v = float(evidence.get(field, 1.0) or 1.0)
            if math.isfinite(v) and v > 0:
                return cred * min(v, 1.0)
        except Exception:
            continue
    return cred


class EdgeMLP:
    def __init__(self, text_dim: int, numeric_dim: int = 6, out_dim: int = EDGE_EMBEDDING_DIM) -> None:
        if torch is None or nn is None:
            raise RuntimeError("torch is required for the edge/value/evidence MLP encoder path")
        torch.manual_seed(20260623)
        self.model = nn.Sequential(
            nn.Linear(text_dim + numeric_dim, 384),
            nn.ReLU(),
            nn.Linear(384, out_dim),
        )
        self.model.eval()

    def encode(self, text_vectors: np.ndarray, nums: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x = np.concatenate([text_vectors.astype(np.float32), nums.astype(np.float32)], axis=1)
            out = self.model(torch.from_numpy(x)).numpy().astype(np.float32)
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return np.divide(out, norms, out=np.zeros_like(out), where=norms > 0)


def build_edge_embeddings(kg_root: Path, output_dir: Path, encoder: TextEncoder, relations: list[str], edge_limit: int, batch_size: int) -> dict[str, Any]:
    outputs: dict[str, Any] = {}
    mlp = EdgeMLP(text_dim=encoder.embedding_dim)
    for relation in relations:
        edge_path = kg_root / "edges" / f"{relation}.parquet"
        evidence_path = kg_root / "evidence" / f"{relation}.parquet"
        if not edge_path.exists():
            outputs[relation] = {"status": "missing_edge_input", "edge_uri": str(edge_path)}
            continue
        edges = read_head(edge_path, edge_limit)
        if evidence_path.exists():
            evidence_rows_total = pq.ParquetFile(evidence_path).metadata.num_rows
            if evidence_rows_total <= 1_000_000:
                # For bounded smoke relations, read the full evidence table so the first N
                # edge rows can join their actual supporting evidence instead of falling
                # back to header-only just because matching evidence appears later.
                evidence = pq.read_table(evidence_path).to_pandas()
            else:
                evidence = read_head(evidence_path, max(edge_limit * 20, edge_limit))
        else:
            evidence = pd.DataFrame()
        if not evidence.empty and "edge_key" not in evidence.columns:
            evidence["edge_key"] = evidence.apply(edge_key_from_row, axis=1)
        rows: list[dict[str, Any]] = []
        for _, edge in edges.iterrows():
            key = edge_key_from_row(edge)
            ev_rows = evidence[evidence["edge_key"] == key].copy() if not evidence.empty else pd.DataFrame()
            sort_cols = [c for c in ["source", "source_dataset", "source_record_id", "paper_id", "study_id"] if c in ev_rows.columns]
            if sort_cols:
                ev_rows = ev_rows.sort_values(sort_cols, kind="stable")
            header = edge_header(edge)
            payloads: list[str] = []
            nums: list[np.ndarray] = []
            weights: list[float] = []
            if ev_rows.empty:
                payloads = [header]
                nums = [numeric_features(edge, None)]
                weights = [edge_weight(edge, None)]
                aggregation = "relation_value_evidence_mlp_header_only"
            else:
                for _, ev in ev_rows.iterrows():
                    payloads.append(header + "\n--- evidence/value row ---\n" + evidence_block(ev))
                    nums.append(numeric_features(edge, ev))
                    weights.append(edge_weight(edge, ev))
                aggregation = "relation_value_evidence_mlp_weighted_mean"
            text_vectors = encoder.encode(payloads, batch_size=batch_size)
            per_row = mlp.encode(text_vectors, np.vstack(nums))
            w = np.asarray(weights, dtype=np.float32)
            if float(w.sum()) <= 0:
                w = np.ones_like(w)
            w = w / w.sum()
            vector = np.average(per_row, axis=0, weights=w).astype(np.float32)
            norm = float(np.linalg.norm(vector))
            if norm > 0:
                vector = vector / norm
            payload_hash = sha256_text(stable_json(payloads))
            evidence_hash = sha256_text(stable_json([p.split("--- evidence/value row ---", 1)[-1] for p in payloads[1:]]))
            edge_hash = sha256_text(header)
            evidence_sources = {}
            if not ev_rows.empty:
                for _, ev in ev_rows.iterrows():
                    source = clean_value(ev.get("source")) or "unknown"
                    dataset = clean_value(ev.get("source_dataset")) or clean_value(ev.get("source_release")) or "unknown"
                    evidence_sources[f"{source}|{dataset}"] = evidence_sources.get(f"{source}|{dataset}", 0) + 1
            rows.append(
                {
                    "edge_embedding_key": f"{key}|{EDGE_ENCODER_NAME}|{EDGE_ENCODER_VERSION}|{payload_hash}|{aggregation}",
                    "edge_key": key,
                    "x_id": clean_value(edge.get("x_id")),
                    "x_type": clean_value(edge.get("x_type")),
                    "y_id": clean_value(edge.get("y_id")),
                    "y_type": clean_value(edge.get("y_type")),
                    "relation": relation,
                    "embedding_model": EDGE_ENCODER_NAME,
                    "embedding_version": EDGE_ENCODER_VERSION,
                    "embedding_dim": EDGE_EMBEDDING_DIM,
                    "embedding_dtype": "float32",
                    "embedding_format": "list_float32",
                    "embedding": vector.astype(np.float32).tolist(),
                    "pooling": aggregation,
                    "normalization": "l2",
                    "payload_hash": payload_hash,
                    "evidence_hash": evidence_hash,
                    "edge_hash": edge_hash,
                    "node_context_hash": "",
                    "n_evidence_rows": int(len(ev_rows)),
                    "n_evidence_rows_encoded": int(len(ev_rows)),
                    "evidence_sources": stable_json(evidence_sources),
                    "source_edge_uri": str(edge_path),
                    "source_evidence_uri": str(evidence_path) if evidence_path.exists() else "",
                    "source_node_context_uris": "[]",
                    "preprocessing": stable_json({"serializer": "edge_payload_serializer_v1", "text_encoder": encoder.model_name, "numeric_fields": ["credibility", "score", "e2g_score", "evidence_score", "effect_size"], "mlp": EDGE_ENCODER_VERSION}),
                    "created_at": CREATED_AT,
                    "builder_code_version": TASK_ID,
                    "license": "inherits edge/evidence input licenses; S-BioBERT model card; local untrained MLP projection",
                    "citation": "edge evidence embedding policy; S-BioBERT model card; PyTorch local MLP projection",
                }
            )
        out_path = output_dir / "features" / "edge_embeddings" / "by_relation" / relation / "relation_value_evidence_mlp" / "edge_policy_v1" / "part-000.parquet"
        write_parquet(rows, out_path)
        outputs[relation] = {
            "status": "embedded",
            "edge_uri": str(edge_path),
            "evidence_uri": str(evidence_path) if evidence_path.exists() else "",
            "edge_rows_total": pq.ParquetFile(edge_path).metadata.num_rows,
            "evidence_rows_total": pq.ParquetFile(evidence_path).metadata.num_rows if evidence_path.exists() else 0,
            "embedded_edges": len(rows),
            "embedding_dim": EDGE_EMBEDDING_DIM,
            "output_path": str(out_path),
            "source_table_metadata_hashes": {"edge": table_metadata_hash(edge_path), "evidence": table_metadata_hash(evidence_path) if evidence_path.exists() else ""},
        }
    return outputs


def write_fallback_artifacts(kg_root: Path, output_dir: Path) -> dict[str, Any]:
    node_types = []
    nodes_dir = kg_root / "nodes"
    if nodes_dir.exists():
        node_types = sorted(p.stem for p in nodes_dir.glob("*.parquet"))
    edge_relations = []
    edges_dir = kg_root / "edges"
    if edges_dir.exists():
        edge_relations = sorted(p.stem for p in edges_dir.glob("*.parquet"))
    artifact = {
        "task_id": TASK_ID,
        "run_id": RUN_ID,
        "policy": "model-side learned fallback only; not a fabricated source-derived canonical embedding",
        "node_fallback": {"module": "torch.nn.Embedding", "key_space": "node_type_unknown_bucket plus optional per-node-id table in downstream training", "dim": 256, "node_types": node_types},
        "edge_fallback": {"module": "torch.nn.Embedding", "key_space": "relation|x_type|y_type unknown/no-payload bucket", "dim": EDGE_EMBEDDING_DIM, "relations": edge_relations},
        "initialization": "normal_(mean=0,std=0.02) in downstream training; regularized and saved with model checkpoint, not KG feature layer",
        "created_at": CREATED_AT,
    }
    path = output_dir / "features" / "embeddings" / "reports" / "learned_fallback_config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    return {"path": str(path), "node_type_count": len(node_types), "edge_relation_count": len(edge_relations)}


def validate_outputs(manifest: dict[str, Any]) -> dict[str, Any]:
    checks: dict[str, Any] = {"parquet_files_checked": 0, "duplicate_keys": {}, "bad_dims": {}, "non_finite_vectors": {}, "all_zero_vectors": {}}
    paths: list[tuple[str, str, str]] = []
    for table, info in manifest["outputs"]["node_text_embeddings"].items():
        if info.get("status") == "embedded":
            paths.append((f"node:{table}", info["output_path"], "embedding_key"))
    for rel, info in manifest["outputs"]["edge_evidence_embeddings"].items():
        if info.get("status") == "embedded":
            paths.append((f"edge:{rel}", info["output_path"], "edge_embedding_key"))
    for label, path, key_col in paths:
        df = pd.read_parquet(path)
        checks["parquet_files_checked"] += 1
        checks["duplicate_keys"][label] = int(df[key_col].duplicated().sum())
        bad_dim = 0
        non_finite = 0
        all_zero = 0
        for _, row in df.iterrows():
            vec = np.asarray(row["embedding"], dtype=np.float32)
            if len(vec) != int(row["embedding_dim"]):
                bad_dim += 1
            if not np.isfinite(vec).all():
                non_finite += 1
            if float(np.linalg.norm(vec)) == 0.0:
                all_zero += 1
        checks["bad_dims"][label] = bad_dim
        checks["non_finite_vectors"][label] = non_finite
        checks["all_zero_vectors"][label] = all_zero
    checks["passed"] = all(
        all(v == 0 for v in checks[name].values())
        for name in ["duplicate_keys", "bad_dims", "non_finite_vectors", "all_zero_vectors"]
    )
    return checks


def write_summary(output_dir: Path, manifest: dict[str, Any]) -> str:
    path = output_dir / "real_embedding_summary.md"
    lines = [
        "# Staged real embedding build summary",
        "",
        f"Task: `{TASK_ID}`",
        f"Run id: `{RUN_ID}`",
        f"Created at: `{CREATED_AT}`",
        "",
        "## Gate",
        "",
        "Staged-only derived features. No canonical KG files were promoted or overwritten.",
        "This run uses real local model inference for official textual summaries: S-BioBERT via sentence-transformers, not HashingVectorizer.",
        "Edge embeddings use a local PyTorch relation/value/evidence MLP path over S-BioBERT evidence payload vectors plus numeric evidence/edge fields, then weighted pooling to one vector per edge.",
    ]
    if manifest.get("input_cache"):
        cache = manifest["input_cache"]
        lines.extend(
            [
                "Input cache: canonical GCS objects were copied to a local cache with `gcloud storage cp` before reading Parquet, so this run does not depend on the macOS FUSE mount being healthy.",
                f"- source GCS root: `{cache['source_gcs_root']}`",
                f"- local cache dir: `{cache['local_cache_dir']}`",
                f"- copied files: `{len(cache['copied_files'])}`",
            ]
        )
    lines.extend([
        "",
        "## Outputs",
    ])
    for table, info in manifest["outputs"]["node_text_embeddings"].items():
        if info.get("status") == "embedded":
            lines.append(f"- text `{table}`: {info['embedded_rows']} rows, dim {info['embedding_dim']} -> `{info['output_path']}`")
    for rel, info in manifest["outputs"]["edge_evidence_embeddings"].items():
        if info.get("status") == "embedded":
            lines.append(f"- edge `{rel}`: {info['embedded_edges']} rows, dim {info['embedding_dim']} -> `{info['output_path']}`")
    lines.extend([
        f"- learned fallback config: `{manifest['outputs']['learned_fallback_config']['path']}`",
        "",
        "## Blocked heavier modalities",
        "",
    ])
    for item in manifest["blocked_modalities"]:
        lines.append(f"- `{item['modality']}`: {item['reason']}")
    lines.extend([
        "",
        "## Validation",
        "",
        f"Validation passed: `{manifest['validation']['passed']}`; files checked: `{manifest['validation']['parquet_files_checked']}`.",
        "",
        "## Recompute command",
        "",
        "```bash",
        manifest["recompute_command"],
        "```",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)


def run(
    kg_root: Path,
    output_dir: Path,
    text_limit_per_table: int = 16,
    edge_relations: list[str] | None = None,
    edge_limit_per_relation: int = 16,
    batch_size: int = 16,
    clean: bool = False,
    test_deterministic_encoder: bool = False,
    gcs_kg_root: str | None = None,
    local_cache_dir: Path | None = None,
) -> dict[str, Any]:
    if clean and output_dir.exists():
        import shutil

        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    if edge_relations is None:
        edge_relations = ["molecule_targets_gene", "tissue_expresses_protein"]
    input_cache: dict[str, Any] | None = None
    if gcs_kg_root:
        if local_cache_dir is None:
            local_cache_dir = output_dir.parent / f"{output_dir.name}_kg_cache"
        input_cache = prepare_local_cache_from_gcs(gcs_kg_root, local_cache_dir, edge_relations)
        kg_root = local_cache_dir
    encoder = TextEncoder(TEXT_MODEL_NAME, test_deterministic=test_deterministic_encoder)
    edge_outputs = build_edge_embeddings(kg_root, output_dir, encoder, edge_relations, edge_limit_per_relation, batch_size) if edge_relations else {}
    outputs = {
        "node_text_embeddings": build_text_embeddings(kg_root, output_dir, encoder, text_limit_per_table, batch_size),
        "edge_evidence_embeddings": edge_outputs,
        "learned_fallback_config": write_fallback_artifacts(kg_root, output_dir),
    }
    blocked_modalities = [
        {"modality": "protein_sequence_esm2", "input": str(kg_root / "features" / "protein_sequence.parquet"), "reason": "official source feature is available; run in envs/embeddings/protein_esm2 with GPU and explicit long-sequence windowing"},
        {"modality": "transcript_cdna_nucleotide_transformer", "input": str(kg_root / "features" / "transcript_sequence.parquet"), "reason": "official source feature is available; run in envs/embeddings/nucleotide_transformer after length/window audit"},
        {"modality": "molecule_smiles_chemberta", "input": str(kg_root / "features" / "molecule_fingerprint.parquet"), "reason": "official Morgan fingerprint/SMILES-derived source feature is available; run learned SMILES encoder in envs/embeddings/molecule_smiles"},
        {"modality": "gene_enhancer_mutation_dna", "input": "gene/enhancer/mutation genomic sequence features", "reason": "source feature tables remain missing/deferred by policy; use model-side learned fallback until reviewed sequence/coordinate features exist"},
    ]
    manifest: dict[str, Any] = {
        "task_id": TASK_ID,
        "run_id": RUN_ID,
        "created_at": CREATED_AT,
        "staged_only": True,
        "canonical_promotion": False,
        "kg_root": str(kg_root),
        "input_cache": input_cache,
        "output_dir": str(output_dir),
        "policy_version": POLICY_VERSION,
        "models": {
            "text": {"embedding_model": encoder.model_name, "embedding_version": encoder.model_version, "embedding_dim": encoder.embedding_dim, "normalization": "l2"},
            "edge": {"embedding_model": EDGE_ENCODER_NAME, "embedding_version": EDGE_ENCODER_VERSION, "embedding_dim": EDGE_EMBEDDING_DIM, "normalization": "l2"},
        },
        "outputs": outputs,
        "blocked_modalities": blocked_modalities,
        "runtime_seconds": time.perf_counter() - start,
        "environment": {"python": platform.python_version(), "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__, "pyarrow": pa.__version__, "torch_available": torch is not None, "sentence_transformers_available": SentenceTransformer is not None or test_deterministic_encoder},
        "recompute_command": " ".join([
            "uv run --with sentence-transformers python -m manage_db.build_real_embeddings",
            *( [f"--gcs-kg-root {input_cache['source_gcs_root']}", f"--local-cache-dir {input_cache['local_cache_dir']}"] if input_cache else [f"--kg-root {kg_root}"] ),
            f"--output-dir {output_dir}",
            f"--text-limit-per-table {text_limit_per_table}",
            f"--edge-limit-per-relation {edge_limit_per_relation}",
            *( ["--edge-relations", *edge_relations] if edge_relations else ["--skip-edge-embeddings"] ),
            "--clean",
        ]),
    }
    manifest["validation"] = validate_outputs(manifest)
    manifest["summary_path"] = write_summary(output_dir, manifest)
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build staged real node/edge embeddings from official Jouvence KG features.")
    parser.add_argument("--kg-root", type=Path, default=Path("/Users/jkobject/mnt/gcs/jouvencekb/main"))
    parser.add_argument("--gcs-kg-root", default=None, help="Canonical GCS KG root, e.g. gs://jouvencekb/main. When set, copy required inputs to --local-cache-dir first and read from that local cache instead of FUSE.")
    parser.add_argument("--local-cache-dir", type=Path, default=None, help="Local cache destination for --gcs-kg-root inputs; defaults next to --output-dir.")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/staged/t_8892763b/text_sbiobert_smoke"))
    parser.add_argument("--text-limit-per-table", type=int, default=16)
    parser.add_argument("--edge-relations", nargs="+", default=["molecule_targets_gene", "tissue_expresses_protein"])
    parser.add_argument("--skip-edge-embeddings", action="store_true", help="Run only node textual embeddings; useful for a first text-model smoke when edge inputs are not staged locally.")
    parser.add_argument("--edge-limit-per-relation", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--test-deterministic-encoder", action="store_true", help="Use only in unit tests; not a real embedding model.")
    args = parser.parse_args(argv)
    manifest = run(
        kg_root=args.kg_root,
        output_dir=args.output_dir,
        text_limit_per_table=args.text_limit_per_table,
        edge_relations=[] if args.skip_edge_embeddings else args.edge_relations,
        edge_limit_per_relation=args.edge_limit_per_relation,
        batch_size=args.batch_size,
        clean=args.clean,
        test_deterministic_encoder=args.test_deterministic_encoder,
        gcs_kg_root=args.gcs_kg_root,
        local_cache_dir=args.local_cache_dir,
    )
    print(json.dumps({"manifest_path": str(Path(manifest["output_dir"]) / "manifest.json"), "summary_path": manifest["summary_path"], "validation": manifest["validation"], "runtime_seconds": manifest["runtime_seconds"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
