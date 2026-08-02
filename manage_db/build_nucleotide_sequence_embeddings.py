from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from manage_db import resumable_embedding_parts as parts

try:  # Optional real nucleotide-transformer runtime; installed by embeddings-nucleotide.
    import torch
    from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer
except Exception:  # pragma: no cover - base/dev envs may omit model deps
    torch = None  # type: ignore[assignment]
    AutoConfig = None  # type: ignore[assignment]
    AutoModelForMaskedLM = None  # type: ignore[assignment]
    AutoTokenizer = None  # type: ignore[assignment]

TASK_ID = os.environ.get("HERMES_KANBAN_TASK", "t_afe9098e")
RUN_ID = f"transcript_nucleotide_transformer_smoke_20260624_{TASK_ID}"
CREATED_AT = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
MODEL_NAME = "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species"
MODEL_REVISION = "81b29e5786726d891dbf929404ef20adca5b36f1"
DNABERT2_FALLBACK_MODEL = "zhihan1996/DNABERT-2-117M"
DNABERT2_FALLBACK_REVISION = "7bce263b15377fc15361f52cfab88f8b586abda0"
MODEL_FAMILY = "nucleotide_transformer"
POLICY_VERSION = "foundation_embedding_policy_v1+transcript_nucleotide_transformer_window_mean_v1"
LONG_SEQUENCE_POLICY = (
    "Transcript cDNA rows are embedded independently from genes. Sequences longer than "
    "max_nucleotides_per_window are split into deterministic non-overlapping or strided nucleotide windows; "
    "each window is embedded with the pinned Nucleotide Transformer checkpoint using attention-masked mean pooling "
    "over last hidden states, then transcript vectors are L2-normalized means of window vectors. Tokenizer truncation "
    "is disabled. Missing/empty sequences, non-dna_iupac rows, invalid alphabet characters, or rows exceeding "
    "--max-windows-per-sequence are written to skipped rows; no placeholder vectors and no transcript-to-gene averaging."
)
VALID_DNA_IUPAC = set("ACGTRYSWKMBDHVN")
SKIPPED_ROW_COLUMNS = [
    "row_index",
    "node_id",
    "feature_key",
    "sequence_length",
    "skip_reason",
    "long_sequence_policy",
    "created_at",
]
LENGTH_QUANTILES = [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]


def stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


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


def gcs_join(root: str, *parts: str) -> str:
    return "/".join([root.rstrip("/"), *(part.strip("/") for part in parts)])


def copy_gcs_input(gcs_kg_root: str, local_cache_dir: Path) -> dict[str, Any]:
    uri = gcs_join(gcs_kg_root, "features", "transcript_sequence.parquet")
    destination = local_cache_dir / "features" / "transcript_sequence.parquet"
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["gcloud", "storage", "cp", "--quiet", uri, str(destination)], check=True)
    return {
        "source_gcs_root": gcs_kg_root.rstrip("/"),
        "source_uri": uri,
        "local_cache_dir": str(local_cache_dir),
        "copied_files": [str(destination)],
        "command_family": "gcloud storage cp local cache prep; avoids brittle macOS FUSE Path.exists() calls",
    }


def source_path(kg_root: Path) -> Path:
    path = kg_root / "features" / "transcript_sequence.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Expected official source table at {path}")
    return path


def parquet_metadata(path: Path) -> dict[str, Any]:
    pf = pq.ParquetFile(path)
    return {
        "path": str(path),
        "rows": pf.metadata.num_rows,
        "row_groups": pf.metadata.num_row_groups,
        "columns": pq.read_schema(path).names,
        "sha256": sha256_file(path),
    }


def read_head(path: Path, limit: int | None = None) -> pd.DataFrame:
    if limit is not None and limit <= 0:
        return pd.DataFrame()
    if limit is None:
        return pq.read_table(path).to_pandas()
    batches: list[pa.Table] = []
    remaining = limit
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=min(max(limit, 1), 1024)):
        table = pa.Table.from_batches([batch])
        batches.append(table.slice(0, remaining))
        remaining -= table.num_rows
        if remaining <= 0:
            break
    if not batches:
        return pd.DataFrame()
    return pa.concat_tables(batches).to_pandas()


def write_parquet(rows: list[dict[str, Any]], path: Path, columns: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        table = pa.Table.from_pylist(rows)
    else:
        table = pa.Table.from_pandas(pd.DataFrame(columns=columns or []), preserve_index=False)
    pq.write_table(table, path, compression="zstd")


def sequence_windows(sequence: str, max_nucleotides_per_window: int, window_stride: int) -> list[tuple[int, int, str]]:
    sequence = sequence.strip().upper().replace("U", "T")
    if not sequence:
        return []
    if max_nucleotides_per_window <= 0:
        raise ValueError("max_nucleotides_per_window must be positive")
    if window_stride <= 0:
        raise ValueError("window_stride must be positive")
    windows: list[tuple[int, int, str]] = []
    start = 0
    while start < len(sequence):
        end = min(start + max_nucleotides_per_window, len(sequence))
        windows.append((start, end, sequence[start:end]))
        if end >= len(sequence):
            break
        start += window_stride
    return windows


def invalid_dna_iupac_reason(sequence: str) -> str | None:
    invalid = sorted(set(sequence.upper()) - VALID_DNA_IUPAC)
    if invalid:
        return "invalid_dna_iupac_characters_" + "".join(invalid[:10])
    return None


def patch_transformers_for_nucleotide_transformer_remote_code() -> None:
    """Restore older Transformers helpers used by the pinned NT v2 remote code."""
    if torch is None:
        return
    try:
        import transformers.modeling_utils as modeling_utils
        import transformers.pytorch_utils as pytorch_utils
    except Exception:
        return
    if not hasattr(pytorch_utils, "find_pruneable_heads_and_indices"):
        def find_pruneable_heads_and_indices(heads: set[int], n_heads: int, head_size: int, already_pruned_heads: set[int]) -> tuple[set[int], Any]:
            mask = torch.ones(n_heads, head_size)
            heads = set(heads) - already_pruned_heads
            for head in heads:
                head = head - sum(1 if pruned_head < head else 0 for pruned_head in already_pruned_heads)
                mask[head] = 0
            mask = mask.view(-1).contiguous().eq(1)
            index = torch.arange(len(mask))[mask].long()
            return heads, index
        pytorch_utils.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices
    if not hasattr(modeling_utils.PreTrainedModel, "all_tied_weights_keys"):
        modeling_utils.PreTrainedModel.all_tied_weights_keys = {}


def patch_remote_esm_base_model(base_model: Any) -> None:
    if not hasattr(base_model, "get_head_mask"):
        def get_head_mask(head_mask: Any, num_hidden_layers: int, is_attention_chunked: bool = False) -> Any:
            return [None] * num_hidden_layers if head_mask is None else head_mask
        base_model.get_head_mask = get_head_mask


def audit_sequence_lengths(path: Path, max_nucleotides_per_window: int, window_stride: int) -> dict[str, Any]:
    table = pq.read_table(path, columns=["length", "sequence", "alphabet"])
    df = table.to_pandas()
    lengths = pd.to_numeric(df.get("length"), errors="coerce")
    fallback_lengths = df["sequence"].fillna("").astype(str).str.len()
    lengths = lengths.fillna(fallback_lengths).astype(int)
    alphabets = df["alphabet"].fillna("").astype(str)
    missing_or_empty = int((lengths <= 0).sum())
    non_dna_iupac = int((alphabets != "dna_iupac").sum())
    over_window = int((lengths > max_nucleotides_per_window).sum())
    quantiles = {f"p{int(q * 100):02d}": int(lengths.quantile(q)) for q in LENGTH_QUANTILES} if len(lengths) else {}
    estimated_windows = lengths.apply(lambda v: len(sequence_windows("A" * int(max(v, 0)), max_nucleotides_per_window, window_stride))).astype(int)
    return {
        "source_rows": int(len(lengths)),
        "min_length": int(lengths.min()) if len(lengths) else 0,
        "max_length": int(lengths.max()) if len(lengths) else 0,
        "mean_length": float(lengths.mean()) if len(lengths) else 0.0,
        "quantiles": quantiles,
        "missing_or_empty_sequences": missing_or_empty,
        "non_dna_iupac_rows": non_dna_iupac,
        "sequences_longer_than_max_window": over_window,
        "max_nucleotides_per_window": int(max_nucleotides_per_window),
        "window_stride": int(window_stride),
        "estimated_total_windows": int(estimated_windows.sum()) if len(estimated_windows) else 0,
        "estimated_max_windows_per_sequence": int(estimated_windows.max()) if len(estimated_windows) else 0,
        "long_sequence_policy": LONG_SEQUENCE_POLICY,
        "pooling": "attention_masked_mean_pool_last_hidden_state_per_window_then_mean_of_window_vectors_l2",
        "transcript_gene_policy": "keep transcript embeddings separate; do not average or collapse to genes",
    }


@dataclass
class NucleotideTransformerEncoder:
    model_name: str = MODEL_NAME
    revision: str = MODEL_REVISION
    device: str = "auto"
    test_deterministic: bool = False

    def __post_init__(self) -> None:
        if self.test_deterministic:
            self.embedding_dim = 32
            self.resolved_revision = "deterministic-test-encoder"
            self.model_version = f"{self.model_name}@{self.resolved_revision}+unit-test-only"
            self.device_resolved = "cpu"
            return
        if AutoConfig is None or AutoTokenizer is None or AutoModelForMaskedLM is None or torch is None:
            raise RuntimeError(
                "transformers and torch are required for real Nucleotide Transformer embeddings. "
                "Run with: uv run --group embeddings-nucleotide python -m manage_db.build_nucleotide_sequence_embeddings ..."
            )
        patch_transformers_for_nucleotide_transformer_remote_code()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, revision=self.revision, trust_remote_code=True)
        config = AutoConfig.from_pretrained(self.model_name, revision=self.revision, trust_remote_code=True)
        for missing_name, default_value in {
            "is_decoder": False,
            "add_cross_attention": False,
            "chunk_size_feed_forward": 0,
        }.items():
            if not hasattr(config, missing_name):
                setattr(config, missing_name, default_value)
        masked_lm = AutoModelForMaskedLM.from_pretrained(self.model_name, revision=self.revision, trust_remote_code=True, config=config)
        self.model = getattr(masked_lm, "esm", None) or getattr(masked_lm, getattr(masked_lm, "base_model_prefix", ""))
        patch_remote_esm_base_model(self.model)
        self.model.eval()
        if self.device == "auto":
            self.device_resolved = "cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
        else:
            self.device_resolved = self.device
        self.model.to(self.device_resolved)
        self.embedding_dim = int(getattr(self.model.config, "hidden_size", 0) or getattr(self.model.config, "d_model", 0) or 0)
        self.resolved_revision = (
            getattr(self.model.config, "_commit_hash", None)
            or getattr(self.tokenizer, "init_kwargs", {}).get("_commit_hash")
            or self.revision
        )
        self.model_version = f"{self.model_name}@{self.resolved_revision}+attention_masked_mean_pool_window_mean_l2+policy_v1"

    def encode_windows(self, sequences: list[str], batch_size: int, tokenizer_max_length: int | None) -> np.ndarray:
        if not sequences:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)
        if self.test_deterministic:
            rows = []
            for sequence in sequences:
                seed = int(sha256_text(sequence)[:16], 16) % (2**32)
                rng = np.random.default_rng(seed)
                vec = rng.normal(size=self.embedding_dim).astype(np.float32)
                norm = np.linalg.norm(vec)
                rows.append(vec / norm if norm > 0 else vec)
            return np.vstack(rows).astype(np.float32)
        assert torch is not None
        vectors: list[np.ndarray] = []
        for start in range(0, len(sequences), batch_size):
            batch = sequences[start : start + batch_size]
            kwargs: dict[str, Any] = {"padding": True, "truncation": False, "return_tensors": "pt"}
            if tokenizer_max_length is not None:
                kwargs["max_length"] = tokenizer_max_length
            encoded = self.tokenizer(batch, **kwargs)
            encoded = {key: value.to(self.device_resolved) for key, value in encoded.items()}
            with torch.no_grad():
                output = self.model(**encoded)
            token_vectors = output.last_hidden_state
            mask = encoded.get("attention_mask")
            if mask is None:
                mask = torch.ones(token_vectors.shape[:2], device=token_vectors.device, dtype=token_vectors.dtype)
            mask = mask.unsqueeze(-1).to(token_vectors.dtype)
            pooled = (token_vectors * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            vectors.append(pooled.detach().cpu().numpy().astype(np.float32))
        return np.vstack(vectors).astype(np.float32)


def source_payload(row: pd.Series, sequence: str, windows: list[tuple[int, int, str]]) -> str:
    payload = {
        "feature_key": clean_value(row.get("feature_key")) or clean_value(row.get("node_id")),
        "node_id": clean_value(row.get("node_id")),
        "node_type": clean_value(row.get("node_type")) or "transcript",
        "sequence_kind": clean_value(row.get("sequence_kind")) or "cdna",
        "alphabet": clean_value(row.get("alphabet")) or "dna_iupac",
        "sequence_sha256": sha256_text(sequence),
        "source_checksum_sha256": clean_value(row.get("checksum_sha256")),
        "window_spans": [(start, end) for start, end, _ in windows],
        "source": clean_value(row.get("source")),
        "source_dataset": clean_value(row.get("source_dataset")),
        "source_record_id": clean_value(row.get("source_record_id")),
        "source_release": clean_value(row.get("source_release")) or clean_value(row.get("release")),
    }
    return stable_json(payload)


def build_embeddings(
    kg_root: Path,
    output_dir: Path,
    encoder: NucleotideTransformerEncoder,
    limit: int | None,
    batch_size: int,
    max_nucleotides_per_window: int,
    window_stride: int,
    max_windows_per_sequence: int | None,
    tokenizer_max_length: int | None,
    part_size: int | None = None,
    row_start: int = 0,
    row_end: int | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    path = source_path(kg_root)
    source_meta = parquet_metadata(path)
    length_audit = audit_sequence_lengths(path, max_nucleotides_per_window, window_stride)
    total_rows = int(source_meta["rows"])
    selected_start = max(int(row_start), 0)
    requested_end = total_rows if row_end is None else min(int(row_end), total_rows)
    if limit is not None:
        requested_end = min(requested_end, selected_start + int(limit))
    selected_end = max(selected_start, requested_end)
    selected_rows = selected_end - selected_start
    chunk_size = int(part_size or selected_rows or 1)
    if chunk_size <= 0:
        raise ValueError("part_size must be positive")
    embedding_dir = output_dir / "features" / "embeddings" / "transcript_nucleotide" / "transcript" / encoder.model_name.lower().replace("/", "__") / "policy_v1"
    skipped_dir = output_dir / "reports" / "transcript_nucleotide_skipped_parts"
    skipped_path = output_dir / "reports" / "transcript_nucleotide_skipped_rows.parquet"
    audit_path = output_dir / "reports" / "transcript_sequence_length_audit.json"
    part_records: list[dict[str, Any]] = []
    embedding_part_paths: list[str] = []
    skipped_part_paths: list[str] = []
    embedded_total = skipped_total = windows_seen = 0

    for chunk_start in range(selected_start, selected_end, chunk_size):
        chunk_end = min(chunk_start + chunk_size, selected_end)
        embedding_path = parts.part_path(embedding_dir, chunk_start, chunk_end)
        skipped_part_path = parts.part_path(skipped_dir, chunk_start, chunk_end)
        meta_path = parts.part_meta_path(embedding_dir, chunk_start, chunk_end)
        expected_meta = {
            "source_path": str(path),
            "source_sha256": source_meta["sha256"],
            "source_rows": total_rows,
            "row_start": chunk_start,
            "row_end": chunk_end,
            "model_name": encoder.model_name,
            "model_version": encoder.model_version,
            "embedding_dim": encoder.embedding_dim,
            "policy_version": POLICY_VERSION,
            "max_nucleotides_per_window": max_nucleotides_per_window,
            "window_stride": window_stride,
            "max_windows_per_sequence": max_windows_per_sequence,
            "tokenizer_max_length": tokenizer_max_length,
        }
        valid, checks = parts.can_skip_valid_part(
            embedding_path,
            skipped_part_path,
            meta_path,
            expected=expected_meta,
            expected_dim=encoder.embedding_dim,
            skipped_required_columns=SKIPPED_ROW_COLUMNS,
        )
        if resume and valid:
            emb_df = pd.read_parquet(embedding_path)
            skip_df = pd.read_parquet(skipped_part_path)
            part_records.append({**expected_meta, "embedding_path": str(embedding_path), "skipped_path": str(skipped_part_path), "meta_path": str(meta_path), "status": "reused", "validation": checks, "embedded_rows": int(len(emb_df)), "skipped_rows": int(len(skip_df))})
            embedding_part_paths.append(str(embedding_path))
            skipped_part_paths.append(str(skipped_part_path))
            embedded_total += int(len(emb_df))
            skipped_total += int(len(skip_df))
            windows_seen += int(emb_df.get("window_count", pd.Series(dtype=int)).sum()) if not emb_df.empty else 0
            continue

        df = parts.read_row_range(path, chunk_start, chunk_end)
        rows: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []
        prepared: list[tuple[int, pd.Series, str, list[tuple[int, int, str]], str]] = []
        for _, row in df.iterrows():
            row_index = int(row["__source_row_index"])
            node_id = clean_value(row.get("node_id"))
            feature_key = clean_value(row.get("feature_key")) or node_id
            sequence = clean_value(row.get("sequence")).upper().replace("U", "T")
            declared_length = int(row.get("length") or len(sequence) or 0)
            alphabet = clean_value(row.get("alphabet"))
            skip_reason = None
            if not sequence:
                skip_reason = "missing_or_empty_sequence"
            elif alphabet != "dna_iupac":
                skip_reason = f"non_dna_iupac_alphabet_{alphabet or 'missing'}"
            else:
                skip_reason = invalid_dna_iupac_reason(sequence)
            windows = sequence_windows(sequence, max_nucleotides_per_window, window_stride) if not skip_reason else []
            if not skip_reason and max_windows_per_sequence is not None and len(windows) > max_windows_per_sequence:
                skip_reason = f"window_count_{len(windows)}_exceeds_max_windows_per_sequence_{max_windows_per_sequence}"
            if skip_reason:
                skipped.append({"row_index": row_index, "node_id": node_id, "feature_key": feature_key, "sequence_length": declared_length, "skip_reason": skip_reason, "long_sequence_policy": LONG_SEQUENCE_POLICY, "created_at": CREATED_AT})
                continue
            payload = source_payload(row, sequence, windows)
            prepared.append((row_index, row, sequence, windows, payload))

        flat_windows = [window_seq for _, _, _, windows, _ in prepared for _, _, window_seq in windows]
        window_vectors = encoder.encode_windows(flat_windows, batch_size=batch_size, tokenizer_max_length=tokenizer_max_length)
        cursor = 0
        for source_row_index, row, sequence, windows, payload in prepared:
            n_windows = len(windows)
            vectors = window_vectors[cursor : cursor + n_windows]
            cursor += n_windows
            if vectors.size == 0:
                continue
            vector = vectors.mean(axis=0).astype(np.float32)
            norm = float(np.linalg.norm(vector))
            if norm > 0:
                vector = vector / norm
            node_id = clean_value(row.get("node_id"))
            feature_key = clean_value(row.get("feature_key")) or node_id
            payload_hash = sha256_text(payload)
            rows.append({
                "embedding_key": f"{node_id}|transcript|{encoder.model_name}|{encoder.model_version}|{feature_key}|{payload_hash}|mean_pool_window_mean_l2",
                "node_id": node_id,
                "node_type": clean_value(row.get("node_type")) or "transcript",
                "source_feature_table": "features/transcript_sequence.parquet",
                "source_feature_key": feature_key,
                "source_feature_hash": payload_hash,
                "source_sequence_sha256": clean_value(row.get("checksum_sha256")) or sha256_text(sequence),
                "modality": "transcript_cdna_nucleotide_sequence",
                "sequence_kind": clean_value(row.get("sequence_kind")) or "cdna",
                "alphabet": clean_value(row.get("alphabet")) or "dna_iupac",
                "embedding_model": encoder.model_name,
                "embedding_version": encoder.model_version,
                "embedding_dim": int(vector.shape[0]),
                "embedding_dtype": "float32",
                "embedding_format": "list_float32",
                "embedding": vector.astype(np.float32).tolist(),
                "pooling": "attention_masked_mean_pool_last_hidden_state_per_window_then_mean_of_window_vectors",
                "normalization": "l2",
                "preprocessing": stable_json({
                    "long_sequence_policy": LONG_SEQUENCE_POLICY,
                    "max_nucleotides_per_window": max_nucleotides_per_window,
                    "window_stride": window_stride,
                    "max_windows_per_sequence": max_windows_per_sequence,
                    "tokenizer_truncation": False,
                    "tokenizer_max_length": tokenizer_max_length,
                    "transcript_gene_policy": "transcript embeddings are not averaged/collapsed to gene embeddings",
                }),
                "input_length": len(sequence),
                "source_row_index": source_row_index,
                "window_count": n_windows,
                "window_spans": stable_json([(start, end) for start, end, _ in windows]),
                "created_at": CREATED_AT,
                "source_feature_release": clean_value(row.get("source_release")) or clean_value(row.get("release")),
                "provenance": stable_json({
                    "task_id": TASK_ID,
                    "run_id": RUN_ID,
                    "kg_root": str(kg_root),
                    "input_path": str(path),
                    "payload": payload,
                    "part_row_range": [chunk_start, chunk_end],
                }),
                "license": clean_value(row.get("license")) or "inherits source transcript sequence feature table license; Nucleotide Transformer model license",
                "citation": clean_value(row.get("citation")) or "source transcript sequence feature table citation; Nucleotide Transformer model card",
            })
        parts.write_parquet(rows, embedding_path)
        parts.write_parquet(skipped, skipped_part_path, columns=SKIPPED_ROW_COLUMNS)
        parts.write_json(meta_path, {**expected_meta, "created_at": CREATED_AT, "embedded_rows": len(rows), "skipped_rows": len(skipped), "window_count": int(sum(row["window_count"] for row in rows))})
        part_checks = parts.validate_part_files(embedding_path, skipped_part_path, meta_path, expected=expected_meta, expected_dim=encoder.embedding_dim, skipped_required_columns=SKIPPED_ROW_COLUMNS)
        part_records.append({**expected_meta, "embedding_path": str(embedding_path), "skipped_path": str(skipped_part_path), "meta_path": str(meta_path), "status": "written", "validation": part_checks, "embedded_rows": len(rows), "skipped_rows": len(skipped)})
        embedding_part_paths.append(str(embedding_path))
        skipped_part_paths.append(str(skipped_part_path))
        embedded_total += len(rows)
        skipped_total += len(skipped)
        windows_seen += int(sum(row["window_count"] for row in rows))

    aggregate_skipped = parts.concat_parquets(skipped_part_paths)
    write_parquet(aggregate_skipped.to_dict("records"), skipped_path, columns=SKIPPED_ROW_COLUMNS)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(length_audit, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "source": source_meta,
        "source_label": "features/transcript_sequence.parquet",
        "length_audit": length_audit,
        "length_audit_path": str(audit_path),
        "row_start": selected_start,
        "row_end": selected_end,
        "part_size": chunk_size,
        "limited_source_rows_read": int(selected_rows),
        "embedded_rows": int(embedded_total),
        "skipped_rows": int(skipped_total),
        "long_sequence_policy": LONG_SEQUENCE_POLICY,
        "output_path": str(embedding_dir),
        "output_part_paths": embedding_part_paths,
        "skipped_part_paths": skipped_part_paths,
        "parts": part_records,
        "skipped_rows_path": str(skipped_path),
        "embedding_dim": encoder.embedding_dim,
        "windows_embedded": windows_seen,
        "max_nucleotides_per_window": max_nucleotides_per_window,
        "window_stride": window_stride,
        "max_windows_per_sequence": max_windows_per_sequence,
        "tokenizer_max_length": tokenizer_max_length,
    }

def validate_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    output = manifest["outputs"]["transcript_nucleotide_embeddings"]
    df = parts.concat_parquets(output.get("output_part_paths", [output["output_path"]]))
    skipped = pd.read_parquet(output["skipped_rows_path"])
    duplicate_keys = int(df["embedding_key"].duplicated().sum()) if not df.empty else 0
    duplicate_feature_keys = int(df["source_feature_key"].duplicated().sum()) if not df.empty and "source_feature_key" in df.columns else 0
    duplicate_covered_rows = 0
    row_coverage_matches = False
    if "source_row_index" in df.columns and "row_index" in skipped.columns:
        covered_rows = [int(v) for v in df["source_row_index"].tolist()] + [int(v) for v in skipped["row_index"].tolist()]
        duplicate_covered_rows = len(covered_rows) - len(set(covered_rows))
        row_coverage_matches = sorted(covered_rows) == list(range(int(output.get("row_start", 0)), int(output.get("row_end", output["limited_source_rows_read"]))))
    bad_dims = non_finite = all_zero = gene_like_rows = 0
    for _, row in df.iterrows():
        vector = np.asarray(row["embedding"], dtype=np.float32)
        expected_len = int(row.get("embedding_dim") if row.get("embedding_dim") is not None else -1)
        if len(vector) != expected_len:
            bad_dims += 1
        if not np.isfinite(vector).all():
            non_finite += 1
        if float(np.linalg.norm(vector)) == 0.0:
            all_zero += 1
        if row.get("node_type") == "gene" or row.get("modality") == "gene_sequence":
            gene_like_rows += 1
    source_rows_match = int(len(df) + len(skipped)) == int(output["limited_source_rows_read"])
    length_audit_covers_source = int(output["length_audit"]["source_rows"]) == int(output["source"]["rows"])
    no_truncation_policy_declared = "Tokenizer truncation is disabled" in output["long_sequence_policy"]
    no_omoc_in_manifest = ".omoc" not in stable_json(manifest)
    part_validations = [part.get("validation", {}) for part in output.get("parts", [])]
    all_parts_valid = bool(part_validations) and all(bool(check.get("passed")) for check in part_validations)
    schema_checks = parts.validate_manifest_schema_minimal({**manifest, "validation": {"passed": True}}, Path(__file__).resolve().parent.parent / "docs" / "foundation_embedding_manifest.schema.json")
    checks = {
        "parquet_files_checked": int(len(output.get("output_part_paths", [])) + 1),
        "part_files_checked": int(len(output.get("parts", []))),
        "all_parts_valid": all_parts_valid,
        "duplicate_embedding_keys": duplicate_keys,
        "duplicate_source_feature_keys": duplicate_feature_keys,
        "duplicate_covered_rows": int(duplicate_covered_rows),
        "bad_dims": bad_dims,
        "non_finite_vectors": non_finite,
        "all_zero_vectors": all_zero,
        "gene_like_rows": gene_like_rows,
        "source_rows_total": int(output["source"]["rows"]),
        "source_rows_read": int(output["limited_source_rows_read"]),
        "source_rows_embedded": int(len(df)),
        "skipped_rows": int(len(skipped)),
        "source_rows_match_embedded_plus_skipped": source_rows_match,
        "row_range_coverage_matches_embedded_plus_skipped": row_coverage_matches,
        "length_audit_covers_full_source_table": length_audit_covers_source,
        "no_tokenizer_truncation_policy_declared": no_truncation_policy_declared,
        "no_omoc_paths_in_manifest": no_omoc_in_manifest,
        "manifest_schema": schema_checks,
    }
    checks["passed"] = (
        duplicate_keys == 0 and duplicate_feature_keys == 0 and duplicate_covered_rows == 0
        and bad_dims == 0 and non_finite == 0 and all_zero == 0 and gene_like_rows == 0
        and source_rows_match and row_coverage_matches and all_parts_valid and int(len(df)) > 0
        and length_audit_covers_source and no_truncation_policy_declared and no_omoc_in_manifest and schema_checks["passed"]
    )
    return checks


def write_summary(output_dir: Path, manifest: dict[str, Any]) -> str:
    path = output_dir / "transcript_nucleotide_embedding_summary.md"
    out = manifest["outputs"]["transcript_nucleotide_embeddings"]
    audit = out["length_audit"]
    lines = [
        "# Staged transcript cDNA nucleotide embedding smoke summary",
        "",
        f"Task: `{manifest['task_id']}`",
        f"Run id: `{manifest['run_id']}`",
        f"Created at: `{manifest['created_at']}`",
        "",
        "## Gate",
        "",
        "Staged-only derived transcript features. No canonical KG files were promoted or overwritten.",
        "Transcript embeddings remain transcript-level outputs and are not averaged/collapsed to gene embeddings.",
        "This smoke uses real Nucleotide Transformer inference unless the unit-test-only deterministic encoder flag is set.",
        "No `.omoc` path is used for input/output commands, manifest, or produced outputs; canonical GCS inputs are read back through the local cache when --gcs-kg-root is used.",
        "",
        "## Input and length/window audit",
        "",
        f"- Source: `{out['source']['path']}` ({out['source']['rows']} total rows, SHA256 `{out['source']['sha256']}`)",
        f"- Length audit path: `{out['length_audit_path']}`",
        f"- Alphabet audit: non-dna_iupac rows {audit['non_dna_iupac_rows']}; missing/empty {audit['missing_or_empty_sequences']}",
        f"- Length range: {audit['min_length']}..{audit['max_length']}; mean {audit['mean_length']:.2f}; quantiles `{stable_json(audit['quantiles'])}`",
        f"- Explicit window/stride: {audit['max_nucleotides_per_window']} nt / stride {audit['window_stride']} nt; over-window rows {audit['sequences_longer_than_max_window']}; estimated total windows {audit['estimated_total_windows']}; estimated max windows/sequence {audit['estimated_max_windows_per_sequence']}",
        f"- Long-sequence/skipped policy: {out['long_sequence_policy']}",
        "",
        "## Output",
        "",
        f"- Transcript nucleotide embeddings: {out['embedded_rows']} rows, dim {out['embedding_dim']} -> `{out['output_path']}`",
        f"- Skipped rows: {out['skipped_rows']} -> `{out['skipped_rows_path']}`",
        "",
        "## Runtime / model",
        "",
        f"- Runtime seconds: {manifest['runtime_seconds']:.3f}",
        f"- Batch size: {manifest['batch_size']}",
        f"- Device: `{manifest['environment']['device']}`; torch cuda available: `{manifest['environment']['torch_cuda_available']}`; torch mps available: `{manifest['environment']['torch_mps_available']}`",
        f"- Model: `{manifest['models']['nucleotide_transformer']['embedding_model']}` requested revision `{manifest['models']['nucleotide_transformer']['requested_revision']}` resolved `{manifest['models']['nucleotide_transformer']['resolved_revision']}`",
        f"- DNABERT-2 fallback (not used unless explicitly selected): `{DNABERT2_FALLBACK_MODEL}@{DNABERT2_FALLBACK_REVISION}`",
        "",
        "## Validation",
        "",
        f"Validation passed: `{manifest['validation']['passed']}`; checks: `{stable_json(manifest['validation'])}`",
        "",
        "## Recompute command",
        "",
        "```bash",
        manifest["recompute_command"],
        "```",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)


def run(
    kg_root: Path,
    output_dir: Path,
    limit: int | None = 1,
    batch_size: int = 1,
    model_name: str = MODEL_NAME,
    model_revision: str = MODEL_REVISION,
    device: str = "auto",
    max_nucleotides_per_window: int = 6000,
    window_stride: int = 6000,
    max_windows_per_sequence: int | None = None,
    tokenizer_max_length: int | None = None,
    clean: bool = False,
    test_deterministic_encoder: bool = False,
    gcs_kg_root: str | None = None,
    local_cache_dir: Path | None = None,
    part_size: int | None = None,
    row_start: int = 0,
    row_end: int | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    if clean and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_cache: dict[str, Any] | None = None
    if gcs_kg_root:
        if local_cache_dir is None:
            local_cache_dir = output_dir.parent.parent / "cache" / f"{TASK_ID}_transcript_nucleotide_kg_cache"
        input_cache = copy_gcs_input(gcs_kg_root, local_cache_dir)
        kg_root = local_cache_dir
    start = time.perf_counter()
    encoder = NucleotideTransformerEncoder(model_name=model_name, revision=model_revision, device=device, test_deterministic=test_deterministic_encoder)
    outputs = {"transcript_nucleotide_embeddings": build_embeddings(kg_root, output_dir, encoder, limit, batch_size, max_nucleotides_per_window, window_stride, max_windows_per_sequence, tokenizer_max_length, part_size=part_size, row_start=row_start, row_end=row_end, resume=resume)}
    runtime_seconds = time.perf_counter() - start
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
        "batch_size": batch_size,
        "resume": resume,
        "row_range": [row_start, row_end],
        "part_size": part_size,
        "models": {
            "nucleotide_transformer": {
                "embedding_model": encoder.model_name,
                "embedding_version": encoder.model_version,
                "requested_revision": model_revision,
                "resolved_revision": encoder.resolved_revision,
                "embedding_dim": encoder.embedding_dim,
                "normalization": "l2",
                "pooling": "attention_masked_mean_pool_last_hidden_state_per_window_then_mean_of_window_vectors",
                "fallback_allowed_only_if_documented": f"{DNABERT2_FALLBACK_MODEL}@{DNABERT2_FALLBACK_REVISION}",
            }
        },
        "outputs": outputs,
        "runtime_seconds": runtime_seconds,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pyarrow": pa.__version__,
            "torch_available": torch is not None,
            "torch_cuda_available": bool(torch is not None and torch.cuda.is_available()),
            "torch_mps_available": bool(torch is not None and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
            "transformers_available": AutoConfig is not None and AutoTokenizer is not None and AutoModelForMaskedLM is not None,
            "device": encoder.device_resolved,
        },
        "recompute_command": " ".join([
            "uv run --group embeddings-nucleotide python -m manage_db.build_nucleotide_sequence_embeddings",
            *( [f"--gcs-kg-root {input_cache['source_gcs_root']}", f"--local-cache-dir {input_cache['local_cache_dir']}"] if input_cache else [f"--kg-root {kg_root}"] ),
            f"--output-dir {output_dir}",
            f"--limit {limit}" if limit is not None else "--no-limit",
            f"--batch-size {batch_size}",
            f"--model-name {model_name}",
            f"--model-revision {model_revision}",
            f"--device {device}",
            f"--max-nucleotides-per-window {max_nucleotides_per_window}",
            f"--window-stride {window_stride}",
            *( [f"--max-windows-per-sequence {max_windows_per_sequence}"] if max_windows_per_sequence is not None else [] ),
            *( [f"--tokenizer-max-length {tokenizer_max_length}"] if tokenizer_max_length is not None else [] ),
            *( [f"--part-size {part_size}"] if part_size is not None else [] ),
            f"--row-start {row_start}",
            *( [f"--row-end {row_end}"] if row_end is not None else [] ),
            *( ["--resume"] if resume else [] ),
            "--clean",
        ]),
    }
    manifest["validation"] = validate_manifest(manifest)
    manifest["summary_path"] = write_summary(output_dir, manifest)
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build staged transcript cDNA nucleotide embeddings with pinned Nucleotide Transformer and explicit length/window audit.")
    parser.add_argument("--kg-root", type=Path, default=Path("/Users/jkobject/mnt/gcs/jouvencekb/main"))
    parser.add_argument("--gcs-kg-root", default=None, help="Canonical GCS KG root, e.g. gs://jouvencekb/main. When set, copy transcript_sequence.parquet to --local-cache-dir first and read locally.")
    parser.add_argument("--local-cache-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path(f"artifacts/staged/{TASK_ID}/transcript_nucleotide_smoke"))
    parser.add_argument("--limit", type=int, default=1, help="Number of source rows to embed for bounded smoke; ignored with --no-limit.")
    parser.add_argument("--no-limit", action="store_true", help="Read/embed the full source table; use only for reviewed production runs.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-nucleotides-per-window", type=int, default=6000)
    parser.add_argument("--window-stride", type=int, default=6000)
    parser.add_argument("--max-windows-per-sequence", type=int, default=None, help="Optional explicit skip policy for transcripts requiring too many windows.")
    parser.add_argument("--tokenizer-max-length", type=int, default=None, help="Optional tokenizer max length metadata; truncation remains disabled.")
    parser.add_argument("--part-size", type=int, default=None, help="Write deterministic part-<row-start>-<row-end>.parquet chunks of this many source rows.")
    parser.add_argument("--row-start", type=int, default=0, help="Inclusive source row offset for bounded/resumable runs.")
    parser.add_argument("--row-end", type=int, default=None, help="Exclusive source row offset for bounded/resumable runs.")
    parser.add_argument("--resume", action="store_true", help="Skip already-valid part files with matching source/model/policy metadata.")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--test-deterministic-encoder", action="store_true", help="Use only in unit tests; not a real Nucleotide Transformer embedding.")
    args = parser.parse_args(argv)
    manifest = run(
        kg_root=args.kg_root,
        output_dir=args.output_dir,
        limit=None if args.no_limit else args.limit,
        batch_size=args.batch_size,
        model_name=args.model_name,
        model_revision=args.model_revision,
        device=args.device,
        max_nucleotides_per_window=args.max_nucleotides_per_window,
        window_stride=args.window_stride,
        max_windows_per_sequence=args.max_windows_per_sequence,
        tokenizer_max_length=args.tokenizer_max_length,
        clean=args.clean,
        test_deterministic_encoder=args.test_deterministic_encoder,
        gcs_kg_root=args.gcs_kg_root,
        local_cache_dir=args.local_cache_dir,
        part_size=args.part_size,
        row_start=args.row_start,
        row_end=args.row_end,
        resume=args.resume,
    )
    print(json.dumps({
        "manifest_path": str(Path(manifest["output_dir"]) / "manifest.json"),
        "summary_path": manifest["summary_path"],
        "length_audit_path": manifest["outputs"]["transcript_nucleotide_embeddings"]["length_audit_path"],
        "validation": manifest["validation"],
        "runtime_seconds": manifest["runtime_seconds"],
        "environment": manifest["environment"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
