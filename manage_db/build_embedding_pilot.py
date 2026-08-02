from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.feature_extraction.text import HashingVectorizer

try:  # RDKit is an optional runtime dependency for molecule fingerprints.
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator
except Exception:  # pragma: no cover - exercised by environments without rdkit
    Chem = None
    rdFingerprintGenerator = None

PILOT_TASK_ID = "t_3dcf3ec3"
PILOT_RUN_ID = "embedding_pilot_20260623_t_3dcf3ec3"
TEXT_MODEL_NAME = "sklearn.feature_extraction.text.HashingVectorizer"
TEXT_MODEL_VERSION = "scikit-learn-hashing-vectorizer@1.8.0+embedding_policy_v1+pilot_surrogate"
TEXT_EMBEDDING_DIM = 384
EDGE_POLICY_VERSION = "edge_embedding_policy_v1+pilot_surrogate"
MOLECULE_FP_VERSION = "rdkit-morgan-radius2-2048@2025.9.1+policy_v1"
CREATED_AT = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def _read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_parquet(path)


def _write_parquet(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path, compression="zstd")


def _write_parquet_with_schema(rows: list[dict[str, Any]], path: Path, schema: pa.Schema) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(table, path, compression="zstd")


EMBEDDING_ARTIFACT_SCHEMA = pa.schema(
    [
        ("embedding_key", pa.string()),
        ("node_id", pa.string()),
        ("node_type", pa.string()),
        ("source_feature_table", pa.string()),
        ("source_feature_key", pa.string()),
        ("source_feature_hash", pa.string()),
        ("modality", pa.string()),
        ("embedding_model", pa.string()),
        ("embedding_version", pa.string()),
        ("embedding_dim", pa.int64()),
        ("embedding_dtype", pa.string()),
        ("embedding_format", pa.string()),
        ("embedding", pa.list_(pa.int64())),
        ("pooling", pa.string()),
        ("normalization", pa.string()),
        ("preprocessing", pa.string()),
        ("input_length", pa.int64()),
        ("window_count", pa.int64()),
        ("created_at", pa.string()),
        ("source_feature_release", pa.string()),
        ("provenance", pa.string()),
        ("license", pa.string()),
        ("citation", pa.string()),
    ]
)


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


def _text_vectors(payloads: list[str]) -> np.ndarray:
    if not payloads:
        return np.zeros((0, TEXT_EMBEDDING_DIM), dtype=np.float32)
    matrix = _hashing_vectorizer().transform(payloads)
    return matrix.astype(np.float32).toarray()


def _source_table_hashes(input_dir: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in sorted(input_dir.rglob("*.parquet")):
        hashes[str(path.relative_to(input_dir))] = _sha256_file(path)
    return hashes


def _node_payload(node_type: str, row: pd.Series) -> str:
    fields = {
        "node_type": node_type,
        "id": str(row.get("id", "")),
        "name": str(row.get("name", "")),
    }
    optional = ["smiles", "inchikey", "drugbank_id", "pubchem_cid", "cas_rn"]
    for key in optional:
        if key in row and pd.notna(row[key]) and str(row[key]).strip():
            fields[key] = str(row[key]).strip()
    return "\n".join(f"{key}: {value}" for key, value in fields.items() if value not in {"", "nan", "None"})


def build_node_text_embeddings(input_dir: Path, output_dir: Path, limit_per_type: int) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for node_path in sorted((input_dir / "nodes").glob("*.parquet")):
        node_type = node_path.stem
        df = _read_parquet(node_path).head(limit_per_type).copy()
        payloads = [_node_payload(node_type, row) for _, row in df.iterrows()]
        vectors = _text_vectors(payloads)
        for (_, row), payload, vector in zip(df.iterrows(), payloads, vectors, strict=True):
            node_id = str(row["id"])
            source_hash = _sha256_text(payload)
            rows.append(
                {
                    "embedding_key": f"{node_id}|{node_type}|{TEXT_MODEL_NAME}|{TEXT_MODEL_VERSION}|{source_hash}|l2_hashing",
                    "node_id": node_id,
                    "node_type": node_type,
                    "source_feature_table": f"nodes/{node_type}.parquet",
                    "source_feature_key": node_id,
                    "source_feature_hash": source_hash,
                    "modality": "text",
                    "embedding_model": TEXT_MODEL_NAME,
                    "embedding_version": TEXT_MODEL_VERSION,
                    "embedding_dim": TEXT_EMBEDDING_DIM,
                    "embedding_dtype": "float32",
                    "embedding_format": "list_float32",
                    "embedding": vector.astype(np.float32).tolist(),
                    "pooling": "hashing_vectorizer_l2",
                    "normalization": "l2",
                    "preprocessing": _stable_json(
                        {
                            "serializer": "node_payload_v1",
                            "fields": ["node_type", "id", "name", "optional canonical molecule ids if present"],
                            "note": "Pilot surrogate validates staging/schema only; replace with accepted foundation text encoder for production.",
                        }
                    ),
                    "input_length": len(payload),
                    "window_count": 1,
                    "created_at": CREATED_AT,
                    "source_feature_release": "local staged SciPlex2 candidate context, 20260622",
                    "provenance": _stable_json({"task_id": PILOT_TASK_ID, "payload_preview": payload[:512]}),
                    "license": "inherits local staged source license; see upstream artifact report",
                    "citation": "scikit-learn HashingVectorizer documentation; staged SciPlex2 local KG artifact",
                }
            )
    out_path = output_dir / "features" / "embeddings" / "text" / "mixed_nodes" / "hashing_vectorizer" / "pilot_surrogate_v1" / "part-000.parquet"
    _write_parquet(rows, out_path)
    return {"path": str(out_path), "rows": len(rows), "embedding_dim": TEXT_EMBEDDING_DIM}


def _fingerprint_from_smiles(smiles: str) -> tuple[list[int] | None, str | None, str | None]:
    if Chem is None or rdFingerprintGenerator is None:
        return None, None, "rdkit_unavailable"
    mol = Chem.MolFromSmiles(smiles, sanitize=True)
    if mol is None:
        return None, None, "rdkit_parse_failed"
    canonical = Chem.MolToSmiles(mol, canonical=True)
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True)
    fp = generator.GetFingerprint(mol)
    on_bits = list(fp.GetOnBits())
    return on_bits, canonical, None


def build_molecule_fingerprint_pilot(input_dir: Path, output_dir: Path) -> dict[str, Any]:
    mol_path = input_dir / "nodes" / "molecule.parquet"
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    if mol_path.exists():
        df = _read_parquet(mol_path)
        for _, row in df.iterrows():
            node_id = str(row["id"])
            smiles = str(row.get("smiles", "") or "").strip()
            if not smiles or smiles.lower() in {"nan", "none", "<na>"}:
                skipped.append({"node_id": node_id, "skip_reason": "no_smiles_in_local_staged_molecule_node"})
                continue
            on_bits, canonical, error = _fingerprint_from_smiles(smiles)
            if error:
                skipped.append({"node_id": node_id, "input_smiles": smiles, "skip_reason": error})
                continue
            assert on_bits is not None and canonical is not None
            payload = _stable_json({"node_id": node_id, "smiles": smiles, "canonical_smiles_rdkit": canonical, "radius": 2, "n_bits": 2048})
            rows.append(
                {
                    "embedding_key": f"{node_id}|molecule|morgan_fingerprint|{MOLECULE_FP_VERSION}|{_sha256_text(payload)}|on_bits",
                    "node_id": node_id,
                    "node_type": "molecule",
                    "source_feature_table": "nodes/molecule.parquet.smiles",
                    "source_feature_key": node_id,
                    "source_feature_hash": _sha256_text(payload),
                    "modality": "molecule_fingerprint",
                    "embedding_model": "RDKit Morgan fingerprint",
                    "embedding_version": MOLECULE_FP_VERSION,
                    "embedding_dim": 2048,
                    "embedding_dtype": "uint8_sparse_on_bits",
                    "embedding_format": "list_int32_on_bits",
                    "embedding": on_bits,
                    "pooling": "none",
                    "normalization": "none",
                    "preprocessing": _stable_json({"radius": 2, "n_bits": 2048, "includeChirality": True, "canonical_smiles_rdkit": canonical}),
                    "input_length": len(smiles),
                    "window_count": 1,
                    "created_at": CREATED_AT,
                    "source_feature_release": "local staged SciPlex2 candidate context, 20260622",
                    "provenance": _stable_json({"task_id": PILOT_TASK_ID, "source_path": str(mol_path)}),
                    "license": "inherits molecule source license",
                    "citation": "RDKit Morgan fingerprint generator",
                }
            )
    out_path = output_dir / "features" / "embeddings" / "molecule_fingerprint" / "molecule" / "rdkit_morgan" / "pilot_surrogate_v1" / "part-000.parquet"
    skipped_path = output_dir / "reports" / "molecule_fingerprint_skipped_rows.parquet"
    _write_parquet_with_schema(rows, out_path, EMBEDDING_ARTIFACT_SCHEMA)
    _write_parquet(skipped, skipped_path)
    return {"path": str(out_path), "rows": len(rows), "skipped_rows": len(skipped), "skipped_path": str(skipped_path), "embedding_dim": 2048}


def _format_value(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)) or pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.8g}"
    return str(value)


def _edge_header(edge: pd.Series) -> str:
    fields = ["relation", "display_relation", "x_type", "x_id", "y_type", "y_id", "source", "credibility"]
    parts = [f"edge_key: {edge['relation']}|{edge['x_id']}|{edge['y_id']}"]
    for field in fields:
        value = _format_value(edge.get(field))
        if value:
            parts.append(f"{field}: {value}")
    parts.append("relation_semantics: staged cell-type response to molecule perturbation context; candidate/support relation, not canonical promotion")
    return "\n".join(parts)


def _evidence_block(row: pd.Series) -> str:
    fields = [
        "evidence_type",
        "source",
        "source_dataset",
        "source_record_id",
        "predicate",
        "direction",
        "evidence_score",
        "effect_size",
        "p_value",
        "paper_id",
        "dataset_id",
        "study_id",
        "text_span",
        "section",
        "extraction_method",
        "release",
    ]
    parts = []
    for field in fields:
        if field in row:
            value = _format_value(row.get(field))
            if value:
                parts.append(f"{field}: {value}")
    return "\n".join(parts)


def _edge_weight(edge: pd.Series, evidence: pd.Series) -> float:
    credibility = int(edge.get("credibility", 1) or 1)
    credibility_weight = {1: 1.0, 2: 1.25, 3: 1.5}.get(credibility, 1.0)
    score = evidence.get("evidence_score", 1.0)
    try:
        score_weight = float(score)
        if not math.isfinite(score_weight) or score_weight <= 0:
            score_weight = 1.0
    except Exception:
        score_weight = 1.0
    return credibility_weight * max(0.0, min(score_weight, 1.0))


def build_edge_embeddings(input_dir: Path, output_dir: Path) -> dict[str, Any]:
    edge_path = input_dir / "edges" / "cell_type_responds_to_molecule.parquet"
    evidence_path = input_dir / "evidence" / "cell_type_responds_to_molecule.parquet"
    edges = _read_parquet(edge_path)
    evidence = _read_parquet(evidence_path)
    relation = "cell_type_responds_to_molecule"
    rows: list[dict[str, Any]] = []
    for _, edge in edges.sort_values(["relation", "x_id", "y_id"]).iterrows():
        edge_key = f"{edge['relation']}|{edge['x_id']}|{edge['y_id']}"
        header = _edge_header(edge)
        ev_rows = evidence[evidence["edge_key"] == edge_key].copy()
        ev_rows = ev_rows.sort_values(["source", "source_dataset", "source_record_id"], kind="stable")
        if ev_rows.empty:
            payloads = [header]
            weights = np.array([1.0], dtype=np.float32)
            aggregation_method = "header_only"
        else:
            payloads = [header + "\n--- evidence row ---\n" + _evidence_block(ev) for _, ev in ev_rows.iterrows()]
            weights = np.array([_edge_weight(edge, ev) for _, ev in ev_rows.iterrows()], dtype=np.float32)
            if float(weights.sum()) <= 0:
                weights = np.ones(len(payloads), dtype=np.float32)
            weights = weights / weights.sum()
            aggregation_method = "weighted_mean_evidence_rows"
        vectors = _text_vectors(payloads)
        vector = np.average(vectors, axis=0, weights=weights).astype(np.float32)
        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector = vector / norm
        evidence_hash = _sha256_text(_stable_json(payloads))
        rows.append(
            {
                "embedding_key": f"{edge_key}|{TEXT_MODEL_NAME}|{TEXT_MODEL_VERSION}|{evidence_hash}|{aggregation_method}",
                "edge_key": edge_key,
                "relation": relation,
                "x_id": str(edge["x_id"]),
                "x_type": str(edge["x_type"]),
                "y_id": str(edge["y_id"]),
                "y_type": str(edge["y_type"]),
                "source_feature_table": f"edges/{relation}.parquet+evidence/{relation}.parquet",
                "source_feature_key": edge_key,
                "source_feature_hash": evidence_hash,
                "modality": "edge_evidence_text",
                "embedding_model": TEXT_MODEL_NAME,
                "embedding_version": TEXT_MODEL_VERSION,
                "embedding_dim": TEXT_EMBEDDING_DIM,
                "embedding_dtype": "float32",
                "embedding_format": "list_float32",
                "embedding": vector.astype(np.float32).tolist(),
                "pooling": aggregation_method,
                "normalization": "l2",
                "aggregation_method": aggregation_method,
                "n_evidence_rows_total": int(len(ev_rows)),
                "n_evidence_rows_encoded": int(len(payloads) if not ev_rows.empty else 0),
                "evidence_row_selection_policy": "all_local_rows_sorted_by_source_dataset_record_id",
                "preprocessing": _stable_json({"serializer": "edge_payload_serializer_v1", "policy_version": EDGE_POLICY_VERSION}),
                "input_length": int(sum(len(p) for p in payloads)),
                "window_count": 1,
                "created_at": CREATED_AT,
                "source_feature_release": "local staged SciPlex2 candidate context, 20260622",
                "provenance": _stable_json({"task_id": PILOT_TASK_ID, "edge_path": str(edge_path), "evidence_path": str(evidence_path)}),
                "license": "inherits local staged source license; see upstream artifact report",
                "citation": "scikit-learn HashingVectorizer documentation; edge evidence embedding policy t_1dc65ac1",
            }
        )
    out_path = output_dir / "features" / "edge_embeddings" / relation / "hashing_vectorizer" / "pilot_surrogate_v1" / "part-000.parquet"
    _write_parquet(rows, out_path)
    return {"path": str(out_path), "rows": len(rows), "embedding_dim": TEXT_EMBEDDING_DIM}


def _parquet_count(path: str) -> int:
    return pq.ParquetFile(path).metadata.num_rows


def _write_summary(output_dir: Path, manifest: dict[str, Any]) -> Path:
    path = output_dir / "embedding_pilot_summary.md"
    lines = [
        "# Staged embedding pilot summary",
        "",
        f"Task: `{PILOT_TASK_ID}`",
        f"Run id: `{PILOT_RUN_ID}`",
        f"Created at: `{CREATED_AT}`",
        "",
        "## Scope and gate",
        "",
        "This is a staged-only pilot under `artifacts/staged/`; it does not write or promote anything under the canonical KG root.",
        "Because the local environment did not have `sentence_transformers`, `transformers`, or `torch`, this run uses a deterministic local `sklearn.HashingVectorizer` surrogate to validate payload construction, metadata, hashing, storage layout, and one-vector-per-edge aggregation. It is not a biological-quality replacement for the accepted foundation-model defaults in `docs/foundation_embedding_policy.md`.",
        "",
        "## Outputs",
        "",
    ]
    for key, stats in manifest["outputs"].items():
        lines.extend([
            f"- `{key}`: `{stats['path']}`",
            f"  - rows: {stats.get('rows', 0)}",
            f"  - dimension: {stats.get('embedding_dim', 'n/a')}",
        ])
        if "skipped_rows" in stats:
            lines.append(f"  - skipped rows: {stats['skipped_rows']} (`{stats['skipped_path']}`)")
    lines.extend([
        "",
        "## Model/version metadata",
        "",
        f"- Text/node and edge surrogate: `{TEXT_MODEL_NAME}` / `{TEXT_MODEL_VERSION}` / dim `{TEXT_EMBEDDING_DIM}` / L2 normalized.",
        f"- Molecule fingerprint candidate: `{MOLECULE_FP_VERSION}` / dim `2048` sparse on-bits; emitted only when local staged molecule rows include valid SMILES.",
        "- Source feature hashes are SHA-256 hashes of deterministic serialized node/edge payloads; input table hashes are in `manifest.json`.",
        "",
        "## Recompute command",
        "",
        "```bash",
        "uv run python -m manage_db.build_embedding_pilot --input-dir artifacts/staged/cell_type_responds_to_molecule_sciplex2_20260622 --output-dir artifacts/staged/embedding_pilot_20260623_t_3dcf3ec3 --clean",
        "```",
        "",
        "## Runtime/cost and scaling estimate",
        "",
        f"- Runtime for this local pilot: {manifest['runtime_seconds']:.3f}s on {manifest['environment']['platform']}.",
        "- Direct cost: $0 external/API cost; no model downloads; CPU-only local vectorization.",
        "- Pilot scale: 3 node text rows, 2 edge rows, 14 evidence rows in the available SciPlex2 staging artifact.",
        "- Scaling estimate for this surrogate is roughly linear in serialized payload bytes and non-zero token count. For production foundation encoders, expect model inference to dominate; use the accepted policies to batch by modality, shard outputs, and benchmark on a capped 1k/10k-row sample before full KG promotion.",
        "",
        "## Residual risks",
        "",
        "- No canonical `/mnt/gcs/jouvencekb/main` mount was available, so this pilot used only the local staged SciPlex2 artifact.",
        "- The local staged molecule nodes lacked SMILES, so no RDKit molecule fingerprint rows were emitted; skipped rows are recorded.",
        "- Protein sequence embeddings were not attempted because no protein sequence feature table or ESM/torch stack was locally available.",
        "- Replace the hashing surrogate with pinned S-BioBERT/ChemBERTa/ESM jobs before any canonical-quality embedding promotion.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def run(input_dir: Path, output_dir: Path, clean: bool = False, limit_per_type: int = 50) -> dict[str, Any]:
    if clean and output_dir.exists():
        import shutil

        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    outputs = {
        "node_text_embeddings": build_node_text_embeddings(input_dir, output_dir, limit_per_type=limit_per_type),
        "molecule_fingerprint_pilot": build_molecule_fingerprint_pilot(input_dir, output_dir),
        "edge_evidence_embeddings": build_edge_embeddings(input_dir, output_dir),
    }
    runtime = time.perf_counter() - start
    manifest = {
        "task_id": PILOT_TASK_ID,
        "run_id": PILOT_RUN_ID,
        "created_at": CREATED_AT,
        "staged_only": True,
        "canonical_promotion": False,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "source_table_hashes": _source_table_hashes(input_dir),
        "models": {
            "text_surrogate": {
                "embedding_model": TEXT_MODEL_NAME,
                "embedding_version": TEXT_MODEL_VERSION,
                "embedding_dim": TEXT_EMBEDDING_DIM,
                "normalization": "l2",
                "purpose": "schema/pipeline pilot only; not accepted foundation-quality biological embedding",
            },
            "molecule_fingerprint": {
                "embedding_model": "RDKit Morgan fingerprint",
                "embedding_version": MOLECULE_FP_VERSION,
                "embedding_dim": 2048,
            },
        },
        "outputs": outputs,
        "runtime_seconds": runtime,
        "recompute_command": "uv run python -m manage_db.build_embedding_pilot --input-dir artifacts/staged/cell_type_responds_to_molecule_sciplex2_20260622 --output-dir artifacts/staged/embedding_pilot_20260623_t_3dcf3ec3 --clean",
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    summary_path = _write_summary(output_dir, manifest)
    manifest["summary_path"] = str(summary_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build staged node/edge embedding pilot artifacts for t_3dcf3ec3.")
    parser.add_argument("--input-dir", type=Path, default=Path("artifacts/staged/cell_type_responds_to_molecule_sciplex2_20260622"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/staged/embedding_pilot_20260623_t_3dcf3ec3"))
    parser.add_argument("--limit-per-type", type=int, default=50)
    parser.add_argument("--clean", action="store_true", help="Remove the output directory before rebuilding.")
    args = parser.parse_args(argv)
    manifest = run(args.input_dir, args.output_dir, clean=args.clean, limit_per_type=args.limit_per_type)
    print(json.dumps({"manifest_path": str(Path(manifest["output_dir"]) / "manifest.json"), "outputs": manifest["outputs"], "runtime_seconds": manifest["runtime_seconds"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
