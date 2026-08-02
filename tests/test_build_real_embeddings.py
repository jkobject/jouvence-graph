from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import manage_db.build_real_embeddings as builder
import pandas as pd
import pyarrow.parquet as pq

from manage_db.build_real_embeddings import EDGE_EMBEDDING_DIM, RUN_ID, TASK_ID, TEXT_MODEL_DIM, run


def _write_fixture(root: Path) -> None:
    features = root / "features"
    edges = root / "edges"
    evidence = root / "evidence"
    nodes = root / "nodes"
    features.mkdir(parents=True)
    edges.mkdir(parents=True)
    evidence.mkdir(parents=True)
    nodes.mkdir(parents=True)

    pd.DataFrame([{"id": "CHEMBL1", "name": "Aspirin"}]).to_parquet(nodes / "molecule.parquet", index=False)
    pd.DataFrame([{"id": "ENSG1", "name": "GENE1"}]).to_parquet(nodes / "gene.parquet", index=False)

    for table, node_type, node_id, text in [
        ("protein_textual_summary.parquet", "protein", "ENSP1", "Kinase-like protein involved in a fixture pathway."),
        ("gene_textual_summary.parquet", "gene", "ENSG1", "Fixture gene summary."),
        ("molecule_textual_summary.parquet", "molecule", "CHEMBL1", "Fixture molecule summary."),
    ]:
        pd.DataFrame(
            [
                {
                    "feature_key": f"{table}|{node_id}|1",
                    "feature_table": table.replace(".parquet", ""),
                    "node_id": node_id,
                    "node_type": node_type,
                    "summary_kind": "fixture_summary",
                    "summary_text": text,
                    "source": "fixture",
                    "source_dataset": "unit_test",
                    "source_record_id": "r1",
                    "provenance": "fixture",
                    "license": "fixture",
                    "citation": "fixture citation",
                    "release": "test",
                    "created_at": "2026-06-23T00:00:00Z",
                }
            ]
        ).to_parquet(features / table, index=False)

    pd.DataFrame(
        [
            {
                "x_id": "CHEMBL1",
                "x_type": "molecule",
                "y_id": "ENSG1",
                "y_type": "gene",
                "relation": "molecule_targets_gene",
                "display_relation": "targets",
                "source": "fixture",
                "credibility": 2,
                "action_type": "INHIBITOR",
            }
        ]
    ).to_parquet(edges / "molecule_targets_gene.parquet", index=False)
    pd.DataFrame(
        [
            {
                "edge_key": "molecule_targets_gene|CHEMBL1|ENSG1",
                "relation": "molecule_targets_gene",
                "x_id": "CHEMBL1",
                "x_type": "molecule",
                "y_id": "ENSG1",
                "y_type": "gene",
                "evidence_type": "database_record",
                "source": "fixture",
                "source_dataset": "unit_test",
                "source_record_id": "ev1",
                "paper_id": "PMID:1",
                "dataset_id": "DS1",
                "study_id": "S1",
                "evidence_score": 0.8,
                "effect_size": 0.2,
                "p_value": 0.05,
                "direction": "inhibits",
                "confidence_interval": None,
                "predicate": "INHIBITOR",
                "text_span": '{"action_type":"INHIBITOR"}',
                "section": "fixture",
                "extraction_method": "unit_test",
                "license": "fixture",
                "release": "test",
                "created_at": "2026-06-23T00:00:00Z",
            }
        ]
    ).to_parquet(evidence / "molecule_targets_gene.parquet", index=False)


def test_real_embedding_builder_outputs_staged_vectors_and_manifest(tmp_path: Path) -> None:
    kg_root = tmp_path / "kg"
    output_dir = tmp_path / "out"
    _write_fixture(kg_root)

    manifest = run(
        kg_root=kg_root,
        output_dir=output_dir,
        text_limit_per_table=2,
        edge_relations=["molecule_targets_gene"],
        edge_limit_per_relation=2,
        clean=True,
        test_deterministic_encoder=True,
    )

    assert manifest["task_id"] == TASK_ID
    assert manifest["run_id"] == RUN_ID
    assert manifest["staged_only"] is True
    assert manifest["canonical_promotion"] is False
    assert manifest["validation"]["passed"] is True
    assert manifest["models"]["text"]["embedding_dim"] == TEXT_MODEL_DIM
    assert manifest["models"]["edge"]["embedding_dim"] == EDGE_EMBEDDING_DIM
    assert any(item["modality"] == "protein_sequence_esm2" for item in manifest["blocked_modalities"])

    protein_info = manifest["outputs"]["node_text_embeddings"]["protein_textual_summary.parquet"]
    assert protein_info["status"] == "embedded"
    protein_path = Path(protein_info["output_path"])
    assert pq.ParquetFile(protein_path).metadata.num_rows == 1
    protein_df = pd.read_parquet(protein_path)
    assert protein_df.loc[0, "embedding_model"] == "pritamdeka/S-BioBERT-snli-multinli-stsb"
    assert len(protein_df.loc[0, "embedding"]) == TEXT_MODEL_DIM
    assert protein_df.loc[0, "source_feature_hash"]

    edge_info = manifest["outputs"]["edge_evidence_embeddings"]["molecule_targets_gene"]
    edge_path = Path(edge_info["output_path"])
    edge_df = pd.read_parquet(edge_path)
    assert pq.ParquetFile(edge_path).metadata.num_rows == 1
    assert edge_df.loc[0, "edge_key"] == "molecule_targets_gene|CHEMBL1|ENSG1"
    assert edge_df.loc[0, "pooling"] == "relation_value_evidence_mlp_weighted_mean"
    assert edge_df.loc[0, "n_evidence_rows"] == 1
    assert len(edge_df.loc[0, "embedding"]) == EDGE_EMBEDDING_DIM
    assert edge_df.loc[0, "payload_hash"]

    fallback_path = Path(manifest["outputs"]["learned_fallback_config"]["path"])
    fallback = json.loads(fallback_path.read_text())
    assert fallback["policy"].startswith("model-side learned fallback")
    assert "molecule_targets_gene" in fallback["edge_fallback"]["relations"]

    assert (output_dir / "manifest.json").exists()
    assert (output_dir / "real_embedding_summary.md").exists()


def test_gcs_local_cache_smoke_avoids_fuse_path(tmp_path: Path, monkeypatch) -> None:
    source_root = tmp_path / "source-kg"
    cache_root = tmp_path / "cache-kg"
    output_dir = tmp_path / "out-gcs-cache"
    _write_fixture(source_root)
    copied: list[tuple[str, str]] = []

    def fake_subprocess_run(args, check=False, stdout=None, stderr=None):
        assert args[:2] == ["gcloud", "storage"]
        uri = args[-1] if args[2] == "ls" else args[-2]
        rel = uri.removeprefix("gs://jouvencekb/main/")
        source = source_root / rel
        if args[2] == "ls":
            return subprocess.CompletedProcess(args, 0 if source.exists() else 1)
        assert args[2:5] == ["cp", "--quiet", uri]
        destination = Path(args[-1])
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        copied.append((uri, str(destination)))
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(builder.subprocess, "run", fake_subprocess_run)

    manifest = run(
        kg_root=tmp_path / "stale-fuse-not-used",
        output_dir=output_dir,
        text_limit_per_table=1,
        edge_relations=["molecule_targets_gene"],
        edge_limit_per_relation=1,
        clean=True,
        test_deterministic_encoder=True,
        gcs_kg_root="gs://jouvencekb/main",
        local_cache_dir=cache_root,
    )

    assert manifest["kg_root"] == str(cache_root)
    assert manifest["input_cache"]["source_gcs_root"] == "gs://jouvencekb/main"
    assert "--gcs-kg-root gs://jouvencekb/main" in manifest["recompute_command"]
    assert "--local-cache-dir" in manifest["recompute_command"]
    assert "stale-fuse-not-used" not in manifest["recompute_command"]
    assert any(uri.endswith("features/protein_textual_summary.parquet") for uri, _ in copied)
    assert manifest["validation"]["passed"] is True
    assert "does not depend on the macOS FUSE mount" in (output_dir / "real_embedding_summary.md").read_text()


def test_text_only_scaffold_smoke_skips_edge_inputs(tmp_path: Path) -> None:
    kg_root = tmp_path / "kg"
    output_dir = tmp_path / "out-text-only"
    _write_fixture(kg_root)

    manifest = run(
        kg_root=kg_root,
        output_dir=output_dir,
        text_limit_per_table=1,
        edge_relations=[],
        clean=True,
        test_deterministic_encoder=True,
    )

    assert manifest["outputs"]["edge_evidence_embeddings"] == {}
    assert "--skip-edge-embeddings" in manifest["recompute_command"]
    protein_path = Path(manifest["outputs"]["node_text_embeddings"]["protein_textual_summary.parquet"]["output_path"])
    assert protein_path.exists()
    assert pq.ParquetFile(protein_path).metadata.num_rows == 1
    assert manifest["validation"]["passed"] is True


def test_optional_gcs_miss_removes_stale_cached_file(tmp_path: Path, monkeypatch) -> None:
    destination = tmp_path / "cache" / "features" / "stale.parquet"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"stale")
    monkeypatch.setattr(builder, "gcloud_object_exists", lambda _uri: False)

    copied = builder.copy_gcs_object_if_exists(
        "gs://jouvencekb/main/features/stale.parquet",
        destination,
        required=False,
    )

    assert copied is False
    assert not destination.exists()
