from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from manage_db import kg_storage
from manage_db import build_pyg_export as build_mod
from manage_db.build_pyg_export import BuildConfig, build_pyg_export, main
from manage_db.kg_schema import GRAPH_DISCONNECTED_RELATIONS
from manage_db.run_pyg_gnn_smoke import (
    SmokeConfig,
    _build_training_message_passing_graph,
    _split_edges,
    main as smoke_main,
    run_smoke,
)


def _node_df(ids: list[str], **extra: list[str]) -> pd.DataFrame:
    data = {"id": ids}
    data.update(extra)
    return pd.DataFrame(data).convert_dtypes(dtype_backend="pyarrow")


def _edge_df(
    relation: str,
    x_type: str,
    y_type: str,
    pairs: list[tuple[str, str]],
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "x_id": [x for x, _ in pairs],
            "x_type": [x_type] * len(pairs),
            "y_id": [y for _, y in pairs],
            "y_type": [y_type] * len(pairs),
            "relation": [relation] * len(pairs),
            "display_relation": [relation.replace("_", " ")] * len(pairs),
            "source": ["unit-test"] * len(pairs),
            "credibility": [1] * len(pairs),
        }
    ).convert_dtypes(dtype_backend="pyarrow")


def _write_feature(root: kg_storage.KGRoot) -> None:
    root._ensure_dir("features")
    df = pd.DataFrame(
        {
            "node_id": ["ENSG000001", "ENSG000002"],
            "node_type": ["gene", "gene"],
            "feature_key": ["gene_textual_summary", "gene_textual_summary"],
            "summary": ["gene one summary", "gene two summary"],
        }
    ).convert_dtypes(dtype_backend="pyarrow")
    with root.fs.open(root._join("features", "gene_textual_summary.parquet"), "wb") as fh:
        df.to_parquet(fh, index=False)

    numeric_df = pd.DataFrame(
        {
            "node_id": ["EFO:0000001", "EFO:0000002"],
            "node_type": ["disease", "disease"],
            "feature_key": ["disease_prevalence_score", "disease_prevalence_score"],
            "prevalence_score": [0.25, 0.75],
        }
    ).convert_dtypes(dtype_backend="pyarrow")
    with root.fs.open(root._join("features", "disease_prevalence_score.parquet"), "wb") as fh:
        numeric_df.to_parquet(fh, index=False)

    categorical_df = pd.DataFrame(
        {
            "node_id": ["CHEMBL1", "CHEMBL2"],
            "node_type": ["molecule", "molecule"],
            "feature_key": ["molecule_development_phase", "molecule_development_phase"],
            "development_phase": ["approved", "investigational"],
        }
    ).convert_dtypes(dtype_backend="pyarrow")
    with root.fs.open(root._join("features", "molecule_development_phase.parquet"), "wb") as fh:
        categorical_df.to_parquet(fh, index=False)


def _build_tiny_kg(tmp_path: Path) -> Path:
    kg_path = tmp_path / "kg"
    root = kg_storage.open_kg_root(str(kg_path))
    kg_storage.write_nodes(
        root,
        "gene",
        _node_df(
            ["ENSG000001", "ENSG000002", "ENSG000003"],
            ncbi_gene_id=["1", "2", "3"],
            hgnc_id=["HGNC:1", "HGNC:2", "HGNC:3"],
            uniprot_id=["P1", "P2", "P3"],
            gene_name=["G1", "G2", "G3"],
        ),
    )
    kg_storage.write_nodes(
        root,
        "disease",
        _node_df(
            ["EFO:0000001", "EFO:0000002"],
            mondo_id=["MONDO:1", "MONDO:2"],
            omim_id=["OMIM:1", "OMIM:2"],
            doid_id=["DOID:1", "DOID:2"],
            icd10_code=["A", "B"],
            mesh_id=["M1", "M2"],
            hp_id=["HP:1", "HP:2"],
        ),
    )
    kg_storage.write_nodes(
        root,
        "molecule",
        _node_df(
            ["CHEMBL1", "CHEMBL2"],
            drugbank_id=["DB1", "DB2"],
            pubchem_cid=["11", "22"],
            cas_rn=["CAS1", "CAS2"],
            inchikey=["I1", "I2"],
            smiles=["C", "CC"],
        ),
    )
    kg_storage.write_nodes(root, "dataset", _node_df(["dataset:unit"]))
    kg_storage.write_nodes(
        root,
        "paper",
        _node_df(["PMID:1"], doi=["10.0000/unit"], pmc_id=["PMC1"], arxiv_id=[""]),
    )
    kg_storage.write_nodes(
        root,
        "cell_line",
        _node_df(["CVCL_0001"], ccle_name=["UNIT"], cosmic_id=["1"], efo_id=["EFO:cell-line"]),
    )
    kg_storage.write_edges(
        root,
        "dataset_contains_cell_line",
        _edge_df("dataset_contains_cell_line", "dataset", "cell_line", [("dataset:unit", "CVCL_0001")]),
    )
    kg_storage.write_nodes(
        root,
        "tissue",
        _node_df(["UBERON:0001"], bto_id=["BTO:0001"], mesh_id=["MESH:T1"], fma_id=["FMA:1"]),
    )
    kg_storage.write_edges(
        root,
        "dataset_contains_tissue",
        _edge_df("dataset_contains_tissue", "dataset", "tissue", [("dataset:unit", "UBERON:0001")]),
    )
    kg_storage.write_edges(
        root,
        "disease_associated_gene",
        _edge_df(
            "disease_associated_gene",
            "gene",
            "disease",
            [("ENSG000001", "EFO:0000001"), ("ENSG000002", "EFO:0000002")],
        ),
    )
    kg_storage.write_edges(
        root,
        "molecule_targets_gene",
        _edge_df(
            "molecule_targets_gene",
            "molecule",
            "gene",
            [("CHEMBL1", "ENSG000001"), ("CHEMBL2", "ENSG000003")],
        ),
    )
    _write_feature(root)
    return kg_path


def test_build_pyg_export_writes_bounded_artifacts(tmp_path: Path) -> None:
    kg_path = _build_tiny_kg(tmp_path)
    output_path = tmp_path / "pyg"

    result = build_pyg_export(
        BuildConfig(
            kg_root=str(kg_path),
            output_root=str(output_path),
            node_types=("gene", "disease", "molecule"),
            relations=("disease_associated_gene", "molecule_targets_gene"),
            max_nodes_per_type=2,
            max_edges_per_relation=10,
            feature_tables=("gene_textual_summary",),
            strict=False,
            build_name="unit-test",
        )
    )

    assert result.node_counts == {"gene": 2, "disease": 2, "molecule": 2}
    assert result.edge_counts["disease_associated_gene"] == 2
    assert result.edge_counts["molecule_targets_gene"] == 1  # capped gene map drops ENSG000003 in non-strict mode

    gene_map = pq.read_table(output_path / "node_maps" / "gene.id_to_index.parquet").to_pandas()
    assert gene_map.to_dict(orient="records") == [
        {"id": "ENSG000001", "node_type": "gene", "node_index": 0},
        {"id": "ENSG000002", "node_type": "gene", "node_index": 1},
    ]

    edge_index = np.load(output_path / "edges" / "gene__disease_associated_gene__disease" / "edge_index.npy")
    assert edge_index.shape == (2, 2)
    assert edge_index.tolist() == [[0, 1], [0, 1]]

    relation_map = pq.read_table(output_path / "schema" / "relation_to_edge_type.parquet").to_pandas()
    assert set(relation_map["relation"]) == {"disease_associated_gene", "molecule_targets_gene"}
    assert tuple(relation_map.loc[relation_map["relation"] == "molecule_targets_gene", ["x_type", "y_type"]].iloc[0]) == ("molecule", "gene")

    feature_manifest = json.loads(
        (output_path / "node_features" / "gene" / "gene_textual_summary.feature_manifest.json").read_text()
    )
    assert feature_manifest["mapped_rows"] == 2
    assert feature_manifest["tensor_policy"].startswith("raw text")

    validation = json.loads((output_path / "validation_report.json").read_text())
    assert validation["status"] == "fail"  # non-strict records dropped endpoint as validation issue
    assert validation["checks"]["reproducibility_hashes_present"] is True
    assert any(issue["check"] == "y_endpoint_in_node_map" for issue in validation["issues"])

    manifest = json.loads((output_path / "manifest.json").read_text())
    assert manifest["artifact_format"] == "jouvencekb-kg-pyg-v1"
    assert manifest["tensor_formats"]["edge_index.npy"].startswith("always written")
    assert (output_path / "heterodata" / "full_graph.metadata.json").exists()
    assert (output_path / "heterodata" / "full_graph.pt").exists()

    with (output_path / "heterodata" / "full_graph.pt").open("rb") as fh:
        heterodata = pickle.load(fh)
    assert heterodata["gene"].x.shape == (2, 256)
    assert heterodata["disease"].x.shape == (2, 256)
    assert not np.allclose(heterodata["gene"].x.numpy(), np.ones((2, 256)))
    assert not np.allclose(heterodata["disease"].x.numpy(), np.ones((2, 256)))
    assert heterodata["gene", "disease_associated_gene", "disease"].edge_index.tolist() == [[0, 1], [0, 1]]
    assert heterodata["disease", "rev_disease_associated_gene", "gene"].edge_index.tolist() == [[0, 1], [0, 1]]


def test_pyg_export_excludes_dataset_and_paper_from_training_graph_by_default(tmp_path: Path) -> None:
    kg_path = _build_tiny_kg(tmp_path)
    output_path = tmp_path / "pyg-provenance-excluded"

    result = build_pyg_export(
        BuildConfig(
            kg_root=str(kg_path),
            output_root=str(output_path),
            node_types=("gene", "disease", "dataset", "paper", "cell_line", "tissue"),
            relations=("disease_associated_gene", "dataset_contains_cell_line", "dataset_contains_tissue"),
            max_nodes_per_type=None,
            max_edges_per_relation=10,
            strict=True,
            build_name="unit-provenance-excluded",
        )
    )

    assert result.node_counts == {"gene": 3, "disease": 2, "cell_line": 1, "tissue": 1}
    assert result.edge_counts == {"disease_associated_gene": 2}
    manifest = json.loads((output_path / "manifest.json").read_text())
    assert manifest["node_types"] == ["gene", "disease", "cell_line", "tissue"]
    assert manifest["relations"] == ["disease_associated_gene"]
    policy = manifest["training_graph_exclusion_policy"]
    assert policy["excluded_node_types_by_default"] == ["dataset", "paper"]
    assert policy["excluded_requested_node_types"] == ["dataset", "paper"]
    assert policy["excluded_requested_relations"] == ["dataset_contains_cell_line", "dataset_contains_tissue"]
    assert policy["graph_disconnected_relations_by_default"] == sorted(GRAPH_DISCONNECTED_RELATIONS)
    assert policy["include_provenance_node_types"] is False
    assert policy["policy_label"] == "metadata-only/non-training; retain canonical files with exporter exclusions unless explicitly audited"
    assert policy["rationale"] == "paper and dataset entities are provenance/catalog metadata only, not message-passing graph nodes"
    assert not (output_path / "node_maps" / "dataset.id_to_index.parquet").exists()
    assert not (output_path / "edges" / "dataset__dataset_contains_cell_line__cell_line").exists()
    assert not (output_path / "edges" / "dataset__dataset_contains_tissue__tissue").exists()


def test_pyg_export_can_opt_in_to_provenance_node_types_for_audit(tmp_path: Path) -> None:
    kg_path = _build_tiny_kg(tmp_path)
    output_path = tmp_path / "pyg-provenance-included"

    result = build_pyg_export(
        BuildConfig(
            kg_root=str(kg_path),
            output_root=str(output_path),
            node_types=("dataset", "cell_line"),
            relations=("dataset_contains_cell_line",),
            max_nodes_per_type=None,
            max_edges_per_relation=10,
            strict=True,
            build_name="unit-provenance-included",
            include_provenance_node_types=True,
        )
    )

    assert result.node_counts == {"dataset": 1, "cell_line": 1}
    assert result.edge_counts == {"dataset_contains_cell_line": 1}
    manifest = json.loads((output_path / "manifest.json").read_text())
    assert manifest["training_graph_exclusion_policy"]["include_provenance_node_types"] is True
    assert manifest["training_graph_exclusion_policy"]["excluded_requested_node_types"] == []
    assert manifest["training_graph_exclusion_policy"]["excluded_requested_relations"] == []


def test_run_pyg_gnn_smoke_trains_on_exported_heterodata(tmp_path: Path) -> None:
    kg_path = _build_tiny_kg(tmp_path)
    _extend_disease_edges_for_smoke(kg_path)
    output_path = tmp_path / "pyg-smoke"
    build_pyg_export(
        BuildConfig(
            kg_root=str(kg_path),
            output_root=str(output_path),
            node_types=("gene", "disease", "molecule"),
            relations=("disease_associated_gene", "molecule_targets_gene"),
            max_nodes_per_type=None,
            max_edges_per_relation=10,
            strict=True,
            build_name="unit-smoke",
        )
    )

    result = run_smoke(
        SmokeConfig(
            export_root=output_path,
            relation="disease_associated_gene",
            epochs=2,
            hidden_channels=4,
            max_train_edges=2,
            output_json=tmp_path / "smoke_metrics.json",
        )
    )

    assert result.status == "pass"
    assert result.validation["checks"]["feature_tensors_present"] is True
    assert result.validation["checks"]["edge_tensors_present"] is True
    assert result.validation["checks"]["reverse_edges_are_transpose"] is True
    assert result.split_counts == {
        "train_positive_edges": 2,
        "train_negative_edges": 2,
        "valid_positive_edges": 1,
        "valid_negative_edges": 1,
        "test_positive_edges": 1,
        "test_negative_edges": 1,
    }
    assert result.metrics["epochs"] == 2.0
    assert result.config["relation_role"] == "primary_link_prediction_relation_with_other_relations_as_auxiliary_message_passing_context"
    assert result.graph_sizes["primary_relation"] == "disease_associated_gene"
    assert result.graph_sizes["primary_message_passing_edge_count"] == result.split_counts["train_positive_edges"]
    assert result.graph_sizes["auxiliary_forward_relations"] == ["molecule_targets_gene"]
    assert "test_loss" in result.metrics
    assert "test_accuracy" in result.metrics
    assert result.validation["checks"]["heldout_edges_removed_from_message_passing"] is True
    assert (tmp_path / "smoke_metrics.json").exists()


def test_no_cap_export_defaults_to_sidecar_without_full_heterodata_pickle(tmp_path: Path, monkeypatch) -> None:
    kg_path = _build_tiny_kg(tmp_path)
    output_path = tmp_path / "pyg-sidecar"

    def _forbidden_heterodata_writer(*args, **kwargs):
        raise AssertionError("sidecar production mode must not build a full HeteroData pickle")

    monkeypatch.setattr(build_mod, "_write_heterodata_artifact", _forbidden_heterodata_writer)
    result = build_pyg_export(
        BuildConfig(
            kg_root=str(kg_path),
            output_root=str(output_path),
            node_types=("gene", "disease", "molecule"),
            relations=("disease_associated_gene", "molecule_targets_gene"),
            max_nodes_per_type=None,
            max_edges_per_relation=None,
            strict=True,
            build_name="unit-sidecar",
        )
    )

    assert result.edge_counts == {"disease_associated_gene": 2, "molecule_targets_gene": 2}
    manifest = json.loads((output_path / "manifest.json").read_text())
    assert manifest["artifact_mode"] == "sidecar"
    assert manifest["bounded"] is False
    validation = json.loads((output_path / "validation_report.json").read_text())
    assert validation["checks"]["heterodata_pickle_required"] is False
    assert (output_path / "sidecar_artifact.metadata.json").exists()
    assert not (output_path / "heterodata" / "full_graph.pt").exists()


def test_run_pyg_gnn_smoke_loads_selected_relation_from_sidecars(tmp_path: Path) -> None:
    kg_path = _build_tiny_kg(tmp_path)
    _extend_disease_edges_for_smoke(kg_path)
    output_path = tmp_path / "pyg-sidecar-smoke"
    build_pyg_export(
        BuildConfig(
            kg_root=str(kg_path),
            output_root=str(output_path),
            node_types=("gene", "disease", "molecule"),
            relations=("disease_associated_gene", "molecule_targets_gene"),
            max_nodes_per_type=None,
            max_edges_per_relation=None,
            strict=True,
            build_name="unit-sidecar-smoke",
            artifact_mode="sidecar",
        )
    )

    result = run_smoke(
        SmokeConfig(
            export_root=output_path,
            relation="disease_associated_gene",
            epochs=1,
            hidden_channels=4,
            max_train_edges=1,
        )
    )
    assert result.status == "pass"
    assert result.edge_type == ("gene", "disease_associated_gene", "disease")
    assert result.validation["checks"]["edge_tensors_present"] is True


def test_build_pyg_export_cli(tmp_path: Path, capsys) -> None:
    kg_path = _build_tiny_kg(tmp_path)
    output_path = tmp_path / "pyg-cli"

    rc = main(
        [
            "--kg-root",
            str(kg_path),
            "--output-root",
            str(output_path),
            "--node-types",
            "gene",
            "disease",
            "--relations",
            "disease_associated_gene",
            "--max-nodes-per-type",
            "10",
            "--max-edges-per-relation",
            "10",
            "--build-name",
            "cli-test",
        ]
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["edge_counts"] == {"disease_associated_gene": 2}
    assert (output_path / "README.md").exists()


def test_build_pyg_export_plan_only_writes_metadata_manifest(tmp_path: Path, capsys, monkeypatch) -> None:
    kg_path = _build_tiny_kg(tmp_path)
    output_path = tmp_path / "pyg-plan"

    def _forbidden_table_materialization(*args, **kwargs):
        raise AssertionError("plan-only manifest must use Parquet footer metadata, not materialize node/edge tables")

    def _forbidden_tensor_writer(*args, **kwargs):
        raise AssertionError("plan-only manifest must not write edge tensors or row-map tensors")

    monkeypatch.setattr(build_mod, "_read_parquet_limited", _forbidden_table_materialization)
    monkeypatch.setattr(build_mod, "_write_tensor", _forbidden_tensor_writer)

    rc = main(
        [
            "--kg-root",
            str(kg_path),
            "--output-root",
            str(output_path),
            "--node-types",
            "gene",
            "disease",
            "dataset",
            "--relations",
            "disease_associated_gene",
            "dataset_contains_cell_line",
            "--max-nodes-per-type",
            "0",
            "--max-edges-per-relation",
            "0",
            "--build-name",
            "cli-plan-test",
            "--plan-only",
            "--remote-output-root",
            "gs://jouvencekb/staging/ml/pyg/cli-plan-test",
        ]
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["node_counts"] == {"gene": 3, "disease": 2}
    assert payload["edge_counts"] == {"disease_associated_gene": 2}
    manifest = json.loads((output_path / "production_plan_manifest.json").read_text())
    assert manifest["artifact_format"] == "jouvencekb-kg-pyg-production-plan-v1"
    assert manifest["metadata_source"].startswith("Parquet footer metadata only")
    assert manifest["total_edge_rows"] == 2
    assert "--output-root gs://jouvencekb/staging/ml/pyg/cli-plan-test" in manifest["remote_command"]
    assert "--artifact-mode sidecar" in manifest["remote_command"]
    assert manifest["artifact_mode"] == "sidecar"
    policy = manifest["training_graph_exclusion_policy"]
    assert policy["excluded_requested_node_types"] == ["dataset"]
    assert policy["excluded_requested_relations"] == ["dataset_contains_cell_line"]
    validation = json.loads((output_path / "validation_report.json").read_text())
    assert validation["checks"]["metadata_only_no_table_scan"] is True
    assert not (output_path / "heterodata").exists()


def _write_unit_embeddings(root: kg_storage.KGRoot) -> Path:
    features = Path(root.uri) / "features"
    node_dir = features / "embeddings" / "text" / "gene" / "unit-model" / "policy_v1"
    node_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "node_id": ["ENSG000001", "ENSG000002"],
            "node_type": ["gene", "gene"],
            "embedding_dim": [4, 4],
            "embedding_dtype": ["float32", "float32"],
            "embedding_format": ["list_float32", "list_float32"],
            "embedding": [[0.10, 0.20, 0.30, 0.40], [0.50, 0.60, 0.70, 0.80]],
        }
    ).to_parquet(node_dir / "part-000.parquet", index=False)

    edge_dir = features / "edge_embeddings" / "by_relation" / "disease_associated_gene" / "unit-encoder" / "edge_policy_v1"
    edge_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "edge_key": [
                "disease_associated_gene|ENSG000001|EFO:0000001",
                "disease_associated_gene|ENSG000002|EFO:0000002",
            ],
            "x_id": ["ENSG000001", "ENSG000002"],
            "x_type": ["gene", "gene"],
            "y_id": ["EFO:0000001", "EFO:0000002"],
            "y_type": ["disease", "disease"],
            "relation": ["disease_associated_gene", "disease_associated_gene"],
            "embedding_dim": [3, 3],
            "embedding_dtype": ["float32", "float32"],
            "embedding_format": ["list_float32", "list_float32"],
            "embedding": [[1.0, 0.0, 0.5], [0.0, 1.0, 0.25]],
        }
    ).to_parquet(edge_dir / "part-000.parquet", index=False)

    fallback_config = {
        "node_fallback": {"module": "torch.nn.Embedding", "dim": 4},
        "edge_fallback": {"module": "torch.nn.Embedding", "dim": 3},
        "policy": "model-side learned fallback only",
    }
    config_path = features / "embeddings" / "reports" / "learned_fallback_config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(fallback_config, indent=2) + "\n")
    return features


def test_pyg_export_wires_real_embeddings_and_learned_fallbacks(tmp_path: Path) -> None:
    kg_path = _build_tiny_kg(tmp_path)
    root = kg_storage.open_kg_root(str(kg_path))
    embedding_root = _write_unit_embeddings(root)
    output_path = tmp_path / "pyg-embeddings"

    build_pyg_export(
        BuildConfig(
            kg_root=str(kg_path),
            output_root=str(output_path),
            node_types=("gene", "disease", "molecule"),
            relations=("disease_associated_gene", "molecule_targets_gene"),
            max_nodes_per_type=None,
            max_edges_per_relation=10,
            feature_tables=("gene_textual_summary", "disease_prevalence_score", "molecule_development_phase"),
            strict=True,
            build_name="unit-embeddings",
            embedding_features_root=str(embedding_root),
            learned_fallback_config_path=str(embedding_root / "embeddings" / "reports" / "learned_fallback_config.json"),
        )
    )

    with (output_path / "heterodata" / "full_graph.pt").open("rb") as fh:
        heterodata = pickle.load(fh)

    assert heterodata["gene"].x.shape == (3, 4)
    assert np.allclose(heterodata["gene"].x[:2].numpy(), [[0.10, 0.20, 0.30, 0.40], [0.50, 0.60, 0.70, 0.80]])
    assert heterodata["gene"].x[2].shape == (4,)
    assert not np.allclose(heterodata["gene"].x[2].numpy(), np.ones(4))
    assert heterodata["disease"].x.shape == (2, 4)
    assert not np.allclose(heterodata["disease"].x.numpy(), np.ones((2, 4)))

    edge_type = ("gene", "disease_associated_gene", "disease")
    assert heterodata[edge_type].edge_attr.shape == (2, 3)
    assert heterodata[edge_type].edge_attr.tolist() == [[1.0, 0.0, 0.5], [0.0, 1.0, 0.25]]
    fallback_edge_type = ("molecule", "molecule_targets_gene", "gene")
    assert heterodata[fallback_edge_type].edge_attr.shape == (2, 3)
    assert not np.allclose(heterodata[fallback_edge_type].edge_attr.numpy(), np.ones((2, 3)))

    metadata = json.loads((output_path / "heterodata" / "full_graph.metadata.json").read_text())
    assert metadata["node_embedding_policy"]["gene"]["real_rows"] == 2
    assert metadata["node_embedding_policy"]["disease"]["fallback_rows"] == 2
    assert metadata["edge_embedding_policy"][str(edge_type)]["real_rows"] == 2
    assert metadata["edge_embedding_policy"][str(fallback_edge_type)]["fallback_rows"] == 2

    manifest = json.loads((output_path / "manifest.json").read_text())
    assert manifest["node_embeddings"]["gene"][0]["embedding_model"] == "unit-model"
    assert manifest["node_embeddings"]["gene"][0]["embedding_dim"] == 4
    assert manifest["node_embeddings"]["gene"][0]["embedding_dtype"] == "float32"
    assert manifest["node_embeddings"]["gene"][0]["sidecar_node_id_mapping"] == {
        "embedding_join_key": "node_id",
        "node_map_join_key": "id",
        "pyg_index_column": "node_index",
    }
    assert manifest["missing_feature_policy"]["node_types"]["gene"]["embedding_status"] == "available"
    assert manifest["missing_feature_policy"]["node_types"]["disease"]["embedding_status"] == "absent"
    assert manifest["missing_feature_policy"]["node_types"]["disease"]["absent_embeddings"][0]["fallback"] == "model-side learned torch.nn.Embedding rows"
    assert manifest["missing_feature_policy"]["node_types"]["molecule"]["intentionally_deferred"] == [
        {"field": "chemical_encoder_embedding", "reason": "learned chemical encoder production build pending", "status": "deferred"}
    ]
    assert manifest["missing_feature_policy"]["node_types"]["gene"]["available_feature_values"][0]["value_kind"] == "text/raw"
    assert manifest["missing_feature_policy"]["node_types"]["disease"]["available_feature_values"] == [
        {
            "feature_table": "disease_prevalence_score",
            "mapped_rows": 2,
            "node_count": 2,
            "row_map": "node_features/disease/disease_prevalence_score.row_map.parquet",
            "source_path": f"{kg_path}/features/disease_prevalence_score.parquet",
            "status": "available",
            "value_kind": "numeric",
        }
    ]
    assert manifest["missing_feature_policy"]["node_types"]["molecule"]["available_feature_values"] == [
        {
            "feature_table": "molecule_development_phase",
            "mapped_rows": 2,
            "node_count": 2,
            "row_map": "node_features/molecule/molecule_development_phase.row_map.parquet",
            "source_path": f"{kg_path}/features/molecule_development_phase.parquet",
            "status": "available",
            "value_kind": "categorical",
        }
    ]
    assert manifest["edge_embeddings"]["disease_associated_gene"][0]["embedding_model"] == "unit-encoder"
    assert manifest["edge_embeddings"]["disease_associated_gene"][0]["embedding_dim"] == 3
    assert manifest["edge_embeddings"]["disease_associated_gene"][0]["embedding_dtype"] == "float32"
    assert manifest["edge_embeddings"]["disease_associated_gene"][0]["sidecar_edge_id_mapping"] == {
        "embedding_join_key": "edge_key",
        "edge_row_map_join_key": "edge_key",
        "pyg_index_column": "edge_pos",
    }
    assert manifest["missing_feature_policy"]["edge_types"]["molecule_targets_gene"]["embedding_status"] == "absent"
    assert manifest["missing_feature_policy"]["edge_types"]["molecule_targets_gene"]["available_feature_values"] == [
        {"field": "credibility", "source": "edges/{relation}.parquet and edge_attr.parquet", "status": "available", "value_kind": "numeric"},
        {"field": "source", "source": "edge_row_map.parquet", "status": "available", "value_kind": "categorical"},
    ]
    assert manifest["missing_feature_policy"]["edge_types"]["molecule_targets_gene"]["absent_embeddings"] == [
        {
            "embedding_family": "relation_value_evidence_mlp",
            "fallback": "model-side learned torch.nn.Embedding rows",
            "reason": "no manifest-visible edge embedding sidecar found for selected relation",
            "status": "absent",
        }
    ]
    assert "full 100M-edge tensor materialization" in manifest["missing_feature_policy"]["materialization_policy"]

    # Keep the original two-edge fixture for the exact embedding assertions
    # above, then rebuild a split-capable graph for the leak-free smoke.
    _extend_disease_edges_for_smoke(kg_path)
    build_pyg_export(
        BuildConfig(
            kg_root=str(kg_path),
            output_root=str(output_path),
            node_types=("gene", "disease", "molecule"),
            relations=("disease_associated_gene", "molecule_targets_gene"),
            max_nodes_per_type=None,
            max_edges_per_relation=10,
            feature_tables=("gene_textual_summary", "disease_prevalence_score", "molecule_development_phase"),
            strict=True,
            build_name="unit-embeddings-smoke",
            embedding_features_root=str(embedding_root),
            learned_fallback_config_path=str(embedding_root / "embeddings" / "reports" / "learned_fallback_config.json"),
        )
    )
    smoke = run_smoke(
        SmokeConfig(
            export_root=output_path,
            relation="disease_associated_gene",
            epochs=1,
            hidden_channels=4,
            max_train_edges=1,
        )
    )
    assert smoke.status == "pass"
    assert smoke.validation["checks"]["edge_attr_tensors_present"] is True
    assert smoke.validation["checks"]["selected_edge_attr_consumed_by_predictor"] is True
    assert smoke.validation["edge_attr_usage"]["all_edge_types_have_edge_attr"] is True
    assert smoke.graph_sizes["forward_edge_counts"] == {"disease_associated_gene": 4, "molecule_targets_gene": 2}


def test_pyg_gnn_smoke_split_includes_disjoint_heldout_test_edges() -> None:
    import torch

    edge_index = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5],
            [10, 11, 12, 13, 14, 15],
        ],
        dtype=torch.long,
    )

    train_pos, valid_pos, test_pos, train_idx, valid_idx, test_idx = _split_edges(
        edge_index, max_train_edges=2, seed=7423
    )

    assert train_pos.size(1) == 2
    assert valid_pos.size(1) == 1
    assert test_pos.size(1) == 1
    assert len(set(train_idx.tolist()) & set(valid_idx.tolist())) == 0
    assert len(set(train_idx.tolist()) & set(test_idx.tolist())) == 0
    assert len(set(valid_idx.tolist()) & set(test_idx.tolist())) == 0


def test_pyg_gnn_smoke_message_passing_graph_excludes_valid_and_test_labels() -> None:
    import torch
    from torch_geometric.data import HeteroData

    data = HeteroData()
    data["gene"].num_nodes = 6
    data["gene"].x = torch.randn(6, 4)
    data["disease"].num_nodes = 6
    data["disease"].x = torch.randn(6, 4)
    edge_type = ("gene", "disease_associated_gene", "disease")
    reverse_edge_type = ("disease", "rev_disease_associated_gene", "gene")
    data[edge_type].edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    data[edge_type].edge_attr = torch.arange(12, dtype=torch.float32).view(4, 3)
    data[reverse_edge_type].edge_index = data[edge_type].edge_index[[1, 0], :]
    data[reverse_edge_type].edge_attr = data[edge_type].edge_attr.clone()

    train_idx = torch.tensor([0, 2], dtype=torch.long)
    valid_idx = torch.tensor([1], dtype=torch.long)
    test_idx = torch.tensor([3], dtype=torch.long)

    mp_data = _build_training_message_passing_graph(
        data, edge_type, reverse_edge_type, train_idx, valid_idx, test_idx
    )

    assert mp_data[edge_type].edge_index.tolist() == [[0, 2], [1, 3]]
    assert mp_data[reverse_edge_type].edge_index.tolist() == [[1, 3], [0, 2]]
    assert mp_data[edge_type].edge_attr.tolist() == [[0.0, 1.0, 2.0], [6.0, 7.0, 8.0]]
    assert [1, 2] not in mp_data[edge_type].edge_index.t().tolist()
    assert [3, 4] not in mp_data[edge_type].edge_index.t().tolist()
    assert [2, 1] not in mp_data[reverse_edge_type].edge_index.t().tolist()
    assert [4, 3] not in mp_data[reverse_edge_type].edge_index.t().tolist()


def _extend_disease_edges_for_smoke(kg_path: Path) -> None:
    root = kg_storage.open_kg_root(str(kg_path))
    kg_storage.write_edges(
        root,
        "disease_associated_gene",
        _edge_df(
            "disease_associated_gene",
            "gene",
            "disease",
            [
                ("ENSG000001", "EFO:0000001"),
                ("ENSG000002", "EFO:0000002"),
                ("ENSG000003", "EFO:0000001"),
                ("ENSG000001", "EFO:0000002"),
            ],
        ),
    )


def test_run_pyg_gnn_smoke_cli_writes_leak_free_metrics(tmp_path: Path, capsys) -> None:
    kg_path = _build_tiny_kg(tmp_path)
    _extend_disease_edges_for_smoke(kg_path)
    output_path = tmp_path / "pyg-smoke-cli"
    metrics_path = tmp_path / "smoke-cli-metrics.json"
    build_pyg_export(
        BuildConfig(
            kg_root=str(kg_path),
            output_root=str(output_path),
            node_types=("gene", "disease", "molecule"),
            relations=("disease_associated_gene", "molecule_targets_gene"),
            max_nodes_per_type=None,
            max_edges_per_relation=10,
            strict=True,
            build_name="unit-smoke-cli",
        )
    )

    rc = smoke_main(
        [
            "--export-root",
            str(output_path),
            "--relation",
            "disease_associated_gene",
            "--epochs",
            "1",
            "--hidden-channels",
            "4",
            "--max-train-edges",
            "2",
            "--seed",
            "7423",
            "--output-json",
            str(metrics_path),
        ]
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    persisted = json.loads(metrics_path.read_text())
    for result in (payload, persisted):
        assert result["status"] == "pass"
        assert result["split_counts"] == {
            "train_positive_edges": 2,
            "train_negative_edges": 2,
            "valid_positive_edges": 1,
            "valid_negative_edges": 1,
            "test_positive_edges": 1,
            "test_negative_edges": 1,
        }
        assert result["validation"]["checks"]["heldout_edges_removed_from_message_passing"] is True
        assert result["validation"]["checks"]["message_passing_edge_count_matches_train_split"] is True
        assert result["validation"]["checks"]["selected_edge_attr_consumed_by_predictor"] is True
        assert result["validation"]["edge_attr_usage"]["selected_edge_attr_consumed_by_predictor"] is True
        assert result["graph_sizes"]["primary_relation"] == "disease_associated_gene"
        assert result["graph_sizes"]["primary_message_passing_edge_count"] == 2
        assert result["graph_sizes"]["auxiliary_forward_relations"] == ["molecule_targets_gene"]
        assert result["config"]["seed"] == 7423
        assert result["metrics"]["train_loss_trace"]


def _extend_molecule_target_edges_for_smoke(kg_path: Path) -> None:
    root = kg_storage.open_kg_root(str(kg_path))
    kg_storage.write_nodes(
        root,
        "molecule",
        _node_df(
            ["CHEMBL1", "CHEMBL2", "CHEMBL3", "CHEMBL4"],
            drugbank_id=["DB1", "DB2", "DB3", "DB4"],
            pubchem_cid=["11", "22", "33", "44"],
            cas_rn=["CAS1", "CAS2", "CAS3", "CAS4"],
            inchikey=["I1", "I2", "I3", "I4"],
            smiles=["C", "CC", "CCC", "CCCC"],
        ),
    )
    kg_storage.write_edges(
        root,
        "molecule_targets_gene",
        _edge_df(
            "molecule_targets_gene",
            "molecule",
            "gene",
            [
                ("CHEMBL1", "ENSG000001"),
                ("CHEMBL2", "ENSG000002"),
                ("CHEMBL3", "ENSG000003"),
                ("CHEMBL4", "ENSG000001"),
            ],
        ),
    )


def test_run_pyg_gnn_smoke_evaluates_multiple_primary_relations(tmp_path: Path) -> None:
    kg_path = _build_tiny_kg(tmp_path)
    _extend_disease_edges_for_smoke(kg_path)
    _extend_molecule_target_edges_for_smoke(kg_path)
    output_path = tmp_path / "pyg-multitarget-smoke"
    build_pyg_export(
        BuildConfig(
            kg_root=str(kg_path),
            output_root=str(output_path),
            node_types=("gene", "disease", "molecule"),
            relations=("disease_associated_gene", "molecule_targets_gene"),
            max_nodes_per_type=None,
            max_edges_per_relation=10,
            strict=True,
            build_name="unit-multitarget-smoke",
        )
    )

    result = run_smoke(
        SmokeConfig(
            export_root=output_path,
            primary_relations=("disease_associated_gene", "molecule_targets_gene"),
            epochs=1,
            hidden_channels=4,
            max_train_edges=2,
            output_json=tmp_path / "multitarget_metrics.json",
        )
    )

    assert result.status == "pass"
    assert result.config["primary_relations"] == ["disease_associated_gene", "molecule_targets_gene"]
    assert set(result.primary_results) == {"disease_associated_gene", "molecule_targets_gene"}
    for relation, payload in result.primary_results.items():
        assert payload["relation"] == relation
        assert payload["validation"]["checks"]["heldout_edges_removed_from_message_passing"] is True
        assert payload["validation"]["checks"]["message_passing_edge_count_matches_train_split"] is True
        assert payload["validation"]["checks"]["selected_edge_attr_consumed_by_predictor"] is True
        assert payload["metrics"]["train_loss_trace"]
    assert set(result.metrics["primary_relation_metrics"]) == {"disease_associated_gene", "molecule_targets_gene"}
    assert (tmp_path / "multitarget_metrics.json").exists()
