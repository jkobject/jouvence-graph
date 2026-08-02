from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from manage_db.pyg_minibatch import EmbeddingSpec, iter_relation_minibatches


def _write(path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def _tiny_kg(tmp_path):
    root = tmp_path / "main"
    _write(root / "nodes" / "disease.parquet", [
        {"id": "D1", "name": "one"},
        {"id": "D2", "name": "two"},
        {"id": "D3", "name": "three"},
    ])
    _write(root / "nodes" / "gene.parquet", [
        {"id": "G1", "symbol": "A"},
        {"id": "G2", "symbol": "B"},
        {"id": "G3", "symbol": "C"},
    ])
    _write(root / "edges" / "disease_associated_gene.parquet", [
        {"relation": "disease_associated_gene", "x_id": "G1", "x_type": "gene", "y_id": "D1", "y_type": "disease", "credibility": 1, "score": 0.1},
        {"relation": "disease_associated_gene", "x_id": "G2", "x_type": "gene", "y_id": "D1", "y_type": "disease", "credibility": 2, "score": 0.2},
        {"relation": "disease_associated_gene", "x_id": "G3", "x_type": "gene", "y_id": "D2", "y_type": "disease", "credibility": 3, "score": 0.3},
    ])
    _write(root / "evidence" / "disease_associated_gene.parquet", [
        {
            "edge_key": "gene|G1|disease|D1|disease_associated_gene",
            "relation": "disease_associated_gene",
            "x_id": "G1",
            "x_type": "gene",
            "y_id": "D1",
            "y_type": "disease",
            "evidence_type": "curated",
            "source": "fixture",
        }
    ])
    _write(root / "embeddings" / "disease_embedding.parquet", [
        {"node_id": "D1", "embedding": [1.0, 0.0]},
        {"node_id": "D2", "embedding": [0.0, 1.0]},
        {"node_id": "D3", "embedding": [0.5, 0.5]},
    ])
    _write(root / "embeddings" / "gene_embedding.parquet", [
        {"node_id": "G1", "embedding": [1.0, 1.0]},
        {"node_id": "G2", "embedding": [2.0, 2.0]},
        {"node_id": "G3", "embedding": [3.0, 3.0]},
    ])
    return root


def test_streams_complete_relation_as_compact_featured_heterodata_batches(tmp_path) -> None:
    root = _tiny_kg(tmp_path)
    specs = {
        "disease": EmbeddingSpec(str(root / "embeddings" / "disease_embedding.parquet")),
        "gene": EmbeddingSpec(str(root / "embeddings" / "gene_embedding.parquet")),
    }

    batches = list(iter_relation_minibatches(
        str(root),
        "disease_associated_gene",
        batch_size=2,
        embeddings=specs,
        edge_feature_columns=("credibility", "score"),
    ))

    assert [batch.num_edges for batch in batches] == [2, 1]
    assert sum(batch.num_edges for batch in batches) == 3

    first = batches[0]
    edge_type = ("gene", "disease_associated_gene", "disease")
    assert first["disease"].node_id == ["D1"]
    assert first["gene"].node_id == ["G1", "G2"]
    assert first["disease"].x.tolist() == [[1.0, 0.0]]
    assert first["gene"].x.tolist() == [[1.0, 1.0], [2.0, 2.0]]
    assert first[edge_type].edge_index.tolist() == [[0, 1], [0, 0]]
    assert np.allclose(first[edge_type].edge_attr.numpy(), [[1.0, 0.1], [2.0, 0.2]])
    assert first[edge_type].edge_id == [
        "disease_associated_gene|G1|D1",
        "disease_associated_gene|G2|D1",
    ]
    assert first[edge_type].evidence_to_edge.tolist() == [0]
    assert first[edge_type].evidence_records[0]["evidence_type"] == "curated"


def test_rejects_missing_embedding_rows_in_strict_mode(tmp_path) -> None:
    root = _tiny_kg(tmp_path)
    _write(root / "embeddings" / "gene_incomplete.parquet", [
        {"node_id": "G1", "embedding": [1.0, 1.0]},
    ])

    with pytest.raises(ValueError, match="missing embeddings"):
        next(iter_relation_minibatches(
            str(root),
            "disease_associated_gene",
            batch_size=2,
            embeddings={
                "disease": EmbeddingSpec(str(root / "embeddings" / "disease_embedding.parquet")),
                "gene": EmbeddingSpec(str(root / "embeddings" / "gene_incomplete.parquet")),
            },
            strict_embeddings=True,
        ))
