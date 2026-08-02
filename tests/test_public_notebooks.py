from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from manage_db import public_notebooks as public


def test_read_bounded_parquet_seeks_row_groups(tmp_path: Path) -> None:
    path = tmp_path / "rows.parquet"
    pq.write_table(pa.table({"id": list(range(12)), "value": [f"v{i}" for i in range(12)]}), path, row_group_size=3)

    result = public.read_bounded_parquet(path, columns=["id"], offset=4, limit=5)

    assert result["id"].tolist() == [4, 5, 6, 7, 8]
    with pytest.raises(ValueError, match="between 1"):
        public.read_bounded_parquet(path, limit=public.MAX_SAMPLE_ROWS + 1)


def test_requester_pays_requires_caller_project(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("JOUVENCE_BILLING_PROJECT", raising=False)
    with pytest.raises(ValueError, match="JOUVENCE_BILLING_PROJECT"):
        public._storage_options("gs://jouvencekb/main/nodes/gene.parquet", None)
    assert public._storage_options("gs://bucket/object", "consumer-project") == {
        "requester_pays": "consumer-project",
        "token": "google_default",
    }

    monkeypatch.setattr(
        "gcsfs.credentials.gauth.default",
        lambda **_kwargs: (object(), "adc-default-project"),
    )
    fs, path = public.url_to_fs(
        "gs://bucket/object",
        **public._storage_options("gs://bucket/object", "consumer-project"),
        skip_instance_cache=True,
    )
    assert path == "bucket/object"
    assert fs.project == "adc-default-project"
    assert fs.requester_pays == "consumer-project"

    with pytest.raises(ValueError, match="exact instance"):
        public.query_lamindb_edges(relation="x", instance="other/instance")
    with pytest.raises(ValueError, match="unsupported exact-ID"):
        public.query_lamindb_node("unknown", "id")


def test_fixture_catalog_queries_and_embeddings(tmp_path: Path) -> None:
    root = public.build_public_fixture(tmp_path / "kg")

    catalog = public.parquet_catalog(root)
    assert {"nodes", "edges", "evidence", "features"}.issubset(set(catalog["layer"]))
    assert catalog["rows"].gt(0).all()

    diseases = public.diseases_with_gene_evidence(root, "ENSG00000141510", limit=10)
    assert diseases["disease_id"].tolist() == ["EFO:0000616", "MONDO:0007254"]
    assert diseases["evidence_rows"].tolist() == [1, 1]

    neighbors = public.nearest_embeddings(
        root / "features" / "embeddings" / "text" / "fixture.parquet",
        "ENSG00000012048",
        limit=2,
    )
    assert neighbors.iloc[0]["node_id"] == "ENSG00000139618"
    assert np.isfinite(neighbors["cosine_similarity"]).all()

    joined = public.bounded_edge_evidence_join(
        root,
        "disease_associated_gene",
        edge_limit=5,
        evidence_limit=10,
    )
    assert len(joined) == 5
    assert {
        "source_assertion",
        "source_evidence",
        "source_record_id",
    }.issubset(joined.columns)


def test_sampled_pyg_and_ml_reuse_existing_pipeline(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    root = public.build_public_fixture(tmp_path / "kg")
    export = tmp_path / "pyg"

    result = public.build_sampled_pyg(root, export)
    smoke = public.run_sampled_ml(export, seed=13)

    assert result.node_counts == {"gene": 4, "disease": 3, "molecule": 3}
    assert result.edge_counts == {"disease_associated_gene": 5, "molecule_targets_gene": 4}
    assert smoke["status"] == "pass"
    assert smoke["validation"]["checks"]["nonempty_splits"] is True
