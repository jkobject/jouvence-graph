from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import nbformat
import pytest

from manage_db import data_explorer

ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "scripts" / "build_data_explorer_notebook.py"
NOTEBOOK = ROOT / "notebooks" / "07_data_inventory_explorer.ipynb"


def test_data_explorer_is_deterministic_and_bounded(tmp_path: Path) -> None:
    first = tmp_path / "first.ipynb"
    second = tmp_path / "second.ipynb"
    subprocess.run(
        [sys.executable, str(GENERATOR), "--output", str(first)], cwd=ROOT, check=True
    )
    subprocess.run(
        [sys.executable, str(GENERATOR), "--output", str(second)], cwd=ROOT, check=True
    )
    assert first.read_bytes() == second.read_bytes()
    assert NOTEBOOK.read_bytes() == first.read_bytes()

    notebook = nbformat.read(NOTEBOOK, as_version=4)
    text = "\n".join(str(cell.source) for cell in notebook.cells)
    assert notebook.metadata["jouvence"]["bounded"] is True
    assert notebook.metadata["jouvence"]["read_only"] is True
    assert notebook.metadata["jouvence"]["data_mode"] == "live-gcs-only"
    assert "canonical-inferred" in text
    assert "non-canonical" in text
    assert "read_bounded_parquet" in text
    assert "list_parquet_uris" in text
    assert "live-gcs-only" in text
    assert "fs.glob(" not in text
    assert "build_public_fixture" not in text
    assert "kg-fixture" not in text
    assert "fixture_rule_engine" not in text
    assert "JOUVENCE_DATA_MODE" not in text
    assert "SELECTED_LAYER" in text
    assert "SELECTED_NAME" in text
    assert "/Users/jkobject/mnt/gcs" not in text
    assert "jkobject-1549353370965" not in text
    assert all(not cell.get("outputs") for cell in notebook.cells if cell.cell_type == "code")


@pytest.mark.parametrize(
    ("variable", "value", "message"),
    [
        ("JOUVENCE_EXPLORER_SAMPLE_ROWS", "not-an-integer", "must be integers"),
        ("JOUVENCE_EXPLORER_SAMPLE_ROWS", "0", "must be between 1 and 100"),
        ("JOUVENCE_EXPLORER_SAMPLE_ROWS", "101", "must be between 1 and 100"),
        ("JOUVENCE_EXPLORER_MAX_FILES", "not-an-integer", "must be integers"),
        ("JOUVENCE_EXPLORER_MAX_FILES", "0", "must be between 1 and 2000"),
        ("JOUVENCE_EXPLORER_MAX_FILES", "2001", "must be between 1 and 2000"),
    ],
)
def test_optimized_python_cannot_remove_safety_bounds(
    variable: str, value: str, message: str
) -> None:
    environment = os.environ.copy()
    environment.update({"PYTHONOPTIMIZE": "2", variable: value})
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check_data_explorer_notebook.py"), "--execute"],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert message in result.stdout + result.stderr


def test_live_only_roots_all_use_requester_pays_project() -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    text = "\n".join(str(cell.source) for cell in notebook.cells)

    assert "canonical_root = PUBLIC_KG_ROOT" in text
    assert 'staging_root = "gs://jouvencekb/staging"' in text
    assert "JOUVENCE_CANONICAL_ROOT" not in text
    assert "JOUVENCE_STAGING_ROOT" not in text
    assert "billing_project=BILLING_PROJECT" in text
    assert "_storage_options(uri, BILLING_PROJECT)" in text


def test_notebook_teaches_named_loading_graph_links_umap_gaps_and_pyg() -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    text = "\n".join(str(cell.source) for cell in notebook.cells)

    # One exact layer/name selection surface rather than forcing users to copy URIs.
    assert "def uri_for(layer: str, name: str)" in text
    assert "def load_table(layer: str, name: str" in text
    assert "files_by_type" in text
    assert "full_table_statistics" in text

    # Executable relational walkthroughs across the graph and provenance layers.
    assert "node → edge → node" in text
    assert "edge_with_endpoints" in text
    assert "edge_with_evidence" in text
    assert 'on=["relation", "x_id", "x_type", "y_id", "y_type"]' in text

    # Embeddings and backlog/gap views are explicit and bounded.
    assert "UMAP" in text
    assert "PCA fallback" in text
    assert "embedding_projection" in text
    assert "schema_relations_absent_from_kg" in text
    assert "staging_objects" in text
    assert 'columns=["node_type", "primary_ontology"]' in text

    # Stream canonical Parquets; never build or load a whole-graph pickle.
    assert "iter_relation_minibatches" in text
    assert "EmbeddingSpec(" in text
    assert "edge minibatching" in text
    assert "neighbor sampling" in text
    assert "evidence_records" in text
    assert "evidence_to_edge" in text
    assert "BuildConfig(" not in text
    assert "build_pyg_export" not in text
    assert "full_graph.pt" not in text
    assert "pickle.load" not in text
    assert "torch.load(handle" not in text
    assert "HeteroData" in text
    assert "umap-learn" in (ROOT / "pyproject.toml").read_text()

    # The production path is implemented, not left as hypothetical prose.
    assert "resolve_pyg_build" in text
    assert "materialize_pyg_build" in text
    assert "open_pyg_stores" in text
    assert "make_neighbor_loader" in text
    assert "make_link_neighbor_loader" in text
    assert 'PYG_ROOT = "gs://jouvencekb/pyg"' in text
    assert "gcloud storage cp" in text
    assert "gcsfuse" not in text.lower()
    assert "graph-build-id" not in text


def test_local_listing_is_early_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for index in range(5):
        path = tmp_path / f"part-{index}.parquet"
        path.write_bytes(b"fixture")

    listing = data_explorer.list_parquet_uris(tmp_path, limit=2)

    assert len(listing.uris) == 2
    assert listing.truncated is True
    assert listing.uris[0].endswith("part-0.parquet")

    def failing_walk(_root, *, onerror):
        onerror(PermissionError("local denied"))
        return iter(())

    with monkeypatch.context() as scoped:
        scoped.setattr(data_explorer.os, "walk", failing_walk)
        with pytest.raises(PermissionError, match="local denied"):
            data_explorer.list_parquet_uris(tmp_path, limit=2)


def test_gcs_listing_uses_server_side_cap_and_propagates_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}

    class Blob:
        def __init__(self, name: str):
            self.name = name

    class FakeClient:
        def bucket(self, name: str, *, user_project: str):
            calls["bucket"] = (name, user_project)
            return object()

        def list_blobs(self, _bucket, **kwargs):
            calls["list"] = kwargs
            return [Blob("main/edges/a.parquet"), Blob("main/edges/b.parquet"), Blob("main/edges/c.parquet")]

    monkeypatch.setattr(data_explorer, "Client", FakeClient)
    listing = data_explorer.list_parquet_uris(
        "gs://jouvencekb/main/edges", limit=2, billing_project="caller-project"
    )

    assert listing.uris == (
        "gs://jouvencekb/main/edges/a.parquet",
        "gs://jouvencekb/main/edges/b.parquet",
    )
    assert listing.truncated is True
    assert calls["bucket"] == ("jouvencekb", "caller-project")
    assert calls["list"] == {
        "prefix": "main/edges/",
        "match_glob": "main/edges/**/*.parquet",
        "max_results": 3,
        "page_size": 3,
    }

    class FailingClient(FakeClient):
        def list_blobs(self, _bucket, **kwargs):
            raise PermissionError("denied")

    monkeypatch.setattr(data_explorer, "Client", FailingClient)
    with pytest.raises(PermissionError, match="denied"):
        data_explorer.list_parquet_uris(
            "gs://jouvencekb/main/edges", limit=2, billing_project="caller-project"
        )
