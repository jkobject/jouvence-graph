from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from manage_db.pyg_artifact import (
    DEFAULT_PYG_CACHE,
    make_link_neighbor_loader,
    make_neighbor_loader,
    PygBuild,
    materialize_pyg_build,
    open_pyg_stores,
    resolve_pyg_build,
)

ROOT = Path(__file__).resolve().parents[1]


def test_default_cache_is_repo_local_and_ignored() -> None:
    assert DEFAULT_PYG_CACHE == ROOT / "data" / "pyg"
    assert "/data/pyg/" in (ROOT / ".gitignore").read_text()


def _write_manifest(root: Path) -> Path:
    root.mkdir(parents=True)
    payload = root / "validation" / "ok.txt"
    payload.parent.mkdir(parents=True)
    payload.write_text("ok\n")
    manifest = {
        "format_version": 1,
        "canonical_root": "gs://jouvencekb/main",
        "files": [{
            "path": "validation/ok.txt",
            "size": payload.stat().st_size,
            "sha256": hashlib.sha256(payload.read_bytes()).hexdigest(),
        }],
        "adjacency": [],
        "features": [],
    }
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path


def test_resolve_local_build_and_verify_copy(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_manifest(source)

    build = resolve_pyg_build(source)
    local = materialize_pyg_build(build, tmp_path / "cache", verify=True)

    assert build.uri == str(source)
    assert local.root == (tmp_path / "cache").resolve()
    assert (local.root / "validation" / "ok.txt").read_text() == "ok\n"


def test_materialize_gcs_build_uses_gcloud_storage_cp(tmp_path: Path) -> None:
    manifest = {"format_version": 1, "files": [], "adjacency": [], "features": []}
    build = PygBuild(uri="gs://jouvencekb/pyg", manifest=manifest)
    calls: list[list[str]] = []

    def runner(command: list[str]) -> None:
        calls.append(command)
        (tmp_path / "cache" / "manifest.json").write_text(json.dumps(manifest))

    materialize_pyg_build(build, tmp_path / "cache", verify=True, runner=runner)

    assert calls == [[
        "gcloud", "storage", "cp", "--recursive",
        "gs://jouvencekb/pyg/*", str((tmp_path / "cache").resolve()),
    ]]


def test_checksum_failure_is_fatal(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_manifest(source)
    build = resolve_pyg_build(source)
    (source / "validation" / "ok.txt").write_text("NO\n")

    with pytest.raises(ValueError, match="checksum mismatch"):
        materialize_pyg_build(build, tmp_path / "cache", verify=True)


def test_open_empty_reviewed_stores(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_manifest(source)
    local = materialize_pyg_build(resolve_pyg_build(source), tmp_path / "cache")

    graph_store, feature_store = open_pyg_stores(local, mmap=True)

    assert graph_store.get_all_edge_attrs() == []
    assert feature_store.get_all_tensor_attrs() == []


def test_open_memory_mapped_csc_and_features(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    adjacency = source / "adjacency"
    features = source / "feature_indices"
    adjacency.mkdir()
    features.mkdir()
    row = np.asarray([0, 2, 1], dtype=np.int64)
    colptr = np.asarray([0, 2, 3], dtype=np.int64)
    x = np.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    row.tofile(adjacency / "row.i64")
    colptr.tofile(adjacency / "colptr.i64")
    x.tofile(features / "gene_x.f32")

    files = []
    for path in (adjacency / "row.i64", adjacency / "colptr.i64", features / "gene_x.f32"):
        files.append({
            "path": str(path.relative_to(source)),
            "size": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
    manifest = {
        "format_version": 1,
        "files": files,
        "adjacency": [{
            "edge_type": ["gene", "associated", "disease"],
            "layout": "csc",
            "size": [3, 2],
            "first": {"path": "adjacency/row.i64", "dtype": "int64", "shape": [3]},
            "second": {"path": "adjacency/colptr.i64", "dtype": "int64", "shape": [3]},
        }],
        "features": [{
            "path": "feature_indices/gene_x.f32",
            "dtype": "float32",
            "shape": [3, 2],
            "group_name": "gene",
            "attr_name": "x",
        }],
    }
    (source / "manifest.json").write_text(json.dumps(manifest))
    local = materialize_pyg_build(resolve_pyg_build(source), tmp_path / "cache")

    graph_store, feature_store = open_pyg_stores(local, mmap=True)

    edge_attr = graph_store.get_all_edge_attrs()[0]
    opened_row, opened_colptr = graph_store.get_edge_index(edge_attr)
    assert opened_row.tolist() == [0, 2, 1]
    assert opened_colptr.tolist() == [0, 2, 3]
    tensor_attr = feature_store.get_all_tensor_attrs()[0]
    assert feature_store.get_tensor(tensor_attr).tolist() == x.tolist()

    import torch

    node_loader = make_neighbor_loader(
        graph_store=graph_store,
        feature_store=feature_store,
        input_nodes=("gene", torch.tensor([0, 1])),
        num_neighbors=[1],
        batch_size=2,
    )
    link_loader = make_link_neighbor_loader(
        graph_store=graph_store,
        feature_store=feature_store,
        edge_label_index=(("gene", "associated", "disease"), torch.tensor([[0], [1]])),
        num_neighbors=[1],
        batch_size=1,
    )
    assert node_loader.__class__.__name__ == "NeighborLoader"
    assert link_loader.__class__.__name__ == "LinkNeighborLoader"