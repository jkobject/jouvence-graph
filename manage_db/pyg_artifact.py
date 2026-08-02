"""Materialize and open the single Jouvence PyG build without GCS-FUSE.

Canonical Parquet remains under ``gs://jouvencekb/main``. The derived sampling
artifact lives directly under ``gs://jouvencekb/pyg`` and is copied once to a
user-selected local directory with ``gcloud storage cp`` before memory-mapped
access. The default is the repository-local ignored directory ``data/pyg``.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

PYG_ROOT = "gs://jouvencekb/pyg"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PYG_CACHE = REPO_ROOT / "data" / "pyg"


@dataclass(frozen=True)
class PygBuild:
    uri: str
    manifest: dict[str, Any]


@dataclass(frozen=True)
class LocalPygBuild:
    root: Path
    manifest: dict[str, Any]


CommandRunner = Callable[[list[str]], None]


def _default_runner(command: list[str]) -> None:
    subprocess.run(command, check=True)


def _read_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing PyG manifest: {path}")
    manifest = json.loads(path.read_text())
    if manifest.get("format_version") != 1:
        raise ValueError(f"unsupported PyG format_version: {manifest.get('format_version')!r}")
    for key in ("files", "adjacency", "features"):
        if not isinstance(manifest.get(key), list):
            raise ValueError(f"manifest field {key!r} must be a list")
    return manifest


def resolve_pyg_build(
    uri: str | Path = PYG_ROOT,
    *,
    runner: CommandRunner = _default_runner,
) -> PygBuild:
    """Resolve the one current PyG build and validate its manifest schema."""

    source = str(uri).rstrip("/")
    if source.startswith("gs://"):
        with tempfile.TemporaryDirectory(prefix="jouvence-pyg-manifest-") as directory:
            target = Path(directory) / "manifest.json"
            runner(["gcloud", "storage", "cp", f"{source}/manifest.json", str(target)])
            manifest = _read_manifest(target)
    else:
        manifest = _read_manifest(Path(source) / "manifest.json")
    return PygBuild(uri=source, manifest=manifest)


def _safe_relative(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"manifest path escapes build root: {relative!r}") from exc
    return path


def verify_pyg_build(build: LocalPygBuild) -> None:
    """Verify every manifest-declared file by exact size and SHA-256."""

    for entry in build.manifest["files"]:
        relative = str(entry["path"])
        path = _safe_relative(build.root, relative)
        if not path.is_file():
            raise FileNotFoundError(f"manifest file missing: {relative}")
        expected_size = int(entry["size"])
        if path.stat().st_size != expected_size:
            raise ValueError(f"size mismatch for {relative}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != entry["sha256"]:
            raise ValueError(f"checksum mismatch for {relative}")


def materialize_pyg_build(
    build: PygBuild,
    cache_dir: str | Path = DEFAULT_PYG_CACHE,
    *,
    verify: bool = True,
    runner: CommandRunner = _default_runner,
) -> LocalPygBuild:
    """Copy the complete build locally and optionally verify every file."""

    target = Path(cache_dir).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    if build.uri.startswith("gs://"):
        runner([
            "gcloud",
            "storage",
            "cp",
            "--recursive",
            f"{build.uri}/*",
            str(target),
        ])
    else:
        source = Path(build.uri).resolve()
        if source != target:
            shutil.copytree(source, target, dirs_exist_ok=True)
    local = LocalPygBuild(root=target, manifest=_read_manifest(target / "manifest.json"))
    if local.manifest != build.manifest:
        raise ValueError("materialized manifest differs from resolved manifest")
    if verify:
        verify_pyg_build(local)
    return local


try:
    import torch
    from torch_geometric.data import EdgeAttr, EdgeLayout, FeatureStore, GraphStore, TensorAttr
except Exception:  # pragma: no cover - explicit runtime error in open_pyg_stores
    torch = None  # type: ignore[assignment]
    EdgeAttr = EdgeLayout = FeatureStore = GraphStore = TensorAttr = None  # type: ignore[assignment,misc]


_DTYPE = {
    "float16": np.float16,
    "float32": np.float32,
    "float64": np.float64,
    "int32": np.int32,
    "int64": np.int64,
    "bool": np.bool_,
}

_TORCH_DTYPE = {
    "float16": "float16",
    "float32": "float32",
    "float64": "float64",
    "int32": "int32",
    "int64": "int64",
    "bool": "bool",
}


class JouvenceGraphStore(GraphStore):  # type: ignore[misc]
    def __init__(self) -> None:
        super().__init__()
        self._edges: dict[tuple[Any, ...], tuple[Any, Any, Any]] = {}

    def _put_edge_index(self, edge_index, edge_attr) -> bool:
        self._edges[edge_attr.edge_type] = (edge_index[0], edge_index[1], edge_attr)
        return True

    def _get_edge_index(self, edge_attr):
        item = self._edges.get(edge_attr.edge_type)
        return None if item is None else (item[0], item[1])

    def _remove_edge_index(self, edge_attr) -> bool:
        return self._edges.pop(edge_attr.edge_type, None) is not None

    def get_all_edge_attrs(self):
        return [item[2] for item in self._edges.values()]


class JouvenceFeatureStore(FeatureStore):  # type: ignore[misc]
    def __init__(self) -> None:
        super().__init__(tensor_attr_cls=TensorAttr)
        self._tensors: dict[tuple[str, str], tuple[Any, Any]] = {}

    @staticmethod
    def _key(attr) -> tuple[str, str]:
        return str(attr.group_name), str(attr.attr_name)

    def _put_tensor(self, tensor, attr) -> bool:
        self._tensors[self._key(attr)] = (tensor, attr)
        return True

    def _get_tensor(self, attr):
        item = self._tensors.get(self._key(attr))
        if item is None:
            return None
        tensor = item[0]
        return tensor if attr.index is None else tensor[attr.index]

    def _remove_tensor(self, attr) -> bool:
        return self._tensors.pop(self._key(attr), None) is not None

    def _get_tensor_size(self, attr):
        tensor = self._get_tensor(attr)
        return None if tensor is None else tuple(tensor.size())

    def get_all_tensor_attrs(self):
        return [item[1] for item in self._tensors.values()]


def _mmap_tensor(root: Path, entry: dict[str, Any]):
    if torch is None:
        raise RuntimeError("opening PyG stores requires the 'gnn' dependency group")
    dtype_name = str(entry["dtype"])
    if dtype_name not in _DTYPE:
        raise ValueError(f"unsupported mmap dtype: {dtype_name}")
    shape = tuple(int(value) for value in entry["shape"])
    path = _safe_relative(root, str(entry["path"]))
    size = int(np.prod(shape, dtype=np.int64))
    torch_dtype = getattr(torch, _TORCH_DTYPE[dtype_name])
    return torch.from_file(str(path), shared=False, size=size, dtype=torch_dtype).reshape(shape)


def open_pyg_stores(build: LocalPygBuild, *, mmap: bool = True):
    """Open manifest-declared adjacency/features as PyG stores.

    The current contract is disk-backed only; ``mmap=False`` is rejected rather
    than silently loading the whole artifact into RAM.
    """

    if torch is None or GraphStore is None:
        raise RuntimeError("opening PyG stores requires the 'gnn' dependency group")
    if not mmap:
        raise ValueError("Jouvence PyG stores must be opened with mmap=True")
    graph_store = JouvenceGraphStore()
    feature_store = JouvenceFeatureStore()
    for entry in build.manifest["adjacency"]:
        edge_type = tuple(entry["edge_type"])
        layout = EdgeLayout(str(entry.get("layout", "csc")))
        first = _mmap_tensor(build.root, entry["first"])
        second = _mmap_tensor(build.root, entry["second"])
        attr = EdgeAttr(
            edge_type=edge_type,
            layout=layout,
            size=tuple(int(value) for value in entry["size"]),
            is_sorted=True,
        )
        graph_store.put_edge_index((first, second), attr)
    for entry in build.manifest["features"]:
        tensor = _mmap_tensor(build.root, entry)
        feature_store.put_tensor(
            tensor,
            group_name=str(entry["group_name"]),
            attr_name=str(entry["attr_name"]),
            index=None,
        )
    return graph_store, feature_store


def make_neighbor_loader(*, graph_store, feature_store, **kwargs):
    from torch_geometric.loader import NeighborLoader

    return NeighborLoader(data=(feature_store, graph_store), **kwargs)


def make_link_neighbor_loader(*, graph_store, feature_store, **kwargs):
    from torch_geometric.loader import LinkNeighborLoader

    return LinkNeighborLoader(data=(feature_store, graph_store), **kwargs)
