"""Build a LaminDB-oriented manifest for canonical KG Parquet artifacts.

The manifest is intentionally metadata-only by default: it inspects file and
Parquet footer metadata without materializing node, edge, evidence, or feature
rows.  It is the idempotency driver for ``sync_kg_artifacts_to_lamindb``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import posixpath
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq

from . import kg_storage

try:  # Relation metadata is useful but not required for arbitrary local smokes.
    from .kg_schema import RELATIONS
except Exception:  # pragma: no cover
    RELATIONS = []  # type: ignore[assignment]


LAYER_DIRS = (
    "nodes",
    "edges",
    "edges_inferred",
    "evidence",
    "evidence_inferred",
    "features",
    "embeddings",
)
DEFAULT_MANIFEST_NAME = "lamindb_manifest.json"


@dataclass(frozen=True)
class KGLayerManifestEntry:
    layer: str
    name: str
    key: str
    uri: str
    rows: int | None
    bytes: int | None
    columns: list[str]
    dtypes: dict[str, str]
    row_group_count: int | None
    parquet_schema_hash: str | None
    content_hash: str | None
    metadata_fingerprint: str
    labels: list[str]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class KGManifest:
    kg_version: str
    canonical_root: str
    scan_root: str
    generated_at: str
    code_sha: str
    layers: list[KGLayerManifestEntry]
    summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "kg_version": self.kg_version,
            "canonical_root": self.canonical_root,
            "scan_root": self.scan_root,
            "generated_at": self.generated_at,
            "code_sha": self.code_sha,
            "layers": [asdict(layer) for layer in self.layers],
            "summary": self.summary,
        }


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _current_code_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _list_parquet_stems(root: kg_storage.KGRoot, layer: str) -> list[str]:
    directory = root._join(layer)
    if not directory or not root.fs.exists(directory):
        return []
    stems: list[str] = []
    for entry in root.fs.ls(directory, detail=True):
        if entry.get("type") != "file":
            continue
        name = entry.get("name") or ""
        if name.endswith(".parquet"):
            stems.append(posixpath.splitext(posixpath.basename(name))[0])
    return sorted(stems)


def _internal_path(root: kg_storage.KGRoot, layer: str, name: str) -> str:
    return root._join(layer, f"{name}.parquet")


def _public_uri(root: kg_storage.KGRoot, layer: str, name: str, *, public_root: str | None = None) -> str:
    base = (public_root or root.uri).rstrip("/")
    return f"{base}/{layer}/{name}.parquet"


def _file_info(root: kg_storage.KGRoot, internal_path: str) -> dict[str, Any]:
    try:
        info = root.fs.info(internal_path)
    except Exception:
        return {}
    return dict(info or {})


def _metadata_fingerprint(info: dict[str, Any], schema_hash: str | None, rows: int | None) -> str:
    stable = {
        "size": info.get("size"),
        "mtime": str(info.get("mtime") or info.get("updated") or info.get("LastModified") or ""),
        "etag": info.get("etag") or info.get("ETag"),
        "generation": info.get("generation") or info.get("Generation"),
        "rows": rows,
        "schema_hash": schema_hash,
    }
    return _sha256_text(json.dumps(stable, sort_keys=True, default=str))


def _hash_content(root: kg_storage.KGRoot, internal_path: str, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with root.fs.open(internal_path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _relation_metadata(name: str) -> dict[str, Any]:
    for relation in RELATIONS:
        if getattr(relation, "name", None) == name:
            return {
                "x_type": relation.source.value,
                "y_type": relation.target.value,
                "kind": relation.kind.value,
                "direct": bool(relation.direct),
                "status": relation.status.value,
                "notes": relation.notes,
            }
    return {}


def _labels_for(layer: str, name: str, metadata: dict[str, Any]) -> list[str]:
    labels = ["kg", "kg-main", "canonical", f"kg-layer:{layer}"]
    if layer == "nodes":
        labels.append(f"node_type:{name}")
    elif layer in {"edges", "evidence"}:
        labels.append(f"relation:{name}")
        if metadata.get("x_type"):
            labels.append(f"x_type:{metadata['x_type']}")
        if metadata.get("y_type"):
            labels.append(f"y_type:{metadata['y_type']}")
        if metadata.get("kind"):
            labels.append(f"kind:{metadata['kind']}")
        if "direct" in metadata:
            labels.append(f"direct:{'yes' if metadata['direct'] else 'no'}")
    elif layer == "features":
        labels.append("feature")
        parts = name.split("_", 1)
        if parts:
            labels.append(f"node_type:{parts[0]}")
    return sorted(set(labels))


def _parquet_entry(
    root: kg_storage.KGRoot,
    layer: str,
    name: str,
    *,
    hash_content: bool,
    public_root: str | None = None,
) -> KGLayerManifestEntry:
    internal = _internal_path(root, layer, name)
    info = _file_info(root, internal)
    with root.fs.open(internal, "rb") as handle:
        parquet_file = pq.ParquetFile(handle)
        schema = parquet_file.schema_arrow
        rows = parquet_file.metadata.num_rows if parquet_file.metadata is not None else None
        row_groups = parquet_file.metadata.num_row_groups if parquet_file.metadata is not None else None
    columns = list(schema.names)
    dtypes = {field.name: str(field.type) for field in schema}
    schema_hash = _sha256_text(str(schema))
    content_hash = _hash_content(root, internal) if hash_content else None
    metadata = _relation_metadata(name) if layer in {"edges", "evidence"} else {}
    metadata.update({"parquet_format": "parquet"})
    key = f"main/{layer}/{name}.parquet"
    return KGLayerManifestEntry(
        layer=layer,
        name=name,
        key=key,
        uri=_public_uri(root, layer, name, public_root=public_root),
        rows=rows,
        bytes=info.get("size"),
        columns=columns,
        dtypes=dtypes,
        row_group_count=row_groups,
        parquet_schema_hash=schema_hash,
        content_hash=content_hash,
        metadata_fingerprint=content_hash or _metadata_fingerprint(info, schema_hash, rows),
        labels=_labels_for(layer, name, metadata),
        metadata=metadata,
    )


def build_manifest(
    kg_path: str | Path,
    *,
    kg_version: str = "v2",
    hash_content: bool = False,
    public_root: str | None = None,
) -> KGManifest:
    root = kg_storage.open_kg_root(str(kg_path))
    layers: list[KGLayerManifestEntry] = []
    for layer in LAYER_DIRS:
        for name in _list_parquet_stems(root, layer):
            layers.append(
                _parquet_entry(
                    root,
                    layer,
                    name,
                    hash_content=hash_content,
                    public_root=public_root,
                )
            )
    counts_by_layer: dict[str, int] = {layer: 0 for layer in LAYER_DIRS}
    rows_by_layer: dict[str, int] = {layer: 0 for layer in LAYER_DIRS}
    bytes_by_layer: dict[str, int] = {layer: 0 for layer in LAYER_DIRS}
    for entry in layers:
        counts_by_layer[entry.layer] += 1
        rows_by_layer[entry.layer] += entry.rows or 0
        bytes_by_layer[entry.layer] += entry.bytes or 0
    summary = {
        "artifact_count": len(layers),
        "counts_by_layer": counts_by_layer,
        "rows_by_layer": rows_by_layer,
        "bytes_by_layer": bytes_by_layer,
    }
    return KGManifest(
        kg_version=kg_version,
        canonical_root=(public_root or root.uri).rstrip("/"),
        scan_root=root.uri,
        generated_at=datetime.now(timezone.utc).isoformat(),
        code_sha=_current_code_sha(),
        layers=layers,
        summary=summary,
    )


def write_manifest(manifest: KGManifest, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def load_manifest(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def iter_layer_entries(manifest: dict[str, Any]) -> Iterable[dict[str, Any]]:
    return manifest.get("layers", [])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kg_path", help="Local, FUSE, or gs:// KG root")
    parser.add_argument("--kg-version", default="v2")
    parser.add_argument("--public-root", default=None, help="Canonical URI root to record in the manifest, e.g. gs://jouvencekb/main when scanning via a local FUSE mount")
    parser.add_argument("--output", default=None, help="JSON output path; default: <kg_path>/reports/lamindb_manifest.json for local roots")
    parser.add_argument("--hash-content", action="store_true", help="Compute sha256 of each Parquet file; can be slow for full KG")
    parser.add_argument("--json", action="store_true", help="Print manifest JSON to stdout")
    args = parser.parse_args(argv)

    manifest = build_manifest(args.kg_path, kg_version=args.kg_version, hash_content=args.hash_content, public_root=args.public_root)
    if args.json:
        print(json.dumps(manifest.to_dict(), indent=2, sort_keys=True))
    else:
        if args.output:
            output = Path(args.output)
        elif "://" not in str(args.kg_path):
            output = Path(args.kg_path) / "reports" / DEFAULT_MANIFEST_NAME
        else:
            output = Path(".") / DEFAULT_MANIFEST_NAME
        write_manifest(manifest, output)
        print(f"wrote {output} with {manifest.summary['artifact_count']} artifact entries")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
