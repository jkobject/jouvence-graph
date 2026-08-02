"""Immutable, read-only Jouvence viewer bundle loading.

A viewer bundle is deliberately different from the raw canonical KG.  It is a
small set of manifest-declared query sidecars that can be verified and loaded
without listing or scanning canonical Parquet tables.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import tempfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Iterator

import fcntl

from fsspec.core import url_to_fs

from . import fixture

MANIFEST_NAME = "viewer-manifest.json"
BUNDLE_SCHEMA = "jouvence-viewer-bundle-v1"
GCS_REVIEWED_PREFIX = "gs://jouvencekb/staging/viewer-bundles/"
MAX_MANIFEST_BYTES = 1_000_000
MAX_FILE_BYTES = 256 * 1024 * 1024
MAX_BUNDLE_BYTES = 1024 * 1024 * 1024
MAX_CACHE_BYTES = 2 * 1024 * 1024 * 1024
MAX_CACHE_FILES = 10_000
DATA_FILES = {
    "data/nodes.json",
    "data/edges.json",
    "data/evidence.json",
    "data/features.json",
    "data/long-range.json",
    "data/putative.json",
}
SAFE_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


class BundleError(ValueError):
    """A safe, actionable viewer-bundle validation failure."""


@dataclass
class ViewerData:
    snapshot_id: str
    bundle_version: str
    mode: str
    label: str
    cache_status: str
    requester_pays_warning: str | None
    billing_project: str | None
    manifest: dict[str, Any]
    NODES: dict[tuple[str, str], fixture.Node]
    EDGE_ROWS: list[dict[str, Any]]
    EVIDENCE_ROWS: list[dict[str, Any]]
    FEATURE_ROWS: list[dict[str, Any]]
    LONG_RANGE_ROWS: list[dict[str, Any]]
    PUTATIVE_ROWS: list[dict[str, Any]]

    @staticmethod
    def node_key(node_type: str, node_id: str) -> tuple[str, str]:
        return fixture.node_key(node_type, node_id)


class FixtureData:
    """Adapter preserving the Phase-1 in-process fixture mode."""

    snapshot_id = fixture.SNAPSHOT_ID
    bundle_version = fixture.BUNDLE_VERSION
    mode = fixture.DATA_MODE
    label = "Deterministic fixture"
    cache_status = "in-memory"
    requester_pays_warning = None
    billing_project = None
    manifest: dict[str, Any] = {}
    NODES = fixture.NODES
    EDGE_ROWS = fixture.EDGE_ROWS
    EVIDENCE_ROWS = fixture.EVIDENCE_ROWS
    FEATURE_ROWS = fixture.FEATURE_ROWS
    LONG_RANGE_ROWS = fixture.LONG_RANGE_ROWS
    PUTATIVE_ROWS = fixture.PUTATIVE_ROWS
    node_key = staticmethod(fixture.node_key)


FIXTURE_DATA = FixtureData()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _bundle_checksum(manifest: dict[str, Any]) -> str:
    signed = {
        "schema": manifest.get("schema"),
        "snapshot_id": manifest.get("snapshot_id"),
        "bundle_version": manifest.get("bundle_version"),
        "files": manifest.get("files"),
    }
    return _sha256(_json_bytes(signed))


def build_fixture_bundle(root: str | Path) -> Path:
    """Build a deterministic query-bundle fixture for smoke tests and demos."""

    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    payloads = {
        "data/nodes.json": [asdict(node) for node in fixture.NODES.values()],
        "data/edges.json": fixture.EDGE_ROWS,
        "data/evidence.json": fixture.EVIDENCE_ROWS,
        "data/features.json": fixture.FEATURE_ROWS,
        "data/long-range.json": fixture.LONG_RANGE_ROWS,
        "data/putative.json": fixture.PUTATIVE_ROWS,
    }
    entries: dict[str, dict[str, Any]] = {}
    for relative, value in payloads.items():
        payload = _json_bytes(value)
        path = root_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        entries[relative] = {"sha256": _sha256(payload), "size": len(payload)}
    manifest: dict[str, Any] = {
        "schema": BUNDLE_SCHEMA,
        "snapshot_id": fixture.SNAPSHOT_ID,
        "bundle_version": fixture.BUNDLE_VERSION,
        "files": entries,
        "capabilities": [
            "search",
            "dossier",
            "features",
            "edges",
            "evidence",
            "long_range",
            "putative",
            "export",
        ],
        "read_only": True,
    }
    manifest["bundle_checksum"] = _bundle_checksum(manifest)
    (root_path / MANIFEST_NAME).write_bytes(_json_bytes(manifest))
    return root_path


def _safe_relative(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or ".." in path.parts
        or "." in path.parts
        or "\\" in value
        or value not in DATA_FILES
    ):
        raise BundleError(f"unsafe bundle path in manifest: {value!r}")
    return value


def _validated_manifest(payload: bytes) -> dict[str, Any]:
    if len(payload) > MAX_MANIFEST_BYTES:
        raise BundleError("viewer manifest exceeds the 1 MB safety bound")
    try:
        manifest = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BundleError("viewer manifest is not valid UTF-8 JSON") from exc
    if not isinstance(manifest, dict) or manifest.get("schema") != BUNDLE_SCHEMA:
        raise BundleError(
            "data root is not a compatible viewer query bundle; download a reviewed "
            "bundle or build one with the documented producer before launching"
        )
    if not all(
        isinstance(manifest.get(key), str)
        and SAFE_IDENTIFIER.fullmatch(manifest[key])
        for key in ("snapshot_id", "bundle_version")
    ):
        raise BundleError("viewer manifest is missing snapshot or bundle version")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise BundleError("viewer manifest must declare exactly the supported query sidecars")
    for relative in files:
        _safe_relative(relative)
    if set(files) != DATA_FILES:
        raise BundleError("viewer manifest must declare exactly the supported query sidecars")
    total = 0
    for relative, entry in files.items():
        if not isinstance(entry, dict):
            raise BundleError("viewer manifest file entry is invalid")
        size = entry.get("size")
        digest = entry.get("sha256")
        if not isinstance(size, int) or size < 0 or size > MAX_FILE_BYTES:
            raise BundleError(f"bundle object exceeds safety bound: {relative}")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise BundleError(f"invalid checksum for bundle object: {relative}")
        total += size
    if total > MAX_BUNDLE_BYTES:
        raise BundleError("viewer bundle exceeds the 1 GB bounded-cache limit")
    if manifest.get("bundle_checksum") != _bundle_checksum(manifest):
        raise BundleError("viewer manifest bundle checksum mismatch")
    if manifest.get("read_only") is not True:
        raise BundleError("viewer bundle does not declare the read-only contract")
    return manifest


def _read_bounded(handle: BinaryIO, maximum: int) -> bytes:
    payload = handle.read(maximum + 1)
    if len(payload) > maximum:
        raise BundleError("remote viewer object exceeds its manifest safety bound")
    return payload


def _cache_usage(root: Path) -> tuple[int, int]:
    """Measure the cache without following symlinks, with count/byte ceilings."""

    if not root.exists():
        return 0, 0
    total = 0
    files = 0
    for path in root.rglob("*"):
        if path.is_symlink() or not path.is_file():
            continue
        files += 1
        if files > MAX_CACHE_FILES:
            raise BundleError("viewer cache exceeds the file-count safety bound")
        total += path.stat().st_size
        if total > MAX_CACHE_BYTES:
            raise BundleError("viewer cache exceeds the aggregate 2 GB safety bound")
    return total, files


@contextmanager
def _cache_lock(root: Path) -> Iterator[None]:
    """Serialize quota checks and atomic cache writes across viewer launches."""

    lock_path = root / ".viewer-cache.lock"
    _, existing_files = _cache_usage(root)
    if not lock_path.exists() and existing_files >= MAX_CACHE_FILES:
        raise BundleError("viewer cache exceeds the file-count safety bound")
    root.mkdir(parents=True, exist_ok=True)
    if not root.is_dir():
        raise BundleError("viewer cache root is not a directory")
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as exc:
        raise BundleError("cannot acquire the private viewer cache lock") from exc
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise BundleError("viewer cache lock is not a regular file")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _load_records(payloads: dict[str, bytes]) -> dict[str, Any]:
    records: dict[str, Any] = {}
    for relative, payload in payloads.items():
        try:
            value = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BundleError(f"invalid JSON in bundle object: {relative}") from exc
        if not isinstance(value, list):
            raise BundleError(f"bundle object must contain a JSON array: {relative}")
        records[relative] = value
    return records


_STRING_FIELDS: dict[str, tuple[str, ...]] = {
    "data/nodes.json": (
        "node_type",
        "node_id",
        "display_name",
        "description",
        "source",
    ),
    "data/edges.json": (
        "edge_key",
        "relation",
        "display_relation",
        "x_id",
        "x_type",
        "y_id",
        "y_type",
        "source",
        "kind",
    ),
    "data/evidence.json": (
        "edge_key",
        "relation",
        "x_id",
        "x_type",
        "y_id",
        "y_type",
        "source",
        "source_dataset",
        "source_record_id",
        "predicate",
        "row_kind",
        "paper_id",
        "license",
        "release",
    ),
    "data/features.json": (
        "node_type",
        "node_id",
        "feature_kind",
        "feature_key",
        "value",
        "source",
        "epistemic_kind",
        "release",
    ),
    "data/long-range.json": (
        "anchor_type",
        "anchor_id",
        "target_type",
        "target_id",
        "target_name",
        "ranker_id",
        "ranker_version",
        "support_path",
        "caveats",
        "row_kind",
    ),
    "data/putative.json": (
        "anchor_type",
        "anchor_id",
        "target_type",
        "target_id",
        "target_name",
        "policy_class",
        "template_id",
        "template_version",
        "support_path",
        "leakage_caveat",
        "row_kind",
    ),
}
_NUMBER_FIELDS: dict[str, tuple[str, ...]] = {
    "data/edges.json": ("score", "credibility"),
    "data/evidence.json": ("evidence_score",),
    "data/long-range.json": ("score", "rank", "path_length"),
}
_OPTIONAL_STRING_FIELDS: dict[str, tuple[str, ...]] = {
    "data/edges.json": (
        "effect_direction",
        "interaction_kind",
        "pharmacological_action",
    ),
}
_LIST_STRING_FIELDS: dict[str, tuple[str, ...]] = {
    "data/long-range.json": ("support_relations",),
    "data/putative.json": ("support_edge_hashes",),
}
_BOOL_FIELDS: dict[str, tuple[str, ...]] = {
    "data/long-range.json": ("observed_overlap",),
    "data/putative.json": ("observed_overlap",),
}
_ROW_LABELS = {
    "data/nodes.json": "node",
    "data/edges.json": "edge",
    "data/evidence.json": "evidence",
    "data/features.json": "feature",
    "data/long-range.json": "long-range",
    "data/putative.json": "putative",
}
MAX_ROW_STRING_BYTES = 64 * 1024
MAX_NODE_ALIASES = 100
MAX_ROW_LIST_ITEMS = 100
MAX_NODE_ATTRIBUTES = 100
MAX_ATTRIBUTE_STRING_BYTES = 4096
MAX_SIDECAR_ROW_BYTES = 64 * 1024


def _valid_row_string(value: Any) -> bool:
    return (
        isinstance(value, str)
        and bool(value)
        and len(value.encode("utf-8")) <= MAX_ROW_STRING_BYTES
    )


def _valid_attribute_string(value: Any) -> bool:
    return (
        isinstance(value, str)
        and bool(value)
        and len(value.encode("utf-8")) <= MAX_ATTRIBUTE_STRING_BYTES
    )


def _validate_sidecar_rows(records: dict[str, Any]) -> None:
    """Fail closed before API code indexes manifest-declared row fields."""

    for relative, rows in records.items():
        label = _ROW_LABELS[relative]
        for row in rows:
            if not isinstance(row, dict):
                raise BundleError(f"invalid {label} row in viewer bundle")
            if len(_json_bytes(row)) > MAX_SIDECAR_ROW_BYTES:
                raise BundleError(f"invalid {label} row in viewer bundle")
            allowed_fields = (
                set(_STRING_FIELDS[relative])
                | set(_NUMBER_FIELDS.get(relative, ()))
                | set(_OPTIONAL_STRING_FIELDS.get(relative, ()))
                | set(_LIST_STRING_FIELDS.get(relative, ()))
                | set(_BOOL_FIELDS.get(relative, ()))
            )
            if relative == "data/nodes.json":
                allowed_fields |= {"aliases", "attributes"}
            if set(row) - allowed_fields:
                raise BundleError(f"invalid {label} row in viewer bundle")
            if any(
                not _valid_row_string(row.get(field))
                for field in _STRING_FIELDS[relative]
            ):
                raise BundleError(f"invalid {label} row in viewer bundle")
            if any(
                isinstance(row.get(field), bool)
                or not isinstance(row.get(field), (int, float))
                or not math.isfinite(row[field])
                for field in _NUMBER_FIELDS.get(relative, ())
            ):
                raise BundleError(f"invalid {label} row in viewer bundle")
            if any(
                row.get(field) is not None and not _valid_row_string(row[field])
                for field in _OPTIONAL_STRING_FIELDS.get(relative, ())
            ):
                raise BundleError(f"invalid {label} row in viewer bundle")
            if any(
                not isinstance(row.get(field), list)
                or len(row[field]) > MAX_ROW_LIST_ITEMS
                or any(not _valid_row_string(item) for item in row[field])
                for field in _LIST_STRING_FIELDS.get(relative, ())
            ):
                raise BundleError(f"invalid {label} row in viewer bundle")
            if any(
                not isinstance(row.get(field), bool)
                for field in _BOOL_FIELDS.get(relative, ())
            ):
                raise BundleError(f"invalid {label} row in viewer bundle")

            if relative == "data/nodes.json":
                aliases = row.get("aliases")
                attributes = row.get("attributes")
                if (
                    not isinstance(aliases, list)
                    or len(aliases) > MAX_NODE_ALIASES
                    or not isinstance(attributes, dict)
                    or len(attributes) > MAX_NODE_ATTRIBUTES
                    or any(
                        not _valid_attribute_string(key)
                        or not _valid_attribute_string(value)
                        for key, value in attributes.items()
                    )
                    or any(
                        not isinstance(alias, dict)
                        or set(alias) != {"kind", "value", "source"}
                        or any(
                            not _valid_row_string(alias.get(field))
                            for field in ("kind", "value", "source")
                        )
                        for alias in aliases
                    )
                ):
                    raise BundleError("invalid node row in viewer bundle")


def _viewer_data(
    manifest: dict[str, Any],
    payloads: dict[str, bytes],
    *,
    mode: str,
    label: str,
    cache_status: str,
    billing_project: str | None,
) -> ViewerData:
    records = _load_records(payloads)
    _validate_sidecar_rows(records)
    nodes: dict[tuple[str, str], fixture.Node] = {}
    for row in records["data/nodes.json"]:
        try:
            node = fixture.Node(
                node_type=str(row["node_type"]),
                node_id=str(row["node_id"]),
                display_name=str(row["display_name"]),
                description=str(row["description"]),
                source=str(row["source"]),
                aliases=tuple(row.get("aliases", [])),
                attributes=dict(row.get("attributes", {})),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise BundleError("invalid node row in viewer bundle") from exc
        key = fixture.node_key(node.node_type, node.node_id)
        if key in nodes:
            raise BundleError("duplicate node identity in viewer bundle")
        nodes[key] = node
    warning = (
        "Requester-pays: your consumer project is billed for GCS requests, bytes, and egress."
        if mode == "gcs-requester-pays"
        else None
    )
    return ViewerData(
        snapshot_id=manifest["snapshot_id"],
        bundle_version=manifest["bundle_version"],
        mode=mode,
        label=label,
        cache_status=cache_status,
        requester_pays_warning=warning,
        billing_project=billing_project,
        manifest=manifest,
        NODES=nodes,
        EDGE_ROWS=records["data/edges.json"],
        EVIDENCE_ROWS=records["data/evidence.json"],
        FEATURE_ROWS=records["data/features.json"],
        LONG_RANGE_ROWS=records["data/long-range.json"],
        PUTATIVE_ROWS=records["data/putative.json"],
    )


def _verify_payload(relative: str, payload: bytes, entry: dict[str, Any]) -> None:
    if len(payload) != entry["size"]:
        raise BundleError(f"size mismatch for bundle object: {relative}")
    if _sha256(payload) != entry["sha256"]:
        raise BundleError(f"checksum mismatch for bundle object: {relative}")


def _open_local(root: Path) -> ViewerData:
    root = root.expanduser().resolve()
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise BundleError(
            "data root is not a compatible viewer query bundle; raw kg/v2 tables are "
            "never scanned on a laptop. Download a reviewed bundle or build one with "
            "the documented producer."
        )
    try:
        resolved_manifest = manifest_path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise BundleError("viewer manifest is missing") from exc
    if not resolved_manifest.is_relative_to(root):
        raise BundleError("viewer manifest escapes bundle root")
    if resolved_manifest.stat().st_size > MAX_MANIFEST_BYTES:
        raise BundleError("viewer manifest exceeds the 1 MB safety bound")
    with resolved_manifest.open("rb") as handle:
        manifest = _validated_manifest(_read_bounded(handle, MAX_MANIFEST_BYTES))
    payloads: dict[str, bytes] = {}
    for relative, entry in manifest["files"].items():
        path = root / _safe_relative(relative)
        try:
            resolved = path.resolve(strict=True)
        except FileNotFoundError as exc:
            raise BundleError(f"bundle object is missing: {relative}") from exc
        if not resolved.is_relative_to(root):
            raise BundleError(f"bundle object escapes bundle root: {relative}")
        if not resolved.is_file():
            raise BundleError(f"bundle object is not a regular file: {relative}")
        if resolved.stat().st_size != entry["size"]:
            raise BundleError(f"size mismatch for bundle object: {relative}")
        with resolved.open("rb") as handle:
            payload = _read_bounded(handle, entry["size"])
        _verify_payload(relative, payload, entry)
        payloads[relative] = payload
    return _viewer_data(
        manifest,
        payloads,
        mode="local",
        label="Verified local bundle",
        cache_status="verified-local",
        billing_project=None,
    )


def _default_cache_root() -> Path:
    return Path.home() / ".cache" / "jouvence-viewer"


def _cached_gcs_payloads(
    fs: Any,
    root_path: str,
    manifest: dict[str, Any],
    cache_base: Path,
) -> tuple[dict[str, bytes], bool]:
    cache_dir = cache_base / manifest["snapshot_id"] / manifest["bundle_checksum"]
    cache_bytes, cache_files = _cache_usage(cache_base)
    payloads: dict[str, bytes] = {}
    cache_hit = True
    for relative, entry in manifest["files"].items():
        relative = _safe_relative(relative)
        cached = cache_dir / relative
        if not cached.resolve(strict=False).is_relative_to(cache_base):
            raise BundleError(f"cache path escapes configured cache root: {relative}")
        payload: bytes | None = None
        if cached.is_symlink():
            cached.unlink()
        elif cached.is_file():
            cached_size = cached.stat().st_size
            if cached_size == entry["size"]:
                with cached.open("rb") as handle:
                    candidate = _read_bounded(handle, entry["size"])
                try:
                    _verify_payload(relative, candidate, entry)
                except BundleError:
                    cached.unlink()
                    cache_bytes -= cached_size
                    cache_files -= 1
                else:
                    payload = candidate
            else:
                cached.unlink()
                cache_bytes -= cached_size
                cache_files -= 1
        if payload is None:
            cache_hit = False
            if cache_files + 1 > MAX_CACHE_FILES:
                raise BundleError("viewer cache exceeds the file-count safety bound")
            if cache_bytes + entry["size"] > MAX_CACHE_BYTES:
                raise BundleError(
                    "viewer cache would exceed the aggregate 2 GB safety bound; "
                    "remove an old immutable snapshot cache and retry"
                )
            try:
                with fs.open(f"{root_path}/{relative}", "rb") as handle:
                    payload = _read_bounded(handle, entry["size"])
            except BundleError:
                raise
            except Exception as exc:
                raise BundleError(f"cannot read declared viewer object: {relative}") from exc
            _verify_payload(relative, payload, entry)
            cached.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=cached.parent, delete=False) as temporary:
                temporary.write(payload)
                temporary_path = Path(temporary.name)
            temporary_path.replace(cached)
            cache_bytes += len(payload)
            cache_files += 1
        payloads[relative] = payload
    return payloads, cache_hit


def _open_gcs(
    root_uri: str,
    *,
    billing_project: str | None,
    cache_root: Path | None,
    expected_manifest_sha256: str | None,
) -> ViewerData:
    if not billing_project:
        raise BundleError("GCS requester-pays mode requires --billing-project <consumer-project>")
    bundle_id = root_uri.removeprefix(GCS_REVIEWED_PREFIX)
    if not root_uri.startswith(GCS_REVIEWED_PREFIX) or not SAFE_IDENTIFIER.fullmatch(bundle_id):
        raise BundleError(
            "GCS data root must name a reviewed viewer bundle under "
            f"{GCS_REVIEWED_PREFIX}<bundle-id>; raw kg/v2 is never scanned"
        )
    if not expected_manifest_sha256 or not re.fullmatch(r"[0-9a-f]{64}", expected_manifest_sha256):
        raise BundleError(
            "GCS reviewed bundles require --manifest-sha256 with the independently "
            "published 64-character lowercase SHA-256"
        )
    try:
        fs, root_path = url_to_fs(
            root_uri,
            project=billing_project,
            requester_pays=billing_project,
            token="google_default",
        )
    except Exception as exc:
        raise BundleError(
            "cannot initialize GCS access; check ADC, the consumer billing project, "
            "and the Cloud Storage client installation"
        ) from exc
    root_path = root_path.rstrip("/")
    try:
        with fs.open(f"{root_path}/{MANIFEST_NAME}", "rb") as handle:
            manifest_payload = _read_bounded(handle, MAX_MANIFEST_BYTES)
    except BundleError:
        raise
    except Exception as exc:
        raise BundleError(
            "cannot read the reviewed viewer manifest; check ADC, bucket read IAM, "
            "Storage JSON API, serviceusage.services.use, and requester-pays billing"
        ) from exc
    if _sha256(manifest_payload) != expected_manifest_sha256:
        raise BundleError("reviewed viewer manifest does not match --manifest-sha256")
    manifest = _validated_manifest(manifest_payload)
    cache_base = (cache_root or _default_cache_root()).expanduser().resolve()
    with _cache_lock(cache_base):
        payloads, cache_hit = _cached_gcs_payloads(fs, root_path, manifest, cache_base)
    return _viewer_data(
        manifest,
        payloads,
        mode="gcs-requester-pays",
        label="GCS requester-pays bundle",
        cache_status="cache-hit" if cache_hit else "downloaded-verified",
        billing_project=billing_project,
    )


def open_viewer_bundle(
    data_root: str,
    *,
    billing_project: str | None = None,
    cache_root: str | Path | None = None,
    expected_manifest_sha256: str | None = None,
) -> ViewerData:
    """Open one immutable local or reviewed requester-pays query bundle."""

    if data_root.startswith("gs://"):
        return _open_gcs(
            data_root.rstrip("/"),
            billing_project=billing_project,
            cache_root=Path(cache_root) if cache_root is not None else None,
            expected_manifest_sha256=expected_manifest_sha256,
        )
    return _open_local(Path(data_root))
