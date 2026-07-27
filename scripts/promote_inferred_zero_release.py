#!/usr/bin/env python3
"""Promote the reviewed zero-row formal-inference v2 release immutably."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

BUCKET = "jouvencekb"
CANONICAL_ROOT = "kg/v2"
PRODUCER_REVISION = "12fe3286f5091bd1a69a8287649e02e169737402"
PRODUCER_TASK = "t_50a6f3ce"
PROMOTION_TASK = "t_45cd6464"
RELEASE_ID = "post-operand-12fe3286f509-zero-rows"
FAMILY = "formal-relation-inference-v2"
SOURCE_NAMES = (
    "input_manifest.json",
    "pilot_report.json",
    "template_registry_v2.json",
)
SOURCE_SHA256 = {
    "input_manifest.json": "c71eddb953d069a914ecf1ded11844d39e8237fab718dcafaafd4796cc4cdcde",
    "pilot_report.json": "9392e83bc8c94143031d73a3bb78acb6211129643936f9e707ac70e44c0405b2",
    "template_registry_v2.json": "e23e34dfe51e3a568b2ee9e928ed6ddce4c8395bcac540c2b03dfcf03bb45e1a",
}
LAYERS = ("edges_inferred", "evidence_inferred")
BASE_OBJECT_NAMES = (*SOURCE_NAMES, "release_manifest.json")
RECEIPT_NAME = "promotion_receipt.json"
MARKER_NAME = "COMPLETED.json"


class PromotionError(RuntimeError):
    """Base class for safe promotion failures."""


class ReleaseConflict(PromotionError):
    """The immutable release identity already exists with conflicting state."""


class StorageBackend(Protocol):
    def list(self, prefixes: tuple[str, ...]) -> set[str]: ...

    def read(self, name: str) -> bytes: ...

    def metadata(self, name: str) -> "ObjectMetadata": ...

    def create(self, name: str, data: bytes, sha256: str) -> "ObjectMetadata": ...


@dataclass(frozen=True)
class ObjectMetadata:
    name: str
    generation: int
    size: int
    sha256: str

    @property
    def uri(self) -> str:
        return f"gs://{BUCKET}/{self.name}"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json(payload: object) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


def release_prefix(layer: str, release_id: str = RELEASE_ID) -> str:
    if layer not in LAYERS:
        raise PromotionError(f"unauthorized inferred layer: {layer}")
    return f"{CANONICAL_ROOT}/{layer}/{FAMILY}/releases/{release_id}"


def release_listing_prefix(layer: str, release_id: str = RELEASE_ID) -> str:
    """Return an exact release-directory prefix, excluding sibling release IDs."""
    return f"{release_prefix(layer, release_id)}/"


def object_name(layer: str, leaf: str, release_id: str = RELEASE_ID) -> str:
    return f"{release_prefix(layer, release_id)}/manifest/{leaf}"


def marker_name(release_id: str = RELEASE_ID) -> str:
    return f"{release_prefix('edges_inferred', release_id)}/{MARKER_NAME}"


def _validate_source(source_dir: Path) -> dict[str, bytes]:
    all_files = tuple(path for path in source_dir.rglob("*") if path.is_file())
    if any(path.suffix == ".parquet" for path in all_files):
        raise PromotionError("zero-row release must not contain placeholder Parquets")
    actual = {path.relative_to(source_dir).as_posix() for path in all_files}
    if actual != set(SOURCE_NAMES):
        raise PromotionError(
            f"source inventory mismatch: expected={sorted(SOURCE_NAMES)} actual={sorted(actual)}"
        )

    result: dict[str, bytes] = {}
    for name in SOURCE_NAMES:
        data = (source_dir / name).read_bytes()
        digest = sha256_bytes(data)
        if digest != SOURCE_SHA256[name]:
            raise PromotionError(
                f"accepted source hash mismatch for {name}: expected={SOURCE_SHA256[name]} actual={digest}"
            )
        result[name] = data

    report = json.loads(result["pilot_report.json"])
    registry = json.loads(result["template_registry_v2.json"])
    if len(registry) != 24:
        raise PromotionError(f"expected 24 templates, found {len(registry)}")
    if report.get("artifacts") != {}:
        raise PromotionError("accepted zero-row report unexpectedly declares row artifacts")
    if any(item.get("output_rows") != 0 for item in report["counts_by_template"].values()):
        raise PromotionError("accepted report contains nonzero inferred output")
    if not report.get("require_canonical_target_inventory"):
        raise PromotionError("complete canonical anti-join receipt requirement is absent")
    return result


def build_base_objects(
    source_dir: Path,
    *,
    release_id: str = RELEASE_ID,
    producer_revision: str = PRODUCER_REVISION,
) -> dict[str, bytes]:
    sources = _validate_source(source_dir)
    source_inventory = {
        name: {"bytes": len(data), "sha256": sha256_bytes(data)}
        for name, data in sorted(sources.items())
    }
    prefixes = {layer: release_prefix(layer, release_id) for layer in LAYERS}
    manifest = canonical_json(
        {
            "anti_join_receipt": {
                "canonical_overlap_rows": 0,
                "complete_target_inventory_required": True,
                "generated_paths_before_antijoin": 0,
                "staged_overlap_rows": 0,
            },
            "canonical_observed": False,
            "counts": {
                "derived_view_parquets": 0,
                "evidence_inferred_parquets": 0,
                "evidence_inferred_rows": 0,
                "fully_signed_paths": 0,
                "inferred_edge_parquets": 0,
                "inferred_edge_rows": 0,
                "joined_paths": 701,
                "templates": 24,
            },
            "epistemic_contract": {
                "abstention_reason": "disease mechanism unknown on all 701 joined protein paths",
                "algebra": "action * disease_mechanism * disease_direction",
                "classes_preserved_in": "template_registry_v2.json",
                "unknown_or_conflicting_operand": "abstain",
            },
            "expected_row_artifacts": [],
            "layers": prefixes,
            "marker": f"gs://{BUCKET}/{marker_name(release_id)}",
            "producer_revision": producer_revision,
            "producer_task": PRODUCER_TASK,
            "promotion_task": PROMOTION_TASK,
            "release_id": release_id,
            "schema_version": "inferred-zero-release-v1",
            "source_inventory": source_inventory,
        }
    )

    objects: dict[str, bytes] = {}
    for layer in LAYERS:
        for name, data in sources.items():
            objects[object_name(layer, name, release_id)] = data
        objects[object_name(layer, "release_manifest.json", release_id)] = manifest
    return objects


def _metadata_payload(metadata: ObjectMetadata) -> dict[str, object]:
    return {
        "bytes": metadata.size,
        "generation": metadata.generation,
        "sha256": metadata.sha256,
        "uri": metadata.uri,
    }


def build_receipt(
    base_metadata: dict[str, ObjectMetadata], *, release_id: str = RELEASE_ID
) -> bytes:
    return canonical_json(
        {
            "base_objects": [
                _metadata_payload(base_metadata[name]) for name in sorted(base_metadata)
            ],
            "canonical_readback": "byte-for-byte sha256 verified",
            "create_only_precondition": "if_generation_match=0",
            "marker_published": False,
            "release_id": release_id,
            "schema_version": "inferred-promotion-receipt-v1",
            "temporary_or_staging_objects": [],
        }
    )


def build_marker(
    promoted_metadata: dict[str, ObjectMetadata], *, release_id: str = RELEASE_ID
) -> bytes:
    return canonical_json(
        {
            "completed": True,
            "expected_inventory": [
                _metadata_payload(promoted_metadata[name])
                for name in sorted(promoted_metadata)
            ],
            "marker_is_last_object": True,
            "release_id": release_id,
            "schema_version": "inferred-release-completion-v1",
            "zero_row_release": True,
        }
    )


def _assert_exact_readback(
    backend: StorageBackend, expected: dict[str, bytes]
) -> dict[str, ObjectMetadata]:
    metadata: dict[str, ObjectMetadata] = {}
    for name, data in expected.items():
        remote = backend.read(name)
        if remote != data:
            raise ReleaseConflict(f"canonical byte conflict at gs://{BUCKET}/{name}")
        item = backend.metadata(name)
        digest = sha256_bytes(remote)
        if item.size != len(data) or item.sha256 != digest:
            raise ReleaseConflict(f"canonical metadata conflict at gs://{BUCKET}/{name}")
        metadata[name] = item
    return metadata


def _expected_names(release_id: str = RELEASE_ID) -> set[str]:
    names = {
        object_name(layer, leaf, release_id)
        for layer in LAYERS
        for leaf in (*BASE_OBJECT_NAMES, RECEIPT_NAME)
    }
    names.add(marker_name(release_id))
    return names


def promote(
    backend: StorageBackend,
    source_dir: Path,
    *,
    release_id: str = RELEASE_ID,
    producer_revision: str = PRODUCER_REVISION,
) -> dict[str, object]:
    base = build_base_objects(
        source_dir, release_id=release_id, producer_revision=producer_revision
    )
    prefixes = tuple(release_listing_prefix(layer, release_id) for layer in LAYERS)
    existing = backend.list(prefixes)

    if existing:
        expected_names = _expected_names(release_id)
        if existing != expected_names:
            raise ReleaseConflict(
                "same-identity inventory conflict before mutation: "
                f"missing={sorted(expected_names - existing)} extra={sorted(existing - expected_names)}"
            )
        base_metadata = _assert_exact_readback(backend, base)
        receipt = build_receipt(base_metadata, release_id=release_id)
        receipts = {
            object_name(layer, RECEIPT_NAME, release_id): receipt for layer in LAYERS
        }
        receipt_metadata = _assert_exact_readback(backend, receipts)
        promoted = {**base_metadata, **receipt_metadata}
        marker = {marker_name(release_id): build_marker(promoted, release_id=release_id)}
        marker_metadata = _assert_exact_readback(backend, marker)
        final_inventory = backend.list(prefixes)
        if final_inventory != expected_names:
            raise ReleaseConflict("same-identity inventory changed during replay readback")
        return _result(
            "verified-no-op", {**promoted, **marker_metadata}, release_id=release_id
        )

    base_metadata: dict[str, ObjectMetadata] = {}
    for name in sorted(base):
        data = base[name]
        base_metadata[name] = backend.create(name, data, sha256_bytes(data))
    base_metadata = _assert_exact_readback(backend, base)

    receipt = build_receipt(base_metadata, release_id=release_id)
    receipts = {
        object_name(layer, RECEIPT_NAME, release_id): receipt for layer in LAYERS
    }
    receipt_metadata: dict[str, ObjectMetadata] = {}
    for name in sorted(receipts):
        data = receipts[name]
        receipt_metadata[name] = backend.create(name, data, sha256_bytes(data))
    receipt_metadata = _assert_exact_readback(backend, receipts)

    promoted = {**base_metadata, **receipt_metadata}
    pre_marker_inventory = backend.list(prefixes)
    if pre_marker_inventory != set(promoted):
        raise ReleaseConflict(
            "release inventory changed before marker publication; marker not created"
        )
    # GCS has no atomic prefix compare-and-swap. The immutable marker therefore
    # enumerates the only objects that belong to this release; an uncooperative
    # external writer cannot make an extra object a release member. The exact
    # post-create listing below additionally detects namespace contamination.
    marker_data = build_marker(promoted, release_id=release_id)
    marker_path = marker_name(release_id)
    marker_metadata = backend.create(marker_path, marker_data, sha256_bytes(marker_data))
    all_metadata = {**promoted, marker_path: marker_metadata}
    _assert_exact_readback(backend, {**base, **receipts, marker_path: marker_data})

    final_inventory = backend.list(prefixes)
    if final_inventory != _expected_names(release_id):
        raise PromotionError("post-promotion inventory mismatch")
    if any(".tmp" in name or "/staging/" in name for name in final_inventory):
        raise PromotionError("temporary/staging residue detected")
    if any(name.endswith(".parquet") for name in final_inventory):
        raise PromotionError("zero-row release contains placeholder Parquet")
    return _result("created", all_metadata, release_id=release_id)


def _result(
    outcome: str, metadata: dict[str, ObjectMetadata], *, release_id: str
) -> dict[str, object]:
    ordered = [_metadata_payload(metadata[name]) for name in sorted(metadata)]
    return {
        "counts": {
            "canonical_objects": len(ordered),
            "evidence_inferred_rows": 0,
            "inferred_edge_rows": 0,
            "row_parquets": 0,
        },
        "inventory": ordered,
        "marker": f"gs://{BUCKET}/{marker_name(release_id)}",
        "outcome": outcome,
        "release_id": release_id,
    }


class GCSBackend:
    def __init__(self) -> None:
        from google.cloud import storage

        self._client = storage.Client(project="jkobject-1549353370965")
        self._bucket = self._client.bucket(BUCKET)

    def list(self, prefixes: tuple[str, ...]) -> set[str]:
        names: set[str] = set()
        for prefix in prefixes:
            names.update(blob.name for blob in self._client.list_blobs(BUCKET, prefix=prefix))
        return names

    def read(self, name: str) -> bytes:
        return self._bucket.blob(name).download_as_bytes()

    def metadata(self, name: str) -> ObjectMetadata:
        blob = self._bucket.get_blob(name)
        if blob is None or blob.generation is None or blob.size is None:
            raise PromotionError(f"canonical object missing: gs://{BUCKET}/{name}")
        data = blob.download_as_bytes()
        digest = sha256_bytes(data)
        declared = (blob.metadata or {}).get("sha256")
        if declared != digest:
            raise ReleaseConflict(
                f"canonical sha256 metadata mismatch at gs://{BUCKET}/{name}"
            )
        return ObjectMetadata(name, int(blob.generation), int(blob.size), digest)

    def create(self, name: str, data: bytes, sha256: str) -> ObjectMetadata:
        from google.api_core.exceptions import PreconditionFailed

        blob = self._bucket.blob(name)
        blob.metadata = {
            "immutable": "true",
            "release_id": RELEASE_ID,
            "sha256": sha256,
            "task_id": PROMOTION_TASK,
        }
        try:
            blob.upload_from_string(
                data,
                content_type="application/json",
                if_generation_match=0,
            )
        except PreconditionFailed as exc:
            raise ReleaseConflict(
                f"create-only precondition failed at gs://{BUCKET}/{name}"
            ) from exc
        blob.reload()
        return ObjectMetadata(name, int(blob.generation), len(data), sha256)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument(
        "--conflict-probe",
        action="store_true",
        help="Prove a divergent producer revision fails before mutation.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    backend = GCSBackend()
    if args.conflict_probe:
        before = backend.list(
            tuple(release_listing_prefix(layer) for layer in LAYERS)
        )
        try:
            promote(
                backend,
                args.source_dir,
                producer_revision="conflicting-revision-probe",
            )
        except ReleaseConflict as exc:
            after = backend.list(
                tuple(release_listing_prefix(layer) for layer in LAYERS)
            )
            if after != before:
                raise PromotionError("conflict probe mutated canonical inventory") from exc
            print(
                json.dumps(
                    {
                        "conflict_probe": "passed-fail-closed",
                        "error": str(exc),
                        "inventory_unchanged": True,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        raise PromotionError("conflicting identity was not rejected")

    print(json.dumps(promote(backend, args.source_dir), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
