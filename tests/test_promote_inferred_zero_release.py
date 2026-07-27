from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import promote_inferred_zero_release as promotion


class FakeBackend:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.generations: dict[str, int] = {}
        self.create_calls: list[str] = []
        self._next_generation = 100

    def list(self, prefixes: tuple[str, ...]) -> set[str]:
        return {
            name for name in self.objects if any(name.startswith(prefix) for prefix in prefixes)
        }

    def read(self, name: str) -> bytes:
        return self.objects[name]

    def metadata(self, name: str) -> promotion.ObjectMetadata:
        data = self.objects[name]
        return promotion.ObjectMetadata(
            name=name,
            generation=self.generations[name],
            size=len(data),
            sha256=promotion.sha256_bytes(data),
        )

    def create(
        self, name: str, data: bytes, sha256: str
    ) -> promotion.ObjectMetadata:
        if name in self.objects:
            raise promotion.ReleaseConflict(f"create-only conflict: {name}")
        assert sha256 == promotion.sha256_bytes(data)
        self._next_generation += 1
        self.objects[name] = data
        self.generations[name] = self._next_generation
        self.create_calls.append(name)
        return self.metadata(name)


def source_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    source = tmp_path / "manifest"
    source.mkdir()
    report = {
        "artifacts": {},
        "counts_by_template": {
            f"template_{index}": {"output_rows": 0} for index in range(24)
        },
        "require_canonical_target_inventory": True,
    }
    registry = [{"template_id": f"template_{index}"} for index in range(24)]
    payloads = {
        "input_manifest.json": {"files": {}, "snapshot_id": "accepted"},
        "pilot_report.json": report,
        "template_registry_v2.json": registry,
    }
    hashes: dict[str, str] = {}
    for name, payload in payloads.items():
        data = promotion.canonical_json(payload)
        (source / name).write_bytes(data)
        hashes[name] = promotion.sha256_bytes(data)
    monkeypatch.setattr(promotion, "SOURCE_SHA256", hashes)
    return source


def test_create_marker_last_and_replay_is_verified_noop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = source_dir(tmp_path, monkeypatch)
    backend = FakeBackend()

    created = promotion.promote(backend, source)

    assert created["outcome"] == "created"
    assert created["counts"] == {
        "canonical_objects": 11,
        "evidence_inferred_rows": 0,
        "inferred_edge_rows": 0,
        "row_parquets": 0,
    }
    assert backend.create_calls[-1] == promotion.marker_name()
    assert len(backend.objects) == 11
    assert not any(name.endswith(".parquet") for name in backend.objects)
    creates_before_replay = list(backend.create_calls)

    replayed = promotion.promote(backend, source)

    assert replayed["outcome"] == "verified-no-op"
    assert backend.create_calls == creates_before_replay
    assert replayed["inventory"] == created["inventory"]


def test_same_identity_byte_conflict_fails_before_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = source_dir(tmp_path, monkeypatch)
    backend = FakeBackend()
    promotion.promote(backend, source)
    before_objects = dict(backend.objects)
    before_creates = list(backend.create_calls)

    with pytest.raises(promotion.ReleaseConflict, match="canonical byte conflict"):
        promotion.promote(
            backend,
            source,
            producer_revision="conflicting-revision-probe",
        )

    assert backend.objects == before_objects
    assert backend.create_calls == before_creates


def test_partial_same_identity_inventory_fails_before_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = source_dir(tmp_path, monkeypatch)
    backend = FakeBackend()
    partial_name = promotion.object_name("edges_inferred", "input_manifest.json")
    backend.create(partial_name, b"partial", promotion.sha256_bytes(b"partial"))
    before = dict(backend.objects)

    with pytest.raises(promotion.ReleaseConflict, match="inventory conflict"):
        promotion.promote(backend, source)

    assert backend.objects == before
    assert backend.create_calls == [partial_name]


def test_sibling_release_with_shared_name_prefix_is_excluded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = source_dir(tmp_path, monkeypatch)
    backend = FakeBackend()
    sibling = promotion.object_name(
        "edges_inferred", "release_manifest.json", f"{promotion.RELEASE_ID}-sibling"
    )
    backend.create(sibling, b"sibling", promotion.sha256_bytes(b"sibling"))

    result = promotion.promote(backend, source)

    assert result["outcome"] == "created"
    assert sibling in backend.objects


def test_concurrent_unexpected_object_prevents_marker_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = source_dir(tmp_path, monkeypatch)

    class RacingBackend(FakeBackend):
        list_calls = 0

        def list(self, prefixes: tuple[str, ...]) -> set[str]:
            self.list_calls += 1
            if self.list_calls == 2:
                name = f"{promotion.release_prefix('edges_inferred')}/unexpected.json"
                self.objects[name] = b"unexpected"
                self.generations[name] = 999
            return super().list(prefixes)

    backend = RacingBackend()

    with pytest.raises(promotion.ReleaseConflict, match="marker not created"):
        promotion.promote(backend, source)

    assert promotion.marker_name() not in backend.objects


def test_nonzero_or_placeholder_source_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = source_dir(tmp_path, monkeypatch)
    report_path = source / "pilot_report.json"
    report = json.loads(report_path.read_text())
    report["counts_by_template"]["template_0"]["output_rows"] = 1
    data = promotion.canonical_json(report)
    report_path.write_bytes(data)
    hashes = dict(promotion.SOURCE_SHA256)
    hashes[report_path.name] = promotion.sha256_bytes(data)
    monkeypatch.setattr(promotion, "SOURCE_SHA256", hashes)

    with pytest.raises(promotion.PromotionError, match="nonzero inferred output"):
        promotion.build_base_objects(source)

    report["counts_by_template"]["template_0"]["output_rows"] = 0
    data = promotion.canonical_json(report)
    report_path.write_bytes(data)
    hashes[report_path.name] = promotion.sha256_bytes(data)
    monkeypatch.setattr(promotion, "SOURCE_SHA256", hashes)
    nested = source / "unexpected"
    nested.mkdir()
    (nested / "placeholder.parquet").write_bytes(b"")

    with pytest.raises(promotion.PromotionError, match="placeholder Parquets"):
        promotion.build_base_objects(source)
