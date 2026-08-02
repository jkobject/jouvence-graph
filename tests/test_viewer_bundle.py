from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from manage_db.viewer.app import create_app
from manage_db.viewer.bundle import (
    BundleError,
    MANIFEST_NAME,
    _bundle_checksum,
    build_fixture_bundle,
    open_viewer_bundle,
)

TEST_TOKEN = "test-session-token"
LOCAL_HEADERS = {
    "host": "127.0.0.1:8765",
    "x-jouvence-session": TEST_TOKEN,
}


def test_generated_local_bundle_works_end_to_end(tmp_path: Path) -> None:
    bundle_root = build_fixture_bundle(tmp_path / "bundle")
    source = open_viewer_bundle(str(bundle_root))
    client = TestClient(create_app(data_source=source, session_token=TEST_TOKEN))

    session = client.get("/api/session", headers=LOCAL_HEADERS).json()
    assert session["source"]["mode"] == "local"
    assert session["snapshot"]["snapshot_id"] == "fixture-v1"
    assert session["cache"]["status"] == "verified-local"
    assert session["source"]["credential_transport"] == "host-only"

    result = client.get("/api/search", params={"q": "BRCA1"}, headers=LOCAL_HEADERS).json()
    assert result["results"][0]["node_id"] == "ENSG00000012048"


def test_manifest_checksum_and_raw_canonical_fail_closed(tmp_path: Path) -> None:
    bundle_root = build_fixture_bundle(tmp_path / "bundle")
    data_path = bundle_root / "data" / "nodes.json"
    payload = data_path.read_bytes()
    data_path.write_bytes(bytes([payload[0] ^ 1]) + payload[1:])
    with pytest.raises(BundleError, match="checksum mismatch"):
        open_viewer_bundle(str(bundle_root))

    raw = tmp_path / "v2"
    (raw / "nodes").mkdir(parents=True)
    (raw / "nodes" / "gene.parquet").write_bytes(b"PAR1")
    with pytest.raises(BundleError, match="compatible viewer query bundle"):
        open_viewer_bundle(str(raw))


def test_manifest_rejects_path_escape_and_symlink_escape(tmp_path: Path) -> None:
    bundle_root = build_fixture_bundle(tmp_path / "bundle")
    manifest_path = bundle_root / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text())
    manifest["files"]["../outside.json"] = {
        "sha256": hashlib.sha256(b"secret").hexdigest(),
        "size": 6,
    }
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(BundleError, match="unsafe bundle path"):
        open_viewer_bundle(str(bundle_root))

    bundle_root = build_fixture_bundle(tmp_path / "symlink-bundle")
    outside = tmp_path / "outside.json"
    outside.write_text("[]")
    target = bundle_root / "data" / "nodes.json"
    target.unlink()
    target.symlink_to(outside)
    with pytest.raises(BundleError, match="escapes bundle root"):
        open_viewer_bundle(str(bundle_root))

    bundle_root = build_fixture_bundle(tmp_path / "snapshot-bundle")
    manifest_path = bundle_root / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text())
    manifest["snapshot_id"] = "../../outside-cache"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(BundleError, match="snapshot or bundle version"):
        open_viewer_bundle(str(bundle_root))


class FakeGCS:
    def __init__(self, objects: dict[str, bytes]):
        self.objects = objects
        self.opened: list[str] = []

    def open(self, path: str, mode: str = "rb"):
        assert mode == "rb"
        self.opened.append(path)
        if path not in self.objects:
            raise FileNotFoundError(path)
        return io.BytesIO(self.objects[path])


def test_gcs_requester_pays_reads_only_manifest_declared_objects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    local = build_fixture_bundle(tmp_path / "source")
    manifest_sha256 = hashlib.sha256((local / MANIFEST_NAME).read_bytes()).hexdigest()
    prefix = "jouvencekb/staging/viewer-bundles/reviewed-fixture"
    objects = {
        f"{prefix}/{path.relative_to(local).as_posix()}": path.read_bytes()
        for path in local.rglob("*")
        if path.is_file()
    }
    fake = FakeGCS(objects)
    captured: dict[str, object] = {}

    def fake_url_to_fs(uri: str, **kwargs):
        captured.update({"uri": uri, **kwargs})
        return fake, prefix

    monkeypatch.setattr("manage_db.viewer.bundle.url_to_fs", fake_url_to_fs)
    source = open_viewer_bundle(
        "gs://jouvencekb/staging/viewer-bundles/reviewed-fixture",
        billing_project="consumer-project",
        cache_root=tmp_path / "cache",
        expected_manifest_sha256=manifest_sha256,
    )

    assert source.mode == "gcs-requester-pays"
    assert captured == {
        "uri": "gs://jouvencekb/staging/viewer-bundles/reviewed-fixture",
        "project": "consumer-project",
        "requester_pays": "consumer-project",
        "token": "google_default",
    }
    assert fake.opened[0].endswith(MANIFEST_NAME)
    assert set(fake.opened[1:]) == {
        f"{prefix}/{relative}" for relative in source.manifest["files"]
    }
    assert all("../" not in path for path in fake.opened)

    browser_session = TestClient(
        create_app(data_source=source, session_token=TEST_TOKEN)
    ).get("/api/session", headers=LOCAL_HEADERS)
    assert browser_session.status_code == 200
    serialized = browser_session.text
    assert "consumer-project" not in serialized
    assert "google_default" not in serialized
    assert "credential" in serialized and "host-only" in serialized

    with pytest.raises(BundleError, match="does not match"):
        open_viewer_bundle(
            "gs://jouvencekb/staging/viewer-bundles/reviewed-fixture",
            billing_project="consumer-project",
            cache_root=tmp_path / "mismatch-cache",
            expected_manifest_sha256="0" * 64,
        )

    escape_cache = tmp_path / "escape-cache"
    outside_cache = tmp_path / "outside-cache"
    outside_cache.mkdir()
    manifest = source.manifest
    data_parent = (
        escape_cache
        / manifest["snapshot_id"]
        / manifest["bundle_checksum"]
        / "data"
    )
    data_parent.parent.mkdir(parents=True)
    data_parent.symlink_to(outside_cache, target_is_directory=True)
    with pytest.raises(BundleError, match="cache path escapes"):
        open_viewer_bundle(
            "gs://jouvencekb/staging/viewer-bundles/reviewed-fixture",
            billing_project="consumer-project",
            cache_root=escape_cache,
            expected_manifest_sha256=manifest_sha256,
        )

    lock_cache = tmp_path / "lock-cache"
    lock_cache.mkdir()
    outside_lock = tmp_path / "outside-lock"
    outside_lock.write_text("do-not-touch")
    (lock_cache / ".viewer-cache.lock").symlink_to(outside_lock)
    with pytest.raises(BundleError, match="private viewer cache lock"):
        open_viewer_bundle(
            "gs://jouvencekb/staging/viewer-bundles/reviewed-fixture",
            billing_project="consumer-project",
            cache_root=lock_cache,
            expected_manifest_sha256=manifest_sha256,
        )
    assert outside_lock.read_text() == "do-not-touch"

    monkeypatch.setattr("manage_db.viewer.bundle.MAX_CACHE_BYTES", 1)
    with pytest.raises(BundleError, match="aggregate 2 GB safety bound"):
        open_viewer_bundle(
            "gs://jouvencekb/staging/viewer-bundles/reviewed-fixture",
            billing_project="consumer-project",
            cache_root=tmp_path / "quota-cache",
            expected_manifest_sha256=manifest_sha256,
        )

    monkeypatch.setattr("manage_db.viewer.bundle.MAX_CACHE_BYTES", 2 * 1024 * 1024 * 1024)
    monkeypatch.setattr("manage_db.viewer.bundle.MAX_CACHE_FILES", 2)
    with pytest.raises(BundleError, match="file-count safety bound"):
        open_viewer_bundle(
            "gs://jouvencekb/staging/viewer-bundles/reviewed-fixture",
            billing_project="consumer-project",
            cache_root=tmp_path / "file-count-cache",
            expected_manifest_sha256=manifest_sha256,
        )


def test_gcs_cache_lock_does_not_exceed_file_count_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    local = build_fixture_bundle(tmp_path / "source")
    manifest_sha256 = hashlib.sha256((local / MANIFEST_NAME).read_bytes()).hexdigest()
    prefix = "jouvencekb/staging/viewer-bundles/reviewed-fixture"
    objects = {
        f"{prefix}/{path.relative_to(local).as_posix()}": path.read_bytes()
        for path in local.rglob("*")
        if path.is_file()
    }
    monkeypatch.setattr(
        "manage_db.viewer.bundle.url_to_fs",
        lambda *_args, **_kwargs: (FakeGCS(objects), prefix),
    )
    monkeypatch.setattr("manage_db.viewer.bundle.MAX_CACHE_FILES", 2)
    cache_root = tmp_path / "full-cache"
    cache_root.mkdir()
    (cache_root / "existing-a").write_bytes(b"a")
    (cache_root / "existing-b").write_bytes(b"b")

    with pytest.raises(BundleError, match="file-count safety bound"):
        open_viewer_bundle(
            "gs://jouvencekb/staging/viewer-bundles/reviewed-fixture",
            billing_project="consumer-project",
            cache_root=cache_root,
            expected_manifest_sha256=manifest_sha256,
        )

    assert sorted(path.name for path in cache_root.iterdir()) == [
        "existing-a",
        "existing-b",
    ]


def test_gcs_requires_consumer_billing_project(tmp_path: Path) -> None:
    with pytest.raises(BundleError, match="--billing-project"):
        open_viewer_bundle(
            "gs://jouvencekb/staging/viewer-bundles/reviewed",
            cache_root=tmp_path / "cache",
        )
    for unsafe_root in (
        "gs://jouvencekb/main",
        "gs://jouvencekb/staging/viewer-bundles/../edges",
        "gs://other-bucket/kg/v2/viewer-bundles/reviewed",
    ):
        with pytest.raises(BundleError, match="reviewed viewer bundle"):
            open_viewer_bundle(
                unsafe_root,
                billing_project="consumer-project",
                cache_root=tmp_path / "cache",
            )
    with pytest.raises(BundleError, match="--manifest-sha256"):
        open_viewer_bundle(
            "gs://jouvencekb/staging/viewer-bundles/reviewed",
            billing_project="consumer-project",
            cache_root=tmp_path / "cache",
        )


def test_gcs_adapter_initialization_error_is_sanitized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    credential_path = "/private/user/application_default_credentials.json"

    def fail_adapter(*_args, **_kwargs):
        raise RuntimeError(f"credential failed at {credential_path}")

    monkeypatch.setattr("manage_db.viewer.bundle.url_to_fs", fail_adapter)
    with pytest.raises(BundleError) as error:
        open_viewer_bundle(
            "gs://jouvencekb/staging/viewer-bundles/reviewed",
            billing_project="consumer-project",
            cache_root=tmp_path / "cache",
            expected_manifest_sha256="0" * 64,
        )
    assert credential_path not in str(error.value)
    assert "ADC" in str(error.value)


def test_bundle_rejects_malformed_sidecar_rows(tmp_path: Path) -> None:
    bundle_root = build_fixture_bundle(tmp_path / "bundle")
    manifest_path = bundle_root / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text())
    malformed = b"[{}]\n"
    sidecar = bundle_root / "data" / "features.json"
    sidecar.write_bytes(malformed)
    manifest["files"]["data/features.json"].update(
        {"size": len(malformed), "sha256": hashlib.sha256(malformed).hexdigest()}
    )
    manifest["bundle_checksum"] = _bundle_checksum(manifest)
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(BundleError, match="invalid feature row"):
        open_viewer_bundle(str(bundle_root))

    bundle_root = build_fixture_bundle(tmp_path / "nonfinite-bundle")
    manifest_path = bundle_root / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text())
    sidecar = bundle_root / "data" / "edges.json"
    rows = json.loads(sidecar.read_text())
    rows[0]["score"] = float("nan")
    malformed = json.dumps(rows).encode()
    sidecar.write_bytes(malformed)
    manifest["files"]["data/edges.json"].update(
        {"size": len(malformed), "sha256": hashlib.sha256(malformed).hexdigest()}
    )
    manifest["bundle_checksum"] = _bundle_checksum(manifest)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(BundleError, match="invalid edge row"):
        open_viewer_bundle(str(bundle_root))

    bundle_root = build_fixture_bundle(tmp_path / "extra-field-bundle")
    manifest_path = bundle_root / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text())
    sidecar = bundle_root / "data" / "nodes.json"
    rows = json.loads(sidecar.read_text())
    rows[0]["undeclared_nested_payload"] = {"value": "x" * 100_000}
    malformed = json.dumps(rows).encode()
    sidecar.write_bytes(malformed)
    manifest["files"]["data/nodes.json"].update(
        {"size": len(malformed), "sha256": hashlib.sha256(malformed).hexdigest()}
    )
    manifest["bundle_checksum"] = _bundle_checksum(manifest)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(BundleError, match="invalid node row"):
        open_viewer_bundle(str(bundle_root))

    bundle_root = build_fixture_bundle(tmp_path / "nested-attribute-bundle")
    manifest_path = bundle_root / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text())
    sidecar = bundle_root / "data" / "nodes.json"
    rows = json.loads(sidecar.read_text())
    rows[0]["attributes"]["nested"] = {"value": "not allowed"}
    malformed = json.dumps(rows).encode()
    sidecar.write_bytes(malformed)
    manifest["files"]["data/nodes.json"].update(
        {"size": len(malformed), "sha256": hashlib.sha256(malformed).hexdigest()}
    )
    manifest["bundle_checksum"] = _bundle_checksum(manifest)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(BundleError, match="invalid node row"):
        open_viewer_bundle(str(bundle_root))


@pytest.mark.parametrize(
    ("relative", "mutate", "message"),
    [
        (
            "data/edges.json",
            lambda row: row.update({"undeclared_response_field": "x"}),
            "invalid edge row",
        ),
        (
            "data/nodes.json",
            lambda row: row["attributes"].update(
                {f"attribute-{index}": "x" for index in range(101)}
            ),
            "invalid node row",
        ),
        (
            "data/nodes.json",
            lambda row: row["attributes"].update({"bounded": "x" * 4097}),
            "invalid node row",
        ),
        (
            "data/nodes.json",
            lambda row: row["aliases"].extend(
                [
                    {"kind": "alias", "value": "x" * 40_000, "source": "test"},
                    {"kind": "alias", "value": "y" * 40_000, "source": "test"},
                ]
            ),
            "invalid node row",
        ),
    ],
)
def test_bundle_rejects_unbounded_or_undeclared_row_content(
    tmp_path: Path,
    relative: str,
    mutate,
    message: str,
) -> None:
    bundle_root = build_fixture_bundle(tmp_path / "bundle")
    manifest_path = bundle_root / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text())
    sidecar = bundle_root / relative
    rows = json.loads(sidecar.read_text())
    mutate(rows[0])
    payload = json.dumps(rows).encode()
    sidecar.write_bytes(payload)
    manifest["files"][relative].update(
        {"size": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
    )
    manifest["bundle_checksum"] = _bundle_checksum(manifest)
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(BundleError, match=message):
        open_viewer_bundle(str(bundle_root))


def test_token_host_origin_and_error_sanitization() -> None:
    with pytest.raises(ValueError, match="session token"):
        create_app()
    client = TestClient(create_app(session_token="test-secret"))
    local = {"host": "127.0.0.1:8765"}
    assert client.get("/api/session", headers=local).status_code == 401
    assert client.get(
        "/api/session", headers={**local, "x-jouvence-session": "test-secret"}
    ).status_code == 200
    assert client.get(
        "/api/session",
        headers={
            "x-jouvence-session": "test-secret",
            "host": "attacker.example",
        },
    ).status_code == 400
    assert client.get(
        "/api/session",
        headers={
            **local,
            "x-jouvence-session": "test-secret",
            "origin": "https://attacker.example",
        },
    ).status_code == 403
    assert client.get(
        "/api/session",
        headers={
            **local,
            "x-jouvence-session": "test-secret",
            "origin": "http://127.0.0.1.attacker.example:8765",
        },
    ).status_code == 403
    assert client.get(
        "/api/session",
        headers={
            **local,
            "x-jouvence-session": "test-secret",
            "origin": "http://127.0.0.1:8765",
        },
    ).status_code == 200

    assert client.get(
        "/api/files",
        params={"path": "gs://other-bucket/private"},
        headers={**local, "x-jouvence-session": "test-secret"},
    ).status_code == 404
    assert client.post(
        "/api/session/connect",
        json={"data_root": "/etc", "billing_project": "maintainer-project"},
        headers={**local, "x-jouvence-session": "test-secret"},
    ).status_code == 404
