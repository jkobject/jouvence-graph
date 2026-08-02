from __future__ import annotations

from types import SimpleNamespace

import scripts.check_storage_layout_contract as contract


class _BlobIterator:
    def __init__(self, *, objects: set[str], prefixes: set[str]) -> None:
        self._objects = objects
        self.prefixes = prefixes

    def __iter__(self):
        return iter(SimpleNamespace(name=name) for name in self._objects)


class _Client:
    def __init__(self, *, objects: set[str], prefixes: set[str]) -> None:
        self._objects = objects
        self._prefixes = prefixes
        self._bucket = object()

    def bucket(self, bucket: str):
        assert bucket == "jouvencekb"
        return self._bucket

    def list_blobs(self, bucket, *, delimiter: str, max_results: int):
        assert bucket is self._bucket
        assert delimiter == "/"
        assert max_results == 1000
        return _BlobIterator(objects=self._objects, prefixes=self._prefixes)


def test_pyg_and_empty_staging_are_optional_gcs_prefixes(monkeypatch) -> None:
    assert "pyg/" in contract.OPTIONAL_TOP_LEVEL_PREFIXES
    assert "staging/" in contract.OPTIONAL_TOP_LEVEL_PREFIXES
    assert "pyg/" not in contract.REQUIRED_TOP_LEVEL_PREFIXES

    client = _Client(
        objects={"README.md"},
        prefixes={".lamin/", "raw/", "main/"},
    )
    monkeypatch.setattr("google.cloud.storage.Client", lambda: client)

    assert contract.check_live() == []


def test_legacy_prefixes_remain_rejected(monkeypatch) -> None:
    client = _Client(
        objects={"README.md"},
        prefixes={".lamin/", "raw/", "main/", "kg/"},
    )
    monkeypatch.setattr("google.cloud.storage.Client", lambda: client)

    assert "unexpected root prefixes: ['kg/']" in contract.check_live()
