from __future__ import annotations

import json
import shutil
from pathlib import Path

import gcsfs
import pytest

import scripts.parquet_catalog as parquet_catalog
from scripts.parquet_catalog import CATALOG_DIR, INVENTORY_PATH, schema_hash


REPO_ROOT = Path(__file__).resolve().parents[1]


def _inventory() -> dict:
    return json.loads((REPO_ROOT / INVENTORY_PATH).read_text())


def test_catalog_covers_every_logical_dataset() -> None:
    inventory = _inventory()
    datasets = inventory["datasets"]
    ids = [dataset["id"] for dataset in datasets]
    pages = {path.stem for path in (REPO_ROOT / CATALOG_DIR / "datasets").glob("*.md")}

    assert len(ids) == len(set(ids)) == inventory["dataset_count"] == 110
    assert inventory["documented_count"] == 110
    assert inventory["undocumented_count"] == 0
    assert pages == set(ids)


def test_every_schema_hash_reproduces() -> None:
    for dataset in _inventory()["datasets"]:
        assert dataset["schema_hash"] == schema_hash(dataset["fields"]), dataset["id"]


def test_text_embedding_leaves_use_per_node_type_denominators() -> None:
    leaves = [
        dataset
        for dataset in _inventory()["datasets"]
        if dataset["id"].startswith("embedding__") and "_text_" in dataset["id"]
    ]

    assert len(leaves) == 9
    assert sum(dataset["rows"] for dataset in leaves) == 490_336
    assert {dataset["semantics"]["node_type"] for dataset in leaves} == {
        "cell_line",
        "cell_type",
        "disease",
        "gene",
        "molecule",
        "pathway",
        "phenotype",
        "protein",
        "tissue",
    }


def test_remap_is_one_compacted_logical_dataset() -> None:
    datasets = {dataset["id"]: dataset for dataset in _inventory()["datasets"]}
    remap = datasets["features__remap_crm_tf_enhancer_support"]

    assert len(remap["objects"]) == 1
    assert remap["rows"] == 48_768_788
    assert remap["uri"] == "gs://jouvencekb/main/features/remap_crm_tf_enhancer_support.parquet"


def test_catalog_has_no_private_local_paths() -> None:
    paths = [
        REPO_ROOT / INVENTORY_PATH,
        REPO_ROOT / CATALOG_DIR / "embedding-release-inventory.json",
        REPO_ROOT / CATALOG_DIR / "index.md",
        *(REPO_ROOT / CATALOG_DIR / "datasets").glob("*.md"),
    ]
    forbidden = ("/Users/", "/home/ubuntu", ".omoc/")
    for path in paths:
        text = path.read_text()
        assert not any(token in text for token in forbidden), path


def test_check_failure_is_byte_for_byte_read_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    catalog_dir = tmp_path / "parquet-catalog"
    shutil.copytree(REPO_ROOT / CATALOG_DIR, catalog_dir)
    stale_page = next((catalog_dir / "datasets").glob("*.md"))
    stale_page.write_bytes(stale_page.read_bytes() + b"\nstale\n")
    before = {
        path.relative_to(catalog_dir): path.read_bytes()
        for path in catalog_dir.rglob("*")
        if path.is_file()
    }
    monkeypatch.setattr(parquet_catalog, "CATALOG_DIR", catalog_dir)
    monkeypatch.setattr(parquet_catalog, "INVENTORY_PATH", catalog_dir / "inventory.json")

    with pytest.raises(SystemExit, match="stale generated page"):
        parquet_catalog.check()

    after = {
        path.relative_to(catalog_dir): path.read_bytes()
        for path in catalog_dir.rglob("*")
        if path.is_file()
    }
    assert after == before


def test_check_missing_index_fails_cleanly_without_mutation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    catalog_dir = tmp_path / "parquet-catalog"
    shutil.copytree(REPO_ROOT / CATALOG_DIR, catalog_dir)
    (catalog_dir / "index.md").unlink()
    before = {
        path.relative_to(catalog_dir): path.read_bytes()
        for path in catalog_dir.rglob("*")
        if path.is_file()
    }
    monkeypatch.setattr(parquet_catalog, "CATALOG_DIR", catalog_dir)
    monkeypatch.setattr(parquet_catalog, "INVENTORY_PATH", catalog_dir / "inventory.json")

    with pytest.raises(SystemExit, match="stale generated index"):
        parquet_catalog.check()

    after = {
        path.relative_to(catalog_dir): path.read_bytes()
        for path in catalog_dir.rglob("*")
        if path.is_file()
    }
    assert after == before


def test_generated_key_claims_are_truthful_and_evidence_uses_edge_linkage() -> None:
    pages = (REPO_ROOT / CATALOG_DIR / "datasets").glob("*.md")
    text_by_name = {path.name: path.read_text() for path in pages}

    assert all("Primary/unique key" not in text for text in text_by_name.values())
    evidence = [text for name, text in text_by_name.items() if name.startswith("evidence__")]
    assert evidence
    assert all("uniqueness unvalidated" in text for text in evidence)
    assert all("(relation, x_id, y_id)" in text or "edge_key" in text for text in evidence)


def test_read_examples_are_dataset_scoped_and_planned_pages_are_non_executable() -> None:
    pages = (REPO_ROOT / CATALOG_DIR / "datasets").glob("*.md")
    for path in pages:
        text = path.read_text()
        assert "read_parquet('./*.parquet')" not in text, path.name
        if path.name.startswith("planned_embedding__"):
            assert "Current access is unavailable" in text, path.name
            assert "NON-EXECUTABLE POST-PUBLICATION TEMPLATE" in text, path.name
            assert "gcloud storage cp" not in text, path.name
            assert "read_parquet(" not in text, path.name
        else:
            assert f"./parquet-catalog-data/{path.stem}/" in text, path.name
            assert "rm -rf -- \"$LOCAL_DIR\"" in text, path.name
            assert "paths = sorted(fs.glob(" in text, path.name
            assert " ORDER BY " in text, path.name


def test_live_refresh_supplies_requester_pays_billing_project(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def fake_filesystem(**kwargs: object) -> object:
        calls.append(kwargs)
        return object()

    monkeypatch.setattr(gcsfs, "GCSFileSystem", fake_filesystem)
    monkeypatch.setattr(parquet_catalog, "_logical_groups", lambda _fs: [])

    parquet_catalog.refresh(None, write=False)

    assert calls == [
        {
            "project": parquet_catalog.PROJECT,
            "requester_pays": parquet_catalog.PROJECT,
        }
    ]


def test_generic_sharded_uri_pattern_preserves_gcs_scheme() -> None:
    objects = [
        {"uri": "gs://jouvencekb/main/features/example/part-00000.parquet"},
        {"uri": "gs://jouvencekb/main/features/example/part-00001.parquet"},
    ]

    assert parquet_catalog._uri_pattern(objects) == (
        "gs://jouvencekb/main/features/example/part-*.parquet"
    )
