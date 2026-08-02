from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from manage_db import kg_storage
from manage_db.build_lamindb_kg_manifest import build_manifest
from manage_db.sync_kg_artifacts_to_lamindb import (
    _ensure_collection,
    _lamin_feature_dtype,
    _sync_one_artifact,
    list_registered_metadata,
    sync_manifest_to_lamindb,
)


def _write_parquet(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def _tiny_kg(tmp_path: Path) -> Path:
    root_path = tmp_path / "kg"
    root = kg_storage.open_kg_root(str(root_path))
    kg_storage.write_nodes(
        root,
        "gene",
        pd.DataFrame([{"id": "ENSG00000139618", "ncbi_gene_id": "675", "hgnc_id": "HGNC:1101", "uniprot_id": "P51587", "gene_name": "BRCA2"}]),
    )
    kg_storage.write_edges(
        root,
        "gene_interacts_gene",
        pd.DataFrame([
            {
                "x_id": "ENSG00000139618",
                "x_type": "gene",
                "y_id": "ENSG00000141510",
                "y_type": "gene",
                "relation": "gene_interacts_gene",
                "display_relation": "interacts",
                "source": "test",
                "credibility": 1,
            }
        ]),
    )
    _write_parquet(
        root_path / "evidence" / "gene_interacts_gene.parquet",
        [{"relation": "gene_interacts_gene", "x_id": "ENSG00000139618", "y_id": "ENSG00000141510", "source_dataset": "unit"}],
    )
    _write_parquet(
        root_path / "features" / "gene_textual_summary.parquet",
        [{"node_id": "ENSG00000139618", "node_type": "gene", "summary": "BRCA2 summary", "source": "unit"}],
    )
    return root_path


def test_build_manifest_covers_nodes_edges_evidence_and_features(tmp_path: Path) -> None:
    kg_path = _tiny_kg(tmp_path)

    manifest = build_manifest(kg_path).to_dict()

    keys = {entry["key"] for entry in manifest["layers"]}
    assert "main/nodes/gene.parquet" in keys
    assert "main/edges/gene_interacts_gene.parquet" in keys
    assert "main/evidence/gene_interacts_gene.parquet" in keys
    assert "main/features/gene_textual_summary.parquet" in keys
    assert manifest["summary"]["counts_by_layer"] == {
        "nodes": 1,
        "edges": 1,
        "edges_inferred": 0,
        "evidence": 1,
        "evidence_inferred": 0,
        "features": 1,
        "embeddings": 0,
    }

    gene = next(entry for entry in manifest["layers"] if entry["key"] == "main/nodes/gene.parquet")
    assert gene["rows"] == 1
    assert "id" in gene["columns"]
    assert "node_type:gene" in gene["labels"]
    assert gene["metadata_fingerprint"]

    edge = next(entry for entry in manifest["layers"] if entry["key"] == "main/edges/gene_interacts_gene.parquet")
    assert edge["metadata"]["x_type"] == "gene"
    assert edge["metadata"]["y_type"] == "gene"
    assert "relation:gene_interacts_gene" in edge["labels"]


def test_build_manifest_can_record_public_bucket_uris_from_local_scan(tmp_path: Path) -> None:
    kg_path = _tiny_kg(tmp_path)

    manifest = build_manifest(kg_path, public_root="gs://jouvencekb/main").to_dict()

    assert manifest["scan_root"] == str(kg_path)
    assert manifest["canonical_root"] == "gs://jouvencekb/main"
    assert all(entry["uri"].startswith("gs://jouvencekb/main/") for entry in manifest["layers"])


def test_list_registered_metadata_smoke_view_has_all_layers(tmp_path: Path) -> None:
    manifest = build_manifest(_tiny_kg(tmp_path)).to_dict()

    view = list_registered_metadata(manifest)

    assert [item["name"] for item in view["nodes"]] == ["gene"]
    assert [item["name"] for item in view["edges"]] == ["gene_interacts_gene"]
    assert [item["name"] for item in view["evidence"]] == ["gene_interacts_gene"]
    assert [item["name"] for item in view["features"]] == ["gene_textual_summary"]
    assert view["nodes"][0]["rows"] == 1


def test_sync_manifest_dry_run_is_credential_free(tmp_path: Path) -> None:
    manifest = build_manifest(_tiny_kg(tmp_path)).to_dict()

    report = sync_manifest_to_lamindb(manifest, dry_run=True).to_dict()

    assert report["dry_run"] is True
    assert report["summary"]["would_create"] == 4
    assert report["summary"]["created"] == 0
    assert report["errors"] == []


class _Manager:
    def __init__(self) -> None:
        self.records = []

    def add(self, *records) -> None:
        self.records.extend(records)


class _QuerySet(list):
    def one_or_none(self):
        return self[0] if self else None

    def first(self):
        return self[0] if self else None


class _BaseRecord:
    _records: list

    def __init_subclass__(cls) -> None:
        cls._records = []

    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.ulabels = _Manager()
        self.features = _Manager()
        self.artifacts = _Manager()

    @classmethod
    def filter(cls, **kwargs):
        return _QuerySet([
            record for record in cls._records
            if all(getattr(record, key, None) == value for key, value in kwargs.items())
        ])

    def save(self):
        if self not in self.__class__._records:
            self.__class__._records.append(self)
        return self


class FakeArtifact(_BaseRecord):
    pass


class FakeULabel(_BaseRecord):
    pass


class FakeFeature(_BaseRecord):
    pass


class FakeCollection(_BaseRecord):
    pass


class FakeLn:
    Artifact = FakeArtifact
    ULabel = FakeULabel
    Feature = FakeFeature
    Collection = FakeCollection


def test_lamin_feature_dtype_maps_pyarrow_types() -> None:
    assert _lamin_feature_dtype("string") == "str"
    assert _lamin_feature_dtype("int64") == "int"
    assert _lamin_feature_dtype("double") == "float"
    assert _lamin_feature_dtype("bool") == "bool"
    assert _lamin_feature_dtype("timestamp[us]") == "datetime"
    assert _lamin_feature_dtype("list<item: string>") == "object"


def test_ensure_collection_uses_lamindb_key_field() -> None:
    collection = _ensure_collection(FakeLn, "jouvence-kg/v2", dry_run=False)

    assert collection is not None
    assert collection.key == "jouvence-kg/v2"
    assert FakeCollection.filter(key="jouvence-kg/v2").one_or_none() is collection


def test_sync_one_artifact_is_idempotent_with_existing_fingerprint(tmp_path: Path) -> None:
    manifest = build_manifest(_tiny_kg(tmp_path)).to_dict()
    entry = next(item for item in manifest["layers"] if item["key"] == "main/nodes/gene.parquet")
    existing = FakeArtifact(entry["uri"], key=entry["key"], description=f"metadata_fingerprint {entry['metadata_fingerprint']}").save()
    collection = FakeCollection(name="jouvence-kg/v2").save()

    result = _sync_one_artifact(FakeLn, entry, collection, dry_run=False)

    assert result.status == "noop"
    assert FakeArtifact.filter(key=entry["key"]).one_or_none() is existing
    assert collection.artifacts.records == [existing]


def test_sync_one_artifact_updates_changed_metadata_and_attaches_labels(tmp_path: Path) -> None:
    manifest = build_manifest(_tiny_kg(tmp_path)).to_dict()
    entry = next(item for item in manifest["layers"] if item["key"] == "main/features/gene_textual_summary.parquet")
    existing = FakeArtifact(entry["uri"], key=entry["key"], description="old fingerprint").save()
    collection = FakeCollection(name="jouvence-kg/v2").save()

    result = _sync_one_artifact(FakeLn, entry, collection, dry_run=False)

    assert result.status == "updated"
    assert entry["metadata_fingerprint"] in existing.description
    label_names = {label.name for label in existing.ulabels.records}
    assert "kg-layer:features" in label_names
    assert "feature" in label_names
    assert collection.artifacts.records == [existing]
