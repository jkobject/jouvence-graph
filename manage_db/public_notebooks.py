"""Bounded, read-only helpers used by the public Jouvence notebooks.

The helpers keep access policy and memory bounds in one tested place.  They do
not write to the canonical KG or LaminDB.  The only writers create disposable
local fixtures and sampled PyG exports chosen by the caller.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from fsspec.core import url_to_fs

from . import kg_evidence, kg_storage

PUBLIC_KG_ROOT = "gs://jouvencekb/main"
LAMIN_INSTANCE = "jkobject/jouvencekb"
DEFAULT_SAMPLE_ROWS = 100
MAX_SAMPLE_ROWS = 10_000


def _validated_limit(limit: int) -> int:
    value = int(limit)
    if not 1 <= value <= MAX_SAMPLE_ROWS:
        raise ValueError(f"limit must be between 1 and {MAX_SAMPLE_ROWS}")
    return value


def _storage_options(uri: str, billing_project: str | None) -> dict[str, Any]:
    if not uri.startswith("gs://"):
        return {}
    project = billing_project or os.environ.get("JOUVENCE_BILLING_PROJECT")
    if not project:
        raise ValueError(
            "GCS reads require billing_project=... or JOUVENCE_BILLING_PROJECT; "
            "no project-specific default is embedded in Jouvence."
        )
    return {
        "requester_pays": project,
        "token": "google_default",
    }


def read_bounded_parquet(
    uri: str | Path,
    *,
    columns: Iterable[str] | None = None,
    limit: int = DEFAULT_SAMPLE_ROWS,
    offset: int = 0,
    billing_project: str | None = None,
) -> pd.DataFrame:
    """Read at most ``limit`` rows, seeking past complete Parquet row groups.

    Unlike ``pandas.read_parquet(...).head()``, this does not materialize the
    whole object.  A partial selected row group is streamed in bounded batches.
    """

    limit = _validated_limit(limit)
    if int(offset) < 0:
        raise ValueError("offset must be non-negative")

    uri_text = str(uri)
    fs, path = url_to_fs(uri_text, **_storage_options(uri_text, billing_project))
    parquet = pq.ParquetFile(path, filesystem=fs)
    selected = list(columns) if columns is not None else None
    remaining_skip = int(offset)
    remaining_take = limit
    batches: list[pa.RecordBatch] = []

    for row_group_index in range(parquet.metadata.num_row_groups):
        row_group_rows = parquet.metadata.row_group(row_group_index).num_rows
        if remaining_skip >= row_group_rows:
            remaining_skip -= row_group_rows
            continue
        batch_size = min(max(remaining_take + remaining_skip, 1), 65_536)
        for batch in parquet.iter_batches(
            row_groups=[row_group_index], columns=selected, batch_size=batch_size
        ):
            if remaining_skip:
                skipped = min(remaining_skip, batch.num_rows)
                batch = batch.slice(skipped)
                remaining_skip -= skipped
            if batch.num_rows == 0:
                continue
            take = min(remaining_take, batch.num_rows)
            batches.append(batch.slice(0, take))
            remaining_take -= take
            if remaining_take == 0:
                return pa.Table.from_batches(batches).to_pandas()
    if batches:
        return pa.Table.from_batches(batches).to_pandas()

    schema = parquet.schema_arrow
    if selected is not None:
        schema = pa.schema([schema.field(name) for name in selected if name in schema.names])
    return pa.Table.from_batches([], schema=schema).to_pandas()


def parquet_catalog(
    kg_root: str | Path,
    *,
    billing_project: str | None = None,
) -> pd.DataFrame:
    """Return a footer-only catalog for node, edge, evidence and feature Parquets."""

    root_uri = str(kg_root).rstrip("/")
    fs, root_path = url_to_fs(root_uri, **_storage_options(root_uri, billing_project))
    rows: list[dict[str, Any]] = []
    for layer in ("nodes", "edges", "evidence", "features"):
        pattern = f"{root_path.rstrip('/')}/{layer}/**/*.parquet"
        for path in sorted(fs.glob(pattern)):
            parquet = pq.ParquetFile(path, filesystem=fs)
            relative = path.removeprefix(root_path.rstrip("/") + "/")
            rows.append(
                {
                    "layer": layer,
                    "path": relative,
                    "rows": int(parquet.metadata.num_rows),
                    "row_groups": int(parquet.metadata.num_row_groups),
                    "columns": list(parquet.schema_arrow.names),
                }
            )
    return pd.DataFrame(rows, columns=["layer", "path", "rows", "row_groups", "columns"])


def bounded_edge_evidence_join(
    kg_root: str | Path,
    relation: str,
    *,
    edge_limit: int = 100,
    evidence_limit: int = 1_000,
    billing_project: str | None = None,
) -> pd.DataFrame:
    """Join two bounded prefixes of one assertion and evidence table.

    This deliberately reads selected columns from at most one or a few Parquet
    row groups in each object.  It is an inspection helper, not a completeness
    query: missing rows in the result do not mean that evidence is absent.
    """

    edge_limit = _validated_limit(edge_limit)
    evidence_limit = _validated_limit(evidence_limit)
    root = str(kg_root).rstrip("/")
    keys = ["relation", "x_id", "x_type", "y_id", "y_type"]
    edges = read_bounded_parquet(
        f"{root}/edges/{relation}.parquet",
        columns=[*keys, "source"],
        limit=edge_limit,
        billing_project=billing_project,
    )
    evidence = read_bounded_parquet(
        f"{root}/evidence/{relation}.parquet",
        columns=[*keys, "source", "source_record_id"],
        limit=evidence_limit,
        billing_project=billing_project,
    )
    return edges.merge(
        evidence,
        on=keys,
        how="inner",
        suffixes=("_assertion", "_evidence"),
        validate="one_to_many",
    )


def diseases_with_gene_evidence(
    kg_root: str | Path,
    gene_id: str,
    *,
    limit: int = 20,
) -> pd.DataFrame:
    """Answer a bounded biological question by joining assertions and evidence."""

    limit = _validated_limit(limit)
    root = Path(kg_root)
    edges = root / "edges" / "disease_associated_gene.parquet"
    diseases = root / "nodes" / "disease.parquet"
    evidence = root / "evidence" / "disease_associated_gene.parquet"
    for path in (edges, diseases, evidence):
        if not path.exists():
            raise FileNotFoundError(path)
    con = duckdb.connect(database=":memory:")
    con.execute("PRAGMA threads=2")
    return con.execute(
        """
        SELECT e.x_id AS gene_id,
               e.y_id AS disease_id,
               d.name AS disease_name,
               e.source AS assertion_source,
               count(ev.source_record_id)::BIGINT AS evidence_rows,
               string_agg(DISTINCT ev.source, '; ' ORDER BY ev.source) AS evidence_sources,
               max(ev.evidence_score) AS max_evidence_score
        FROM read_parquet(?) e
        JOIN read_parquet(?) d ON d.id = e.y_id
        LEFT JOIN read_parquet(?) ev USING (relation, x_id, x_type, y_id, y_type)
        WHERE e.relation = 'disease_associated_gene' AND e.x_id = ?
        GROUP BY ALL
        ORDER BY evidence_rows DESC, disease_id
        LIMIT ?
        """,
        [str(edges), str(diseases), str(evidence), str(gene_id), limit],
    ).fetchdf()


def nearest_embeddings(
    embedding_uri: str | Path,
    query_node_id: str,
    *,
    limit: int = 5,
    billing_project: str | None = None,
) -> pd.DataFrame:
    """Return cosine-nearest rows from a deliberately bounded embedding table."""

    limit = _validated_limit(limit)
    frame = read_bounded_parquet(
        embedding_uri,
        columns=["node_id", "node_type", "embedding", "embedding_model"],
        limit=MAX_SAMPLE_ROWS,
        billing_project=billing_project,
    )
    match = frame.loc[frame["node_id"].astype(str) == str(query_node_id)]
    if match.empty:
        raise KeyError(f"query node not present in bounded embedding table: {query_node_id}")
    matrix = np.vstack(frame["embedding"].map(lambda value: np.asarray(value, dtype=np.float32)))
    query = np.asarray(match.iloc[0]["embedding"], dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1) * max(float(np.linalg.norm(query)), 1e-12)
    scores = np.divide(matrix @ query, norms, out=np.zeros(len(matrix), dtype=np.float32), where=norms > 0)
    result = frame[["node_id", "node_type", "embedding_model"]].copy()
    result["cosine_similarity"] = scores
    result = result[result["node_id"].astype(str) != str(query_node_id)]
    return result.sort_values(["cosine_similarity", "node_id"], ascending=[False, True]).head(limit).reset_index(drop=True)


def _connect_exact_lamin(instance: str):
    if instance != LAMIN_INSTANCE:
        raise ValueError(f"public examples only connect to exact instance {LAMIN_INSTANCE!r}")
    import lamindb as ln

    current = getattr(getattr(ln.setup.settings, "instance", None), "slug", None)
    if current != instance:
        ln.connect(instance)
    current = getattr(getattr(ln.setup.settings, "instance", None), "slug", None)
    if current != instance:
        raise RuntimeError(f"connected LaminDB instance is {current!r}, expected {instance!r}")
    import lnschema_txgnn as txs

    return txs


def query_lamindb_node(
    node_type: str,
    node_id: str,
    *,
    limit: int = 20,
    instance: str = LAMIN_INSTANCE,
) -> pd.DataFrame:
    """Query an exact-ID Jouvence node registry without writing."""

    registry_keys = {
        "gene": ("Gene", "ensembl_gene_id"),
        "disease": ("Disease", "ontology_id"),
        "molecule": ("Molecule", "chembl_id"),
        "pathway": ("Pathway", "ontology_id"),
        "tissue": ("Tissue", "ontology_id"),
        "cell_type": ("CellType", "ontology_id"),
        "protein": ("Protein", "ensembl_protein_id"),
        "transcript": ("Transcript", "ensembl_transcript_id"),
    }
    if node_type not in registry_keys:
        raise ValueError(f"unsupported exact-ID public node query: {node_type!r}")
    limit = _validated_limit(limit)
    txs = _connect_exact_lamin(instance)
    registry_name, key = registry_keys[node_type]
    registry = getattr(txs, registry_name)
    return registry.objects.filter(**{key: node_id}).order_by(key)[:limit].df()


def query_lamindb_edges(
    *,
    relation: str,
    x_id: str | None = None,
    y_id: str | None = None,
    limit: int = 20,
    instance: str = LAMIN_INSTANCE,
) -> pd.DataFrame:
    """Query the exact Jouvence LaminDB edge registry without writing.

    The current instance is intentionally partial; an empty result is not
    evidence that the assertion is absent from canonical Parquet.
    """

    limit = _validated_limit(limit)
    txs = _connect_exact_lamin(instance)

    filters: dict[str, Any] = {"relation": relation}
    if x_id is not None:
        filters["x_id"] = x_id
    if y_id is not None:
        filters["y_id"] = y_id
    queryset = txs.KGEdge.objects.filter(**filters).order_by("edge_key")[:limit]
    return queryset.df()


def query_lamindb_evidence(
    *,
    relation: str,
    x_id: str | None = None,
    y_id: str | None = None,
    limit: int = 20,
    instance: str = LAMIN_INSTANCE,
) -> pd.DataFrame:
    """Query source-specific support rows from the partial LaminDB mirror."""

    limit = _validated_limit(limit)
    txs = _connect_exact_lamin(instance)
    filters: dict[str, Any] = {"relation": relation}
    if x_id is not None:
        filters["x_id"] = x_id
    if y_id is not None:
        filters["y_id"] = y_id
    queryset = txs.KGEdgeEvidence.objects.filter(**filters).order_by("evidence_key")[:limit]
    return queryset.df()


def build_sampled_pyg(
    kg_root: str | Path,
    output_root: str | Path,
    *,
    max_nodes_per_type: int = 100,
    max_edges_per_relation: int = 200,
):
    """Delegate a bounded graph build to the repository's tested exporter."""

    from .build_pyg_export import BuildConfig, build_pyg_export

    root = Path(kg_root)
    embedding_root = root / "features"
    max_nodes_per_type = _validated_limit(max_nodes_per_type)
    max_edges_per_relation = _validated_limit(max_edges_per_relation)
    return build_pyg_export(
        BuildConfig(
            kg_root=str(root),
            output_root=str(output_root),
            node_types=("gene", "disease", "molecule"),
            relations=("disease_associated_gene", "molecule_targets_gene"),
            max_nodes_per_type=max_nodes_per_type,
            max_edges_per_relation=max_edges_per_relation,
            include_reverse_edges=True,
            strict=True,
            build_name="public-notebook-sample",
            embedding_features_root=str(embedding_root),
            fallback_seed=20260716,
        )
    )


def run_sampled_ml(export_root: str | Path, *, seed: int = 13) -> dict[str, Any]:
    """Run the existing deterministic link-prediction smoke on a sampled export."""

    from dataclasses import asdict

    from .run_pyg_gnn_smoke import SmokeConfig, run_smoke

    result = run_smoke(
        SmokeConfig(
            export_root=Path(export_root),
            relation="disease_associated_gene",
            epochs=2,
            hidden_channels=8,
            seed=int(seed),
            max_train_edges=32,
        )
    )
    return asdict(result)


def load_sampled_pyg(export_root: str | Path):
    """Delegate loading of a freshly generated sample to the tested smoke loader."""

    from .run_pyg_gnn_smoke import _load_heterodata

    return _load_heterodata(Path(export_root))


def build_public_fixture(root_path: str | Path) -> Path:
    """Create a tiny deterministic KG with source-like nodes, evidence and vectors."""

    root_path = Path(root_path)
    root = kg_storage.open_kg_root(str(root_path))
    genes = pd.DataFrame(
        {
            "id": ["ENSG00000139618", "ENSG00000012048", "ENSG00000141510", "ENSG00000146648"],
            "ncbi_gene_id": ["675", "672", "7157", "1956"],
            "hgnc_id": ["HGNC:1101", "HGNC:1100", "HGNC:11998", "HGNC:3236"],
            "uniprot_id": ["P51587", "P38398", "P04637", "P00533"],
            "gene_name": ["BRCA2", "BRCA1", "TP53", "EGFR"],
            "name": ["BRCA2", "BRCA1", "Tumor protein p53", "Epidermal growth factor receptor"],
            "description": ["DNA repair", "DNA repair", "Genome integrity", "Receptor tyrosine kinase"],
            "source": ["Ensembl"] * 4,
        }
    )
    diseases = pd.DataFrame(
        {
            "id": ["EFO:0000305", "MONDO:0007254", "EFO:0000616"],
            "mondo_id": ["MONDO:0007254", "MONDO:0007254", "MONDO:0004992"],
            "efo_id": ["EFO:0000305", "EFO:0000305", "EFO:0000616"],
            "mesh_id": ["D001943", "D001943", "D008545"],
            "hp_id": ["", "", ""],
            "omim_id": ["114480", "114480", "155600"],
            "doid_id": ["DOID:1612", "DOID:1612", "DOID:1909"],
            "icd10_code": ["C50", "C50", "C34"],
            "name": ["breast carcinoma", "breast cancer", "lung carcinoma"],
            "description": ["Breast malignancy", "Breast malignancy ontology term", "Lung malignancy"],
            "source": ["OpenTargets"] * 3,
        }
    )
    molecules = pd.DataFrame(
        {
            "id": ["CHEMBL25", "CHEMBL941", "CHEMBL1201585"],
            "drugbank_id": ["DB00945", "DB00530", "DB00317"],
            "pubchem_cid": ["2244", "3672", "123631"],
            "cas_rn": ["50-78-2", "58-08-2", "183321-74-6"],
            "inchikey": ["BSYNRYMUTXBXSQ-UHFFFAOYSA-N", "RYYVLZVUVIJVGH-UHFFFAOYSA-N", "AAKJLRGGTJKAMG-UHFFFAOYSA-N"],
            "smiles": ["CC(=O)OC1=CC=CC=C1C(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "COC1=NC=NC2=C1C=CN2"],
            "name": ["aspirin", "caffeine", "gefitinib"],
        }
    )
    kg_storage.write_nodes(root, "gene", genes)
    kg_storage.write_nodes(root, "disease", diseases)
    kg_storage.write_nodes(root, "molecule", molecules)

    def edge_frame(relation: str, x_type: str, y_type: str, pairs: list[tuple[str, str]]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "x_id": [pair[0] for pair in pairs],
                "x_type": [x_type] * len(pairs),
                "y_id": [pair[1] for pair in pairs],
                "y_type": [y_type] * len(pairs),
                "relation": [relation] * len(pairs),
                "display_relation": [relation.replace("_", " ")] * len(pairs),
                "source": ["OpenTargets fixture"] * len(pairs),
                "credibility": [3, 2, 3, 2, 1, 3][: len(pairs)],
            }
        )

    association_pairs = [
        ("ENSG00000139618", "EFO:0000305"),
        ("ENSG00000012048", "EFO:0000305"),
        ("ENSG00000141510", "MONDO:0007254"),
        ("ENSG00000141510", "EFO:0000616"),
        ("ENSG00000146648", "EFO:0000616"),
    ]
    target_pairs = [
        ("CHEMBL25", "ENSG00000141510"),
        ("CHEMBL941", "ENSG00000146648"),
        ("CHEMBL1201585", "ENSG00000146648"),
        ("CHEMBL1201585", "ENSG00000141510"),
    ]
    kg_storage.write_edges(root, "disease_associated_gene", edge_frame("disease_associated_gene", "gene", "disease", association_pairs))
    kg_storage.write_edges(root, "molecule_targets_gene", edge_frame("molecule_targets_gene", "molecule", "gene", target_pairs))

    evidence = pd.DataFrame(
        [
            {
                "relation": "disease_associated_gene",
                "x_id": gene,
                "x_type": "gene",
                "y_id": disease,
                "y_type": "disease",
                "evidence_type": "database_record",
                "source": "OpenTargets fixture",
                "source_dataset": "associationByDatasourceDirect",
                "source_record_id": f"fixture:{index}",
                "evidence_score": score,
                "predicate": "associated_with",
                "license": "CC BY 4.0 fixture metadata",
                "release": "fixture-v1",
            }
            for index, ((gene, disease), score) in enumerate(zip(association_pairs, [0.92, 0.89, 0.86, 0.81, 0.74], strict=True))
        ]
    )
    kg_evidence.write_evidence(root, "disease_associated_gene", evidence)

    features = root_path / "features"
    features.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "node_id": genes["id"],
            "node_type": "gene",
            "feature_key": "gene_textual_summary",
            "summary": genes["description"],
            "source": "fixture summaries",
        }
    ).to_parquet(features / "gene_textual_summary.parquet", index=False)
    embedding_dir = features / "embeddings" / "text"
    embedding_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "node_id": [*genes["id"], *diseases["id"], *molecules["id"]],
            "node_type": ["gene"] * len(genes) + ["disease"] * len(diseases) + ["molecule"] * len(molecules),
            "embedding_model": ["fixture-biomedical-encoder-v1"] * (len(genes) + len(diseases) + len(molecules)),
            "embedding": [
                [1.0, 0.9, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0],
                [1.0, 0.95, 0.0, 0.0, 0.05, 0.0, 0.0, 0.0],
                [0.7, 0.4, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.1, 0.0, 0.9, 0.8, 0.0, 0.0, 0.0, 0.0],
                [0.8, 0.8, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.75, 0.82, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.15, 0.0, 0.8, 0.75, 0.0, 0.0, 0.0, 0.0],
                [0.2, 0.1, 0.0, 0.0, 0.8, 0.5, 0.0, 0.0],
                [0.1, 0.1, 0.0, 0.0, 0.7, 0.6, 0.0, 0.0],
                [0.05, 0.0, 0.8, 0.7, 0.0, 0.0, 0.2, 0.0],
            ],
        }
    ).to_parquet(embedding_dir / "fixture.parquet", index=False)
    return root_path
