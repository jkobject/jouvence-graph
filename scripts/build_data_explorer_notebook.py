#!/usr/bin/env python3
"""Build the live, read-only Jouvence data-model explorer notebook."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "07_data_inventory_explorer.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip() + "\n")


def code(text: str):
    return nbf.v4.new_code_cell(text.strip() + "\n")


def build(output: Path = OUTPUT) -> None:
    cells = [
        md("""
# 07 — Explore the live Jouvence knowledge graph

This notebook is a **bounded, read-only GCS explorer and data-model tutorial**. It shows how to:

1. list every discovered Parquet by layer and filename;
2. load one table directly by `layer` and `name`;
3. compute useful footer, missingness, uniqueness, and numeric statistics without accidentally loading a huge table;
4. follow a `node → edge → node` path and an `edge → evidence` path;
5. project a bounded embedding sample with UMAP (or a deterministic PCA fallback);
6. distinguish staging objects from schema-declared nodes/relations that are not yet materialized in the KG;
7. stream the complete KG into bounded PyTorch Geometric minibatches without a global export;
8. copy the single production PyG index from GCS to a user-selected local directory and open real neighbor loaders.

The canonical roots are `gs://jouvencekb/main/{nodes,edges,evidence,features,embeddings,...}`. Temporary candidates belong only under `gs://jouvencekb/staging/`. LaminDB runtime state under `.lamin/` is intentionally excluded.
"""),
        md("""
## 1. Live access and safety bounds

Authentication, bucket authorization, and requester-pays billing are separate. Use Application Default Credentials and, when ADC cannot infer it, set a caller-owned billing project before starting Jupyter:

```bash
gcloud auth application-default login
export JOUVENCE_BILLING_PROJECT='<your-billing-project>'
uv run jupyter lab
```

Listings use a server-side cap. Row reads stop at explicit limits. Full-table statistics refuse tables above `MAX_FULL_STATS_ROWS`; increase that threshold only on an approved in-region worker.
"""),
        code("""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from fsspec.core import url_to_fs
from google.auth import default as google_auth_default
from IPython.display import display

REPO_ROOT = Path.cwd()
if REPO_ROOT.name == "notebooks":
    REPO_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from manage_db.data_explorer import list_parquet_uris
from manage_db.kg_schema import NODE_TYPES, RELATIONS
from manage_db.public_notebooks import PUBLIC_KG_ROOT, _storage_options, read_bounded_parquet

try:
    SAMPLE_ROWS = int(os.environ.get("JOUVENCE_EXPLORER_SAMPLE_ROWS", "8"))
    MAX_LISTED = int(os.environ.get("JOUVENCE_EXPLORER_MAX_FILES", "250"))
    EMBEDDING_ROWS = int(os.environ.get("JOUVENCE_EXPLORER_EMBEDDING_ROWS", "300"))
    MAX_FULL_STATS_ROWS = int(os.environ.get("JOUVENCE_EXPLORER_FULL_STATS_ROWS", "100000"))
except ValueError as exc:
    raise ValueError("explorer row/file limits must be integers") from exc
if not 1 <= SAMPLE_ROWS <= 100:
    raise ValueError("JOUVENCE_EXPLORER_SAMPLE_ROWS must be between 1 and 100")
if not 1 <= MAX_LISTED <= 2000:
    raise ValueError("JOUVENCE_EXPLORER_MAX_FILES must be between 1 and 2000")
if not 10 <= EMBEDDING_ROWS <= 2000:
    raise ValueError("JOUVENCE_EXPLORER_EMBEDDING_ROWS must be between 10 and 2000")
if not 1 <= MAX_FULL_STATS_ROWS <= 1_000_000:
    raise ValueError("JOUVENCE_EXPLORER_FULL_STATS_ROWS must be between 1 and 1000000")

_, ADC_PROJECT = google_auth_default()
BILLING_PROJECT = os.environ.get("JOUVENCE_BILLING_PROJECT") or ADC_PROJECT
if not BILLING_PROJECT:
    raise RuntimeError("Set JOUVENCE_BILLING_PROJECT to your own requester-pays project")
"""),
        code("""
canonical_root = PUBLIC_KG_ROOT
staging_root = "gs://jouvencekb/staging"
print({
    "mode": "live-gcs-only",
    "canonical_root": str(canonical_root),
    "staging_root": staging_root,
    "sample_rows": SAMPLE_ROWS,
    "embedding_rows": EMBEDDING_ROWS,
    "read_only_gcs": True,
})
"""),
        md("""
## 2. List files by data type

`files_by_type` is the main navigation table. `layer` is the directory name in `main/`; `name` is the filename without `.parquet`. This makes selection stable and avoids copying long URIs.
"""),
        code("""
def join_uri(root, suffix: str) -> str:
    return f"{str(root).rstrip('/')}/{suffix.strip('/')}"

SURFACES = pd.DataFrame([
    {"layer": "nodes", "surface": "canonical nodes", "uri": join_uri(canonical_root, "nodes"), "status": "canonical-observed"},
    {"layer": "edges", "surface": "canonical edges", "uri": join_uri(canonical_root, "edges"), "status": "canonical-observed"},
    {"layer": "evidence", "surface": "canonical evidence", "uri": join_uri(canonical_root, "evidence"), "status": "canonical-observed"},
    {"layer": "features", "surface": "canonical features", "uri": join_uri(canonical_root, "features"), "status": "canonical-feature"},
    {"layer": "embeddings", "surface": "canonical embeddings", "uri": join_uri(canonical_root, "embeddings"), "status": "canonical-feature"},
    {"layer": "edges_inferred", "surface": "canonical inferred edges", "uri": join_uri(canonical_root, "edges_inferred"), "status": "canonical-inferred"},
    {"layer": "evidence_inferred", "surface": "canonical inferred evidence", "uri": join_uri(canonical_root, "evidence_inferred"), "status": "canonical-inferred"},
    {"layer": "staging", "surface": "temporary staging", "uri": staging_root, "status": "non-canonical"},
])
display(SURFACES)
"""),
        code("""
def list_parquets(uri: str, layer: str, status: str, surface: str, max_files: int = MAX_LISTED) -> pd.DataFrame:
    listing = list_parquet_uris(uri, limit=max_files, billing_project=BILLING_PROJECT)
    rows = [{
        "layer": layer,
        "status": status,
        "surface": surface,
        "uri": item,
        "file": Path(item).name,
        "name": Path(item).stem,
    } for item in listing.uris]
    result = pd.DataFrame(rows, columns=["layer", "status", "surface", "uri", "file", "name"])
    result.attrs["truncated"] = listing.truncated
    return result

inventories = []
for row in SURFACES.itertuples(index=False):
    frame = list_parquets(row.uri, row.layer, row.status, row.surface)
    inventories.append(frame)
    suffix = " (TRUNCATED)" if frame.attrs["truncated"] else ""
    print(f"{row.layer:20s} {len(frame):4d} Parquet files{suffix}")

inventory = pd.concat(inventories, ignore_index=True)
files_by_type = (
    inventory.groupby("layer", sort=False)["file"]
    .apply(list)
    .rename("files")
    .reset_index()
)
display(files_by_type)
display(inventory[["layer", "name", "status", "uri"]])
"""),
        md("""
### Select and load by name

Use `uri_for("edges", "disease_associated_gene")` or `load_table("nodes", "gene")`. Selection is exact: ambiguous or absent names raise a clear error. `load_table` is bounded by default.
"""),
        code("""
def uri_for(layer: str, name: str) -> str:
    matches = inventory.loc[
        inventory["layer"].eq(layer) & inventory["name"].eq(name), "uri"
    ].drop_duplicates()
    if len(matches) != 1:
        available = sorted(inventory.loc[inventory["layer"].eq(layer), "name"].unique())
        raise KeyError(f"Expected one {layer}/{name}.parquet; found {len(matches)}. Available: {available}")
    return str(matches.iloc[0])


def load_table(layer: str, name: str, limit: int = SAMPLE_ROWS) -> pd.DataFrame:
    return read_bounded_parquet(
        uri_for(layer, name), limit=limit, billing_project=BILLING_PROJECT
    )

# Edit these two strings; no URI surgery is required.
SELECTED_LAYER = "nodes"
SELECTED_NAME = "organism"
selected_uri = uri_for(SELECTED_LAYER, SELECTED_NAME)
selected = load_table(SELECTED_LAYER, SELECTED_NAME)
print("Selected:", selected_uri)
display(selected)
"""),
        md("""
## 3. Schema, row count, and 2–3 useful statistics

The footer gives exact row count, schema, row groups, and physical size without scanning row payloads. Sample diagnostics show missingness and cardinality **in the bounded sample**. `full_table_statistics` scans the whole table only when its footer row count is below the explicit safety threshold.
"""),
        code("""
def parquet_footer(uri: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    options = _storage_options(uri, BILLING_PROJECT) if uri.startswith("gs://") else {}
    fs, path = url_to_fs(uri, **options)
    parquet = pq.ParquetFile(path, filesystem=fs)
    schema = pd.DataFrame([
        {"column": field.name, "arrow_type": str(field.type), "nullable": field.nullable}
        for field in parquet.schema_arrow
    ])
    row_groups = pd.DataFrame([
        {"row_group": i, "rows": parquet.metadata.row_group(i).num_rows,
         "bytes_uncompressed": parquet.metadata.row_group(i).total_byte_size}
        for i in range(parquet.metadata.num_row_groups)
    ])
    summary = {
        "uri": uri,
        "rows": parquet.metadata.num_rows,
        "row_groups": parquet.metadata.num_row_groups,
        "columns": len(parquet.schema_arrow),
        "created_by": parquet.metadata.created_by,
        "serialized_footer_bytes": parquet.metadata.serialized_size,
    }
    return schema, row_groups, summary

selected_schema, selected_row_groups, selected_summary = parquet_footer(selected_uri)
print(json.dumps(selected_summary, indent=2, default=str))
display(selected_schema)
display(selected_row_groups.head(20))
"""),
        code("""
def sample_diagnostics(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "dtype": frame.dtypes.astype(str),
        "null_fraction_in_sample": frame.isna().mean(),
        "distinct_in_sample": frame.nunique(dropna=True),
        "example": [next((str(v)[:100] for v in frame[c] if pd.notna(v)), "") for c in frame],
    }).rename_axis("column").reset_index()


def full_table_statistics(layer: str, name: str, max_rows: int = MAX_FULL_STATS_ROWS) -> dict:
    uri = uri_for(layer, name)
    _, _, footer = parquet_footer(uri)
    if footer["rows"] > max_rows:
        raise RuntimeError(
            f"Refusing full scan of {footer['rows']:,} rows; threshold is {max_rows:,}. "
            "Use bounded samples here or an approved in-region worker."
        )
    options = _storage_options(uri, BILLING_PROJECT) if uri.startswith("gs://") else {}
    fs, path = url_to_fs(uri, **options)
    frame = pq.read_table(path, filesystem=fs).to_pandas()
    numeric = frame.select_dtypes(include=[np.number])
    return {
        "frame": frame,
        "rows": len(frame),
        "duplicate_rows": int(frame.duplicated().sum()),
        "null_fraction_by_column": frame.isna().mean().sort_values(ascending=False),
        "numeric_summary": numeric.describe().T if not numeric.empty else pd.DataFrame(),
    }

display(sample_diagnostics(selected))
selected_stats = full_table_statistics(SELECTED_LAYER, SELECTED_NAME)
print({"rows": selected_stats["rows"], "duplicate_rows": selected_stats["duplicate_rows"]})
display(selected_stats["null_fraction_by_column"].rename("null_fraction"))
display(selected_stats["numeric_summary"])
"""),
        md("""
## 4. Follow one node → edge → node

An edge points to node registries through `x_id → nodes/<x_type>.id` and `y_id → nodes/<y_type>.id`. The example chooses a small relation when available, loads one edge, then uses Parquet filters to retrieve its two endpoint rows. Never join by row number.
"""),
        code("""
def load_filtered(layer: str, name: str, filters: list[tuple[str, str, object]], limit: int = 20) -> pd.DataFrame:
    uri = uri_for(layer, name)
    options = _storage_options(uri, BILLING_PROJECT) if uri.startswith("gs://") else {}
    fs, path = url_to_fs(uri, **options)
    table = pq.read_table(path, filesystem=fs, filters=filters)
    return table.slice(0, limit).to_pandas()

edge_names = set(inventory.loc[inventory["layer"].eq("edges"), "name"])
PATH_RELATION = "organism_has_tissue" if "organism_has_tissue" in edge_names else sorted(edge_names)[0]
path_edge = load_table("edges", PATH_RELATION, limit=1)
edge_row = path_edge.iloc[0]
x_type, y_type = str(edge_row["x_type"]), str(edge_row["y_type"])
x_node = load_filtered("nodes", x_type, [("id", "=", edge_row["x_id"])], limit=1)
y_node = load_filtered("nodes", y_type, [("id", "=", edge_row["y_id"])], limit=1)

x_node = x_node.add_prefix("x_node_").rename(columns={"x_node_id": "x_id"})
y_node = y_node.add_prefix("y_node_").rename(columns={"y_node_id": "y_id"})
edge_with_endpoints = (
    path_edge.merge(x_node, on="x_id", how="left")
    .merge(y_node, on="y_id", how="left")
)
print(f"{x_type} --{PATH_RELATION}--> {y_type}")
display(edge_with_endpoints)
"""),
        md("""
The resulting row is the concrete `[1 node, 1 edge, 1 node]` path. Node rows describe endpoint entities; the edge row is the biological assertion. Presence of both endpoint nodes is necessary for graph integrity, but does not independently prove the assertion.
"""),
        md("""
## 5. Follow one edge → evidence

Observed edge tables deduplicate graph assertions. Evidence tables retain source records. Their stable join is `(relation, x_id, x_type, y_id, y_type)`; evidence multiplicity must not be interpreted as additional graph edges or automatically as independent studies.
"""),
        code("""
edge_names = set(inventory.loc[inventory["layer"].eq("edges"), "name"])
evidence_names = set(inventory.loc[inventory["layer"].eq("evidence"), "name"])
relations_with_evidence = sorted(edge_names & evidence_names)
EVIDENCE_RELATION = (
    "disease_associated_gene"
    if "disease_associated_gene" in relations_with_evidence
    else relations_with_evidence[0]
)
edge_for_evidence = load_table("edges", EVIDENCE_RELATION, limit=1)
edge_identity = edge_for_evidence.iloc[0]
identity_columns = ["relation", "x_id", "x_type", "y_id", "y_type"]
evidence_filters = [(column, "=", edge_identity[column]) for column in identity_columns]
matching_evidence = load_filtered("evidence", EVIDENCE_RELATION, evidence_filters, limit=50)
edge_with_evidence = edge_for_evidence.merge(
    matching_evidence,
    on=["relation", "x_id", "x_type", "y_id", "y_type"],
    how="left",
    suffixes=("_edge", "_evidence"),
)
print("matching evidence rows displayed:", len(matching_evidence))
display(edge_with_evidence)
"""),
        md("""
## 6. UMAP of one embedding table

The projection is bounded to `EMBEDDING_ROWS`. It uses UMAP when `umap-learn` is installed and there are enough vectors; otherwise it uses a deterministic PCA fallback. Geometry is model- and modality-specific: proximity is not functional equivalence, causality, or therapeutic evidence.
"""),
        code("""
embedding_names = sorted(inventory.loc[inventory["layer"].eq("embeddings"), "name"])
EMBEDDING_NAME = (
    "gene_text_sbiobert_snli_multinli_stsb"
    if "gene_text_sbiobert_snli_multinli_stsb" in embedding_names
    else embedding_names[0]
)
embedding_frame = load_table("embeddings", EMBEDDING_NAME, limit=EMBEDDING_ROWS)
vector_candidates = [
    column for column in embedding_frame.columns
    if column.lower() in {"embedding", "vector", "features", "x"}
]
if not vector_candidates:
    raise KeyError(f"No vector column in {EMBEDDING_NAME}: {list(embedding_frame.columns)}")
vector_column = vector_candidates[0]
valid_mask = embedding_frame[vector_column].map(lambda value: value is not None)
embedding_meta = embedding_frame.loc[valid_mask].drop(columns=[vector_column]).reset_index(drop=True)
vectors = np.stack([
    np.asarray(value, dtype=np.float32)
    for value in embedding_frame.loc[valid_mask, vector_column]
])
finite_mask = np.isfinite(vectors).all(axis=1)
vectors, embedding_meta = vectors[finite_mask], embedding_meta.loc[finite_mask].reset_index(drop=True)
print({"table": EMBEDDING_NAME, "vectors": len(vectors), "dimension": vectors.shape[1]})
"""),
        code("""
def embedding_projection(matrix: np.ndarray) -> tuple[np.ndarray, str]:
    if len(matrix) < 3:
        raise ValueError("Need at least three finite vectors for a 2D projection")
    try:
        from umap import UMAP
        neighbors = min(15, len(matrix) - 1)
        return UMAP(n_components=2, n_neighbors=neighbors, min_dist=0.1, random_state=42).fit_transform(matrix), "UMAP"
    except ImportError:
        from sklearn.decomposition import PCA
        return PCA(n_components=2, random_state=42).fit_transform(matrix), "PCA fallback"

projection, projection_method = embedding_projection(vectors)
print("projection method:", projection_method)
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(projection[:, 0], projection[:, 1], s=18, alpha=0.7, color="#0072B2")
ax.set(title=f"{projection_method}: {EMBEDDING_NAME}", xlabel="component 1", ylabel="component 2")
ax.grid(alpha=0.2)
plt.show()
"""),
        md("""
## 7. What remains in staging or is declared but absent from the KG?

These are different questions:

- `staging_objects` lists actual non-canonical Parquets currently under `staging/`.
- `schema_relations_absent_from_kg` compares the relation schema to materialized `main/edges/*.parquet`.
- `schema_nodes_absent_from_kg` compares declared node types to materialized node tables.

A schema declaration is a design/lifecycle entry, not proof that data have been built or accepted. Conversely, an empty staging prefix means no current Parquet candidate was discovered—not that every schema entry is complete.
"""),
        code("""
staging_objects = inventory.loc[inventory["layer"].eq("staging")].copy()
materialized_relations = set(inventory.loc[inventory["layer"].eq("edges"), "name"])
materialized_nodes = set(inventory.loc[inventory["layer"].eq("nodes"), "name"])

schema_relations_absent_from_kg = pd.DataFrame([
    {
        "relation": relation.name,
        "x_type": relation.source.value,
        "y_type": relation.target.value,
        "schema_status": relation.status.value,
        "reason": relation.notes,
    }
    for relation in RELATIONS
    if relation.name not in materialized_relations
], columns=["relation", "x_type", "y_type", "schema_status", "reason"]).sort_values(
    ["schema_status", "relation"], ignore_index=True
)

schema_nodes_absent_from_kg = pd.DataFrame([
    {"node_type": node_type.value, "primary_ontology": info.primary_ontology}
    for node_type, info in NODE_TYPES.items()
    if node_type.value not in materialized_nodes
], columns=["node_type", "primary_ontology"]).sort_values("node_type", ignore_index=True)

print("staging Parquets:", len(staging_objects))
print("schema relations absent from main/edges:", len(schema_relations_absent_from_kg))
print("schema node types absent from main/nodes:", len(schema_nodes_absent_from_kg))
display(staging_objects[["name", "uri", "status"]])
display(schema_relations_absent_from_kg)
display(schema_nodes_absent_from_kg)
"""),
        md("""
## 8. Stream the complete KG into PyG minibatches — no global export

A whole-graph `HeteroData`/`.pt` is too large and duplicates canonical storage. The loader below reads directly from `main/`:

1. stream one edge Parquet in record batches;
2. compact only the endpoint IDs present in that batch;
3. fetch their node rows and available embeddings;
4. attach numeric edge features such as `credibility` and `score`;
5. yield a small independent `HeteroData`.

Calling the iterator to exhaustion traverses **all rows of all selected relations** while keeping only one minibatch resident. Canonical IDs remain in `node_id`/`edge_id`; tensor indices are local to the batch.

This is **edge minibatching**, not multi-hop neighbor sampling. True random neighbor sampling over the full graph needs a reviewed on-disk adjacency/`GraphStore` index. Re-scanning every Parquet to discover neighbors for every seed batch would be slower and more expensive than such an index.
"""),
        code("""
from torch_geometric.data import HeteroData
from manage_db.pyg_minibatch import EmbeddingSpec, iter_kg_minibatches, iter_relation_minibatches

# Choose one accepted embedding table per node type. Types without an accepted
# embedding remain usable through node IDs/metadata and simply have no `.x`.
embedding_uri_by_node_type = {}
for node_type in sorted(NODE_TYPES, key=lambda item: item.value):
    candidates = inventory.loc[
        inventory["layer"].eq("embeddings")
        & inventory["name"].str.startswith(f"{node_type.value}_"),
        ["name", "uri"],
    ]
    if not candidates.empty:
        # Prefer text embeddings for general exploration; edit this policy for a model.
        ranked = candidates.assign(
            preference=candidates["name"].map(lambda name: 0 if "_text_" in name else 1)
        ).sort_values(["preference", "name"])
        embedding_uri_by_node_type[node_type.value] = ranked.iloc[0]["uri"]

embedding_specs = {
    node_type: EmbeddingSpec(uri)
    for node_type, uri in embedding_uri_by_node_type.items()
}
display(pd.Series(embedding_uri_by_node_type, name="selected_embedding_uri"))
"""),
        code("""
MINIBATCH_RELATION = "disease_associated_gene"
MINIBATCH_SIZE = 256

relation_batches = iter_relation_minibatches(
    str(canonical_root),
    MINIBATCH_RELATION,
    batch_size=MINIBATCH_SIZE,
    embeddings=embedding_specs,
    edge_feature_columns=("credibility", "score"),
    billing_project=BILLING_PROJECT,
    strict_embeddings=False,
)
pyg_batch: HeteroData = next(relation_batches)
print(pyg_batch)
print("node types:", pyg_batch.node_types)
print("edge types:", pyg_batch.edge_types)
for edge_type in pyg_batch.edge_types:
    store = pyg_batch[edge_type]
    print(
        edge_type,
        "edges=", store.num_edges,
        "evidence rows=", len(store.evidence_records) if hasattr(store, "evidence_records") else 0,
    )
for node_type in pyg_batch.node_types:
    coverage = (
        float(pyg_batch[node_type].x_mask.float().mean())
        if hasattr(pyg_batch[node_type], "x_mask") else None
    )
    print(node_type, "nodes=", pyg_batch[node_type].num_nodes, "embedding coverage=", coverage)
"""),
        md("""
### Traverse every materialized relation for one epoch

Create a fresh iterator for every epoch. The loop below is the complete-data pattern; it is not executed by default in this explorer because it would intentionally traverse the entire KG. Set the flag only on an approved in-region worker.
"""),
        code("""
RUN_COMPLETE_MINIBATCH_EPOCH = os.environ.get("JOUVENCE_EXPLORER_FULL_PYG_EPOCH") == "1"
materialized_relation_names = sorted(inventory.loc[inventory["layer"].eq("edges"), "name"].unique())

def minibatches_for_epoch():
    return iter_kg_minibatches(
        str(canonical_root),
        materialized_relation_names,
        batch_size=MINIBATCH_SIZE,
        embeddings=embedding_specs,
        edge_feature_columns=("credibility", "score"),
        billing_project=BILLING_PROJECT,
        strict_embeddings=False,
    )

if RUN_COMPLETE_MINIBATCH_EPOCH:
    relation_edge_counts = {}
    for relation_name, batch in minibatches_for_epoch():
        # Training step goes here: move this batch to the accelerator, forward,
        # backward, optimizer.step(), then release it before the next batch.
        relation_edge_counts[relation_name] = relation_edge_counts.get(relation_name, 0) + batch.num_edges
    display(pd.Series(relation_edge_counts, name="edges_seen").sort_index())
else:
    print("Complete epoch skipped. Set JOUVENCE_EXPLORER_FULL_PYG_EPOCH=1 on an approved in-region worker.")
"""),
        md("""
### What is and is not loaded

Each yielded `HeteroData` contains:

- endpoint node IDs and selected node-table metadata;
- node embeddings in `.x` where an accepted table exists;
- `.x_mask` so missing embeddings remain explicit rather than becoming fake random vectors;
- local `edge_index` for the current batch;
- numeric edge features in `edge_attr` and their names in `edge_attr_names`;
- canonical `edge_id` strings;
- matching provenance rows in `evidence_records`, with `evidence_to_edge`
  mapping every evidence row to its local edge.

Evidence is loaded dynamically but deliberately not coerced into message-passing
features: it remains typed provenance for an assertion. A model can derive a
reviewed evidence feature from these rows without losing the original records.
"""),
        md("""
## 9. Open the production neighbor-sampling build

Sequential edge minibatching is useful for inspection, transformation, and edge-wise models, but it is not a multi-hop GNN loader. The production build lives directly under `gs://jouvencekb/pyg/`; there is one current build, not a hierarchy of versions.

GCS is used for transport. We do **not** mount the bucket and do not sample through remote object reads. At job startup, `materialize_pyg_build` runs the equivalent of:

```bash
gcloud storage cp --recursive 'gs://jouvencekb/pyg/*' ./data/pyg/
```

It then verifies the manifest, sizes and SHA-256 checksums before opening adjacency and feature arrays with memory mapping.
"""),
        code("""
from manage_db.pyg_artifact import (
    PYG_ROOT,
    make_link_neighbor_loader,
    make_neighbor_loader,
    materialize_pyg_build,
    open_pyg_stores,
    resolve_pyg_build,
)

PYG_ROOT = "gs://jouvencekb/pyg"
PYG_CACHE_DIR = Path(os.environ.get("JOUVENCE_PYG_CACHE", REPO_ROOT / "data" / "pyg"))
OPEN_PRODUCTION_PYG = os.environ.get("JOUVENCE_EXPLORER_OPEN_PYG") == "1"

print("copy command:")
print(f"gcloud storage cp --recursive '{PYG_ROOT}/*' {PYG_CACHE_DIR}/")

if OPEN_PRODUCTION_PYG:
    pyg_build = resolve_pyg_build(PYG_ROOT)
    local_pyg_build = materialize_pyg_build(pyg_build, PYG_CACHE_DIR, verify=True)
    graph_store, feature_store = open_pyg_stores(local_pyg_build, mmap=True)
    print({
        "local_root": str(local_pyg_build.root),
        "edge_types": len(graph_store.get_all_edge_attrs()),
        "feature_tensors": len(feature_store.get_all_tensor_attrs()),
    })
else:
    print("Production PyG open skipped. Set JOUVENCE_EXPLORER_OPEN_PYG=1 on the training worker.")
"""),
        md("""
### Create node- or link-seed loaders

`NeighborLoader` starts from seed nodes. `LinkNeighborLoader` starts from supervised seed edges and is the default for Jouvence/TxGNN link prediction. The exact seed tensors and fanouts belong to the reviewed training split; validation/test labels and their reverse edges must be absent from message-passing adjacency.
"""),
        code("""
if OPEN_PRODUCTION_PYG:
    # Replace these examples with reviewed integer seed IDs from feature_indices/splits.
    # node_loader = make_neighbor_loader(
    #     graph_store=graph_store,
    #     feature_store=feature_store,
    #     input_nodes=("disease", train_disease_ids),
    #     num_neighbors={edge_type: [15, 5] for edge_type in graph_store_edge_types},
    #     batch_size=512,
    #     shuffle=True,
    # )
    # link_loader = make_link_neighbor_loader(
    #     graph_store=graph_store,
    #     feature_store=feature_store,
    #     edge_label_index=(("molecule", "molecule_treats_disease", "disease"), train_edges),
    #     num_neighbors=reviewed_fanouts,
    #     batch_size=512,
    #     shuffle=True,
    # )
    print("Stores are open. Provide reviewed split seeds/fanouts to make_neighbor_loader or make_link_neighbor_loader.")
else:
    print("Loader construction skipped with the production build open step.")
"""),
        md("""
## 10. Regenerate and validate

```bash
uv run --group notebooks python scripts/build_data_explorer_notebook.py
JOUVENCE_BILLING_PROJECT='<your-project>' \
  uv run --group notebooks --group gnn python scripts/check_data_explorer_notebook.py --execute
```

The committed notebook is deterministic and output-free. Live execution evidence belongs outside the committed `.ipynb`. Full epochs and production PyG materialization should run in-region. The single disk-backed adjacency/GraphStore build is copied to a user-selected local directory (default: `REPO/data/pyg/`); do not mount GCS and do not fall back to a monolithic `.pt`.
"""),
    ]

    for index, cell in enumerate(cells):
        cell["id"] = hashlib.sha256(f"data-explorer:{index}:{cell.cell_type}".encode()).hexdigest()[:12]
        if cell.cell_type == "code":
            cell.execution_count = None
            cell.outputs = []

    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
        "jouvence": {
            "data_mode": "live-gcs-only",
            "bounded": True,
            "read_only": True,
            "purpose": "data-model-and-inventory-explorer",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, output)
    print(f"wrote {output} ({len(cells)} meaningful cells)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    build(args.output)


if __name__ == "__main__":
    main()
