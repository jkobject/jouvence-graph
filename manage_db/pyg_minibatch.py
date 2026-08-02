"""Stream canonical Jouvence Parquets into compact PyG minibatches.

This module deliberately does not create a whole-graph ``.pt`` export.  It
iterates a complete relation by Parquet record batches, compacts the endpoint
IDs inside each batch, fetches only matching node rows and embeddings, and
returns an independent ``HeteroData`` object per batch.

This is edge minibatching.  Multi-hop neighbor sampling needs a separately
reviewed on-disk adjacency/GraphStore index; repeatedly scanning Parquet to find
neighbors would not be scalable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from fsspec.core import url_to_fs

from .kg_schema import RELATION_BY_NAME
from .kg_storage import open_kg_root
from .public_notebooks import _storage_options

try:
    import torch
    from torch_geometric.data import HeteroData
except Exception:  # pragma: no cover - explicit runtime error below
    torch = None  # type: ignore[assignment]
    HeteroData = None  # type: ignore[assignment,misc]


@dataclass(frozen=True)
class EmbeddingSpec:
    """How a node embedding table links vectors to canonical node IDs."""

    uri: str
    id_column: str = "node_id"
    vector_column: str = "embedding"


def _require_pyg() -> None:
    if torch is None or HeteroData is None:
        raise RuntimeError("PyG minibatching requires the 'gnn' dependency group")


def _filtered_frame(
    uri: str,
    key: str,
    values: Sequence[str],
    *,
    billing_project: str | None,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Read rows whose key belongs to a minibatch ID set."""

    if not values:
        return pd.DataFrame(columns=list(columns or ()))
    options = _storage_options(uri, billing_project) if uri.startswith("gs://") else {}
    fs, path = url_to_fs(uri, **options)
    parquet = pq.ParquetFile(path, filesystem=fs)
    available = set(parquet.schema_arrow.names)
    wanted = list(columns) if columns is not None else list(parquet.schema_arrow.names)
    missing = sorted(set(wanted) - available)
    if key not in available:
        raise ValueError(f"{uri} missing join key {key!r}")
    if missing:
        raise ValueError(f"{uri} missing requested columns: {missing}")
    table = pq.read_table(
        path,
        filesystem=fs,
        columns=wanted,
        filters=[(key, "in", list(dict.fromkeys(map(str, values))))],
    )
    return table.to_pandas()


def _ordered_unique(values: Sequence[object]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values))


ASSERTION_KEY = ("relation", "x_id", "x_type", "y_id", "y_type")


def _attach_evidence(
    *,
    data,
    edge_type: tuple[str, str, str],
    kg,
    relation: str,
    edges: pd.DataFrame,
    billing_project: str | None,
) -> None:
    """Attach provenance rows plus their local edge index to an edge batch."""

    evidence_relative = f"evidence/{relation}.parquet"
    if not kg.exists(evidence_relative):
        return
    evidence = _filtered_frame(
        kg._as_public(evidence_relative),
        "x_id",
        _ordered_unique(edges["x_id"].tolist()),
        billing_project=billing_project,
    )
    if evidence.empty:
        return
    for column in ASSERTION_KEY:
        if column not in evidence:
            raise ValueError(f"{evidence_relative} missing stable assertion key column {column!r}")

    edge_keys = [tuple(map(str, row)) for row in edges[list(ASSERTION_KEY)].itertuples(index=False, name=None)]
    key_to_local_edge: dict[tuple[str, ...], int] = {}
    for index, key in enumerate(edge_keys):
        key_to_local_edge.setdefault(key, index)
    evidence_keys = [
        tuple(map(str, row))
        for row in evidence[list(ASSERTION_KEY)].itertuples(index=False, name=None)
    ]
    keep = [key in key_to_local_edge for key in evidence_keys]
    evidence = evidence.loc[keep].reset_index(drop=True)
    evidence_keys = [key for key, retained in zip(evidence_keys, keep, strict=True) if retained]
    if evidence.empty:
        return
    assert torch is not None
    data[edge_type].evidence_to_edge = torch.as_tensor(
        [key_to_local_edge[key] for key in evidence_keys], dtype=torch.long
    )
    data[edge_type].evidence_records = evidence.to_dict(orient="records")


def _node_payload(
    *,
    kg,
    node_type: str,
    ids: list[str],
    embedding: EmbeddingSpec | None,
    billing_project: str | None,
    strict_embeddings: bool,
) -> tuple[pd.DataFrame, np.ndarray | None]:
    node_uri = kg.node_path(node_type)
    nodes = _filtered_frame(
        node_uri,
        "id",
        ids,
        billing_project=billing_project,
    )
    if "id" not in nodes:
        raise ValueError(f"nodes/{node_type}.parquet missing id")
    if nodes["id"].astype(str).duplicated().any():
        raise ValueError(f"nodes/{node_type}.parquet has duplicate IDs in minibatch")
    nodes = (
        pd.DataFrame({"id": ids})
        .merge(nodes.assign(id=nodes["id"].astype(str)), on="id", how="left", validate="one_to_one")
    )
    missing_nodes = nodes.drop(columns=["id"]).isna().all(axis=1) if len(nodes.columns) > 1 else pd.Series(False, index=nodes.index)
    if missing_nodes.any():
        raise ValueError(f"nodes/{node_type}.parquet missing {int(missing_nodes.sum())} minibatch endpoints")

    if embedding is None:
        return nodes, None
    vectors = _filtered_frame(
        embedding.uri,
        embedding.id_column,
        ids,
        billing_project=billing_project,
        columns=(embedding.id_column, embedding.vector_column),
    )
    if vectors[embedding.id_column].astype(str).duplicated().any():
        raise ValueError(f"{embedding.uri} has duplicate embedding IDs in minibatch")
    vectors = pd.DataFrame({embedding.id_column: ids}).merge(
        vectors.assign(**{embedding.id_column: vectors[embedding.id_column].astype(str)}),
        on=embedding.id_column,
        how="left",
        validate="one_to_one",
    )
    missing = vectors[embedding.vector_column].isna()
    if missing.any() and strict_embeddings:
        raise ValueError(
            f"{node_type} missing embeddings for {int(missing.sum())}/{len(ids)} minibatch nodes"
        )
    present = [value for value in vectors[embedding.vector_column] if value is not None and not (isinstance(value, float) and np.isnan(value))]
    if not present:
        return nodes, None
    dimension = len(np.asarray(present[0]))
    matrix = np.full((len(ids), dimension), np.nan, dtype=np.float32)
    for index, value in enumerate(vectors[embedding.vector_column]):
        if value is None or (isinstance(value, float) and np.isnan(value)):
            continue
        vector = np.asarray(value, dtype=np.float32)
        if vector.shape != (dimension,):
            raise ValueError(f"{embedding.uri} has inconsistent embedding dimensions")
        matrix[index] = vector
    return nodes, matrix


def iter_relation_minibatches(
    kg_root: str,
    relation: str,
    *,
    batch_size: int,
    embeddings: Mapping[str, EmbeddingSpec] | None = None,
    edge_feature_columns: Sequence[str] = ("credibility",),
    billing_project: str | None = None,
    strict_embeddings: bool = True,
    include_evidence: bool = True,
) -> Iterator["HeteroData"]:
    """Yield compact PyG minibatches while traversing every edge in a relation.

    Endpoint node indices are local to each yielded batch.  ``node_id`` and
    ``edge_id`` retain canonical identities.  Node table columns are available
    in ``node_attrs``; selected numeric edge columns become ``edge_attr``.
    """

    _require_pyg()
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if relation not in RELATION_BY_NAME:
        raise KeyError(f"unknown relation: {relation}")
    embeddings = dict(embeddings or {})
    kg = open_kg_root(kg_root)
    spec = RELATION_BY_NAME[relation]
    src_type, dst_type = spec.source.value, spec.target.value
    edge_type = (src_type, relation, dst_type)
    edge_path = kg._edge_internal(relation)
    parquet = pq.ParquetFile(edge_path, filesystem=kg.fs)
    required = ["relation", "x_id", "x_type", "y_id", "y_type"]
    available = set(parquet.schema_arrow.names)
    missing = sorted(set(required) - available)
    if missing:
        raise ValueError(f"edges/{relation}.parquet missing columns: {missing}")
    selected_features = [column for column in edge_feature_columns if column in available]
    columns = required + selected_features

    for record_batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
        edges = pa.Table.from_batches([record_batch]).to_pandas()
        if edges.empty:
            continue
        if set(edges["relation"].astype(str)) != {relation}:
            raise ValueError(f"edges/{relation}.parquet contains relation drift")
        if set(edges["x_type"].astype(str)) != {src_type} or set(edges["y_type"].astype(str)) != {dst_type}:
            raise ValueError(f"edges/{relation}.parquet contains endpoint-type drift")

        x_ids = _ordered_unique(edges["x_id"])
        y_ids = _ordered_unique(edges["y_id"])
        ids_by_type = {src_type: x_ids, dst_type: y_ids}
        if src_type == dst_type:
            ids_by_type[src_type] = _ordered_unique([*x_ids, *y_ids])

        data = HeteroData()
        for node_type, ids in ids_by_type.items():
            node_frame, matrix = _node_payload(
                kg=kg,
                node_type=node_type,
                ids=ids,
                embedding=embeddings.get(node_type),
                billing_project=billing_project,
                strict_embeddings=strict_embeddings,
            )
            data[node_type].num_nodes = len(ids)
            data[node_type].node_id = ids
            data[node_type].node_attrs = {
                column: node_frame[column].tolist()
                for column in node_frame.columns
                if column != "id"
            }
            if matrix is not None:
                data[node_type].x = torch.as_tensor(matrix, dtype=torch.float32)
                data[node_type].x_mask = torch.as_tensor(
                    np.isfinite(matrix).all(axis=1), dtype=torch.bool
                )

        src_index = {node_id: index for index, node_id in enumerate(ids_by_type[src_type])}
        dst_index = {node_id: index for index, node_id in enumerate(ids_by_type[dst_type])}
        edge_index = np.vstack([
            edges["x_id"].astype(str).map(src_index).to_numpy(dtype=np.int64),
            edges["y_id"].astype(str).map(dst_index).to_numpy(dtype=np.int64),
        ])
        data[edge_type].edge_index = torch.as_tensor(edge_index, dtype=torch.long)
        data[edge_type].edge_id = [
            f"{relation}|{x_id}|{y_id}"
            for x_id, y_id in zip(edges["x_id"], edges["y_id"], strict=True)
        ]
        if selected_features:
            numeric = edges[selected_features].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
            data[edge_type].edge_attr = torch.as_tensor(numeric, dtype=torch.float32)
            data[edge_type].edge_attr_names = selected_features
        if include_evidence:
            _attach_evidence(
                data=data,
                edge_type=edge_type,
                kg=kg,
                relation=relation,
                edges=edges,
                billing_project=billing_project,
            )
        yield data


def iter_kg_minibatches(
    kg_root: str,
    relations: Iterable[str],
    *,
    batch_size: int,
    embeddings: Mapping[str, EmbeddingSpec] | None = None,
    edge_feature_columns: Sequence[str] = ("credibility", "score"),
    billing_project: str | None = None,
    strict_embeddings: bool = False,
    include_evidence: bool = True,
) -> Iterator[tuple[str, "HeteroData"]]:
    """Traverse complete canonical relations one minibatch at a time.

    Relations are streamed sequentially so only one edge batch and its endpoint
    node/embedding rows are resident.  Call this factory again for each epoch.
    """

    for relation in relations:
        for batch in iter_relation_minibatches(
            kg_root,
            relation,
            batch_size=batch_size,
            embeddings=embeddings,
            edge_feature_columns=edge_feature_columns,
            billing_project=billing_project,
            strict_embeddings=strict_embeddings,
            include_evidence=include_evidence,
        ):
            yield relation, batch
