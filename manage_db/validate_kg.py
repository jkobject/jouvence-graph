"""Validate a stored TxGNN Parquet knowledge graph.

The default mode is an exact DuckDB anti-join validator. It scans Parquet files
relation-by-relation and never materializes high-cardinality node ID sets such
as ``enhancer`` in Python memory. This is the intended validator for the
canonical GCS-FUSE export under ``/mnt/gcs/jouvencekb/main``.

A legacy PyArrow streaming mode is kept for environments where DuckDB cannot
read the storage URI directly, but it still loads node ID sets in memory and is
therefore not appropriate for very large node types.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import duckdb
import pandas as pd
import pyarrow.parquet as pq

from manage_db import kg_storage
from txgnn import KGLoader


@dataclass(frozen=True)
class StreamingValidationReport:
    node_counts: dict[str, int]
    edge_counts: dict[str, int]
    dangling_edges: dict[str, int]

    @property
    def total_nodes(self) -> int:
        return sum(self.node_counts.values())

    @property
    def total_edges(self) -> int:
        return sum(self.edge_counts.values())

    @property
    def total_dangling_edges(self) -> int:
        return sum(self.dangling_edges.values())

    @property
    def ok(self) -> bool:
        return self.total_dangling_edges == 0


ValidationReport = StreamingValidationReport
ProgressCallback = Callable[[str], None]


def _resolve_kg_uri(kg_path: str | Path) -> str:
    uri = str(kg_path)
    if "://" not in uri:
        path = Path(uri)
        if not ((path / "nodes").exists() or (path / "edges").exists()):
            uri = str(path / "kg")
    return uri


def _duckdb_path(path: str | Path) -> str:
    """Return a path literal suitable for DuckDB read_parquet()."""

    return str(path).replace("'", "''")


def _node_path_for_duckdb(root: kg_storage.KGRoot, node_type: str) -> str:
    if "://" in root.uri:
        # DuckDB's Python package is not configured here with fsspec/GCS. Keep
        # remote object-store validation on the pyarrow/fsspec path; the VPS
        # canonical KG should be validated through the GCS-FUSE mount instead.
        raise ValueError(
            "DuckDB validation requires a local filesystem path. For GCS, use "
            "the mounted path such as /mnt/gcs/jouvencekb/main, or pass "
            "--pyarrow-streaming for small remote KGs."
        )
    return _duckdb_path(root._node_internal(node_type))


def _edge_path_for_duckdb(root: kg_storage.KGRoot, relation: str) -> str:
    if "://" in root.uri:
        raise ValueError(
            "DuckDB validation requires a local filesystem path. For GCS, use "
            "the mounted path such as /mnt/gcs/jouvencekb/main, or pass "
            "--pyarrow-streaming for small remote KGs."
        )
    return _duckdb_path(root._edge_internal(relation))


def validate_duckdb(
    kg_path: str | Path,
    *,
    threads: int = 2,
    memory_limit: str | None = None,
    temp_dir: str | Path | None = None,
    progress: ProgressCallback | None = None,
    progress_every_relations: int = 0,
) -> ValidationReport:
    """Validate a KG exactly with bounded-memory DuckDB anti-joins.

    For each edge file, this computes:

    - total edge rows
    - rows whose ``x_id`` is absent from ``nodes/<x_type>.parquet``
    - rows whose ``y_id`` is absent from ``nodes/<y_type>.parquet``

    The final dangling count is the row-wise OR of source/target absence, so a
    row missing both endpoints is counted once. The query groups by observed
    ``x_type``/``y_type`` within the relation, preserving correctness for any
    mixed legacy relation file.
    """

    root = kg_storage.open_kg_root(_resolve_kg_uri(kg_path))
    node_types = root.list_nodes()
    edge_relations = root.list_edges()

    def connect() -> duckdb.DuckDBPyConnection:
        con = duckdb.connect(":memory:")
        con.execute(f"PRAGMA threads={int(threads)}")
        if memory_limit:
            con.execute(f"PRAGMA memory_limit='{memory_limit}'")
        if temp_dir is not None:
            Path(temp_dir).mkdir(parents=True, exist_ok=True)
            con.execute(f"PRAGMA temp_directory='{_duckdb_path(temp_dir)}'")
        return con

    node_counts: dict[str, int] = {}
    for node_type in node_types:
        con = connect()
        path = _node_path_for_duckdb(root, node_type)
        count = con.execute(
            f"SELECT count(id) FROM read_parquet('{path}')"
        ).fetchone()[0]
        con.close()
        node_counts[node_type] = int(count)
        if progress is not None:
            progress(f"counted nodes/{node_type}.parquet rows={count}")

    edge_counts: dict[str, int] = {}
    dangling_edges: dict[str, int] = {}

    for relation_index, relation in enumerate(edge_relations, start=1):
        # Use a fresh DuckDB connection per relation. Large enhancer anti-joins
        # can otherwise leave allocator state resident across the 40-relation
        # loop and trigger cgroup OOM before the final report.
        con = connect()
        edge_path = _edge_path_for_duckdb(root, relation)
        pairs = con.execute(
            f"""
            SELECT x_type::VARCHAR AS x_type, y_type::VARCHAR AS y_type, count(*) AS rows
            FROM read_parquet('{edge_path}')
            GROUP BY 1, 2
            ORDER BY 1, 2
            """
        ).fetchall()

        total_rows = sum(int(row[2]) for row in pairs)
        total_dangling = 0

        for x_type, y_type, _rows in pairs:
            x_type = str(x_type)
            y_type = str(y_type)
            if x_type not in node_counts:
                raise ValueError(
                    f"edges/{relation}.parquet references missing node type {x_type!r}"
                )
            if y_type not in node_counts:
                raise ValueError(
                    f"edges/{relation}.parquet references missing node type {y_type!r}"
                )

            x_node_path = _node_path_for_duckdb(root, x_type)
            y_node_path = _node_path_for_duckdb(root, y_type)
            dangling = con.execute(
                f"""
                WITH edges AS (
                    SELECT x_id::VARCHAR AS x_id, y_id::VARCHAR AS y_id
                    FROM read_parquet('{edge_path}')
                    WHERE x_type = ? AND y_type = ?
                ),
                x_nodes AS (
                    SELECT id::VARCHAR AS id
                    FROM read_parquet('{x_node_path}')
                ),
                y_nodes AS (
                    SELECT id::VARCHAR AS id
                    FROM read_parquet('{y_node_path}')
                )
                SELECT count(*)
                FROM edges e
                LEFT JOIN x_nodes x ON e.x_id = x.id
                LEFT JOIN y_nodes y ON e.y_id = y.id
                WHERE x.id IS NULL OR y.id IS NULL
                """,
                [x_type, y_type],
            ).fetchone()[0]
            total_dangling += int(dangling)

        edge_counts[relation] = total_rows
        dangling_edges[relation] = total_dangling
        con.close()

        if (
            progress is not None
            and progress_every_relations > 0
            and (
                relation_index % progress_every_relations == 0
                or relation_index == len(edge_relations)
            )
        ):
            progress(
                f"validated edges/{relation}.parquet "
                f"relation={relation_index}/{len(edge_relations)} "
                f"rows={edge_counts[relation]} dangling={dangling_edges[relation]}"
            )

    return ValidationReport(node_counts, edge_counts, dangling_edges)


def _load_node_id_sets(
    root: kg_storage.KGRoot,
    *,
    progress: ProgressCallback | None = None,
) -> tuple[dict[str, set[str]], dict[str, int]]:
    node_ids: dict[str, set[str]] = {}
    node_counts: dict[str, int] = {}
    for node_type in root.list_nodes():
        path = root._node_internal(node_type)
        values: set[str] = set()
        rows = 0
        with root.fs.open(path, "rb") as fh:
            parquet_file = pq.ParquetFile(fh)
            for batch in parquet_file.iter_batches(columns=["id"], batch_size=250_000):
                series = batch.column("id").to_pandas().astype(str)
                rows += len(series)
                values.update(series.tolist())
        node_ids[node_type] = values
        node_counts[node_type] = rows
        if progress is not None:
            progress(f"loaded nodes/{node_type}.parquet rows={rows}")
    return node_ids, node_counts


def validate_streaming(
    kg_path: str | Path,
    *,
    batch_size: int = 250_000,
    progress: ProgressCallback | None = None,
    progress_every_relations: int = 0,
) -> StreamingValidationReport:
    """Legacy PyArrow/fsspec streaming validator.

    This scans edges in batches but loads node ID sets in Python. Use only for
    small KGs or remote URIs that DuckDB cannot read directly.
    """

    root = kg_storage.open_kg_root(_resolve_kg_uri(kg_path))
    node_ids, node_counts = _load_node_id_sets(root, progress=progress)
    edge_counts: dict[str, int] = {}
    dangling_edges: dict[str, int] = {}

    edge_relations = root.list_edges()
    for relation_index, relation in enumerate(edge_relations, start=1):
        path = root._edge_internal(relation)
        edge_counts[relation] = 0
        dangling_edges[relation] = 0
        with root.fs.open(path, "rb") as fh:
            parquet_file = pq.ParquetFile(fh)
            for batch in parquet_file.iter_batches(
                columns=["x_id", "x_type", "y_id", "y_type"],
                batch_size=batch_size,
            ):
                df = batch.to_pandas()
                edge_counts[relation] += len(df)
                if df.empty:
                    continue
                # Relations are expected to be homogeneous, but group defensively
                # to keep validation correct for any mixed legacy file.
                for (x_type, y_type), group in df.groupby(["x_type", "y_type"], sort=False):
                    x_known = node_ids.get(str(x_type), set())
                    y_known = node_ids.get(str(y_type), set())
                    dangling = (~group["x_id"].astype(str).isin(x_known)) | (
                        ~group["y_id"].astype(str).isin(y_known)
                    )
                    dangling_edges[relation] += int(dangling.sum())
        if (
            progress is not None
            and progress_every_relations > 0
            and (
                relation_index % progress_every_relations == 0
                or relation_index == len(edge_relations)
            )
        ):
            progress(
                f"validated edges/{relation}.parquet "
                f"relation={relation_index}/{len(edge_relations)} "
                f"rows={edge_counts[relation]} dangling={dangling_edges[relation]}"
            )

    return StreamingValidationReport(node_counts, edge_counts, dangling_edges)


def _print_report(report) -> None:
    print("KG validation summary")
    print(f"  node_types: {len(report.node_counts)}")
    print(f"  edge_types: {len(report.edge_counts)}")
    print(f"  total_nodes: {report.total_nodes}")
    print(f"  total_edges: {report.total_edges}")
    print(f"  total_dangling_edges: {report.total_dangling_edges}")

    if report.dangling_edges:
        printed_header = False
        for relation, count in sorted(report.dangling_edges.items()):
            if count:
                if not printed_header:
                    print("\nDangling edges by relation")
                    printed_header = True
                print(f"  {relation}: {count}")

    if not report.ok:
        print("\nFAIL: KG contains dangling edges", file=sys.stderr)
    else:
        print("\nPASS: KG has no dangling edges")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate a Jouvence Parquet KG with exact bounded-memory anti-joins."
    )
    parser.add_argument(
        "kg_path",
        help="Path to a KG root, or a data directory containing kg/.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=250_000,
        help="Batch size for --pyarrow-streaming legacy mode.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=2,
        help="DuckDB worker threads for the default exact validator.",
    )
    parser.add_argument(
        "--duckdb-memory-limit",
        default=None,
        help="Optional DuckDB memory limit, e.g. '6GB'.",
    )
    parser.add_argument(
        "--duckdb-temp-dir",
        default=None,
        help="Optional DuckDB spill/temp directory for large anti-joins.",
    )
    parser.add_argument(
        "--pyarrow-streaming",
        action="store_true",
        help=(
            "Use the legacy PyArrow/fsspec validator. This loads node ID sets "
            "in Python and is not suitable for high-cardinality enhancer KGs."
        ),
    )
    parser.add_argument(
        "--legacy-loader",
        action="store_true",
        help="Use KGLoader.validate(), which materializes all node and edge tables.",
    )
    parser.add_argument(
        "--progress-every-relations",
        type=int,
        default=0,
        help=(
            "Emit flushed progress heartbeats to stderr after loading/counting "
            "each node file and after every N edge relations. Default 0 keeps "
            "stdout/stderr quiet until the final report."
        ),
    )
    args = parser.parse_args(argv)

    def progress(message: str) -> None:
        print(f"[validate_kg] {message}", file=sys.stderr, flush=True)

    progress_cb = progress if args.progress_every_relations > 0 else None

    if args.legacy_loader:
        report = KGLoader(args.kg_path).validate()
    elif args.pyarrow_streaming:
        report = validate_streaming(
            args.kg_path,
            batch_size=args.batch_size,
            progress=progress_cb,
            progress_every_relations=args.progress_every_relations,
        )
    else:
        report = validate_duckdb(
            args.kg_path,
            threads=args.threads,
            memory_limit=args.duckdb_memory_limit,
            temp_dir=args.duckdb_temp_dir,
            progress=progress_cb,
            progress_every_relations=args.progress_every_relations,
        )

    _print_report(report)
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
