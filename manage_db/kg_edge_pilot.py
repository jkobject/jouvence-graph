from __future__ import annotations

"""Bounded KG edge/evidence schema sync pilot for lnschema_txgnn.

This module deliberately does **not** write to the canonical KG and does not
claim production Lamin relation sync.  It mirrors the proposed generic
``KGEdge``/``KGEdgeEvidence`` records into a small local SQLite review fixture so
we can test deterministic keys, idempotent upsert semantics, and query-helper
behavior before touching a LaminDB instance.
"""

import argparse
import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

DEFAULT_KG_ROOT = Path("/Users/jkobject/mnt/gcs/jouvencekb/main")
DEFAULT_RELATIONS = ("disease_associated_gene", "dataset_contains_tissue")

EDGE_BASE_COLUMNS = [
    "x_id",
    "x_type",
    "y_id",
    "y_type",
    "relation",
    "display_relation",
    "source",
    "credibility",
]
EVIDENCE_BASE_COLUMNS = [
    "edge_key",
    "relation",
    "x_id",
    "x_type",
    "y_id",
    "y_type",
    "evidence_type",
    "source",
    "source_dataset",
    "source_record_id",
    "paper_id",
    "dataset_id",
    "study_id",
    "evidence_score",
    "predicate",
    "direction",
]


@dataclass(frozen=True)
class SyncResult:
    relation: str
    edge_rows_available: int
    edge_rows_selected: int
    evidence_rows_available: int
    evidence_rows_selected: int
    edge_upserts: int = 0
    evidence_upserts: int = 0


def kg_root_path(root: str | Path | None = None) -> Path:
    return Path(root) if root is not None else DEFAULT_KG_ROOT


def edge_key_for(*, relation: str, x_type: str, x_id: str, y_type: str, y_id: str) -> str:
    """Return the deterministic generic KGEdge key proposed by the schema audit."""

    raw = "\t".join([relation, x_type, x_id, y_type, y_id])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def evidence_key_for(row: pd.Series | dict[str, object], *, ordinal: int = 0) -> str:
    """Return a deterministic KGEdgeEvidence key from stable evidence fields."""

    get = row.get if isinstance(row, dict) else row.get
    edge_key = str(get("edge_key") or edge_key_for(
        relation=str(get("relation") or ""),
        x_type=str(get("x_type") or ""),
        x_id=str(get("x_id") or ""),
        y_type=str(get("y_type") or ""),
        y_id=str(get("y_id") or ""),
    ))
    parts = [
        edge_key,
        str(get("source") or ""),
        str(get("source_dataset") or ""),
        str(get("source_record_id") or ""),
        str(get("paper_id") or ""),
        str(get("study_id") or ""),
        str(get("predicate") or ""),
        str(ordinal),
    ]
    return hashlib.sha256("\t".join(parts).encode("utf-8")).hexdigest()


def _clean_scalar(value: object) -> object:
    if pd.isna(value):
        return None
    return value


def _metadata_json(row: pd.Series, base_columns: Iterable[str]) -> str | None:
    base = set(base_columns)
    payload = {
        col: _clean_scalar(row[col])
        for col in row.index
        if col not in base and _clean_scalar(row[col]) is not None
    }
    if not payload:
        return None
    return json.dumps(payload, sort_keys=True, default=str, allow_nan=False)


def _read_limited_parquet(path: Path, limit: int) -> tuple[pd.DataFrame, int]:
    if not path.exists():
        return pd.DataFrame(), 0
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    selected = pf.read().slice(0, int(limit)).to_pandas() if limit else pf.read().to_pandas()
    return selected, pf.metadata.num_rows


def build_edge_frame(root: str | Path, relation: str, *, limit: int) -> tuple[pd.DataFrame, int]:
    path = kg_root_path(root) / "edges" / f"{relation}.parquet"
    frame, total = _read_limited_parquet(path, limit)
    if frame.empty:
        return pd.DataFrame(columns=[*EDGE_BASE_COLUMNS, "edge_key", "metadata_json"]), total
    missing = [col for col in ["x_id", "x_type", "y_id", "y_type", "relation"] if col not in frame.columns]
    if missing:
        raise ValueError(f"{path} missing required edge columns: {missing}")
    frame = frame.copy()
    frame["edge_key"] = [
        edge_key_for(
            relation=str(row["relation"]),
            x_type=str(row["x_type"]),
            x_id=str(row["x_id"]),
            y_type=str(row["y_type"]),
            y_id=str(row["y_id"]),
        )
        for _, row in frame.iterrows()
    ]
    for col in EDGE_BASE_COLUMNS:
        if col not in frame.columns:
            frame[col] = None
    frame["metadata_json"] = [_metadata_json(row, EDGE_BASE_COLUMNS) for _, row in frame.iterrows()]
    return frame[["edge_key", *EDGE_BASE_COLUMNS, "metadata_json"]], total


def build_evidence_frame(root: str | Path, relation: str, *, limit: int) -> tuple[pd.DataFrame, int]:
    path = kg_root_path(root) / "evidence" / f"{relation}.parquet"
    frame, total = _read_limited_parquet(path, limit)
    if frame.empty:
        return pd.DataFrame(columns=["evidence_key", *EVIDENCE_BASE_COLUMNS, "metadata_json"]), total
    frame = frame.copy()
    for col in EVIDENCE_BASE_COLUMNS:
        if col not in frame.columns:
            frame[col] = None
    frame["edge_key"] = [
        edge_key_for(
            relation=str(row["relation"]),
            x_type=str(row["x_type"]),
            x_id=str(row["x_id"]),
            y_type=str(row["y_type"]),
            y_id=str(row["y_id"]),
        )
        for _, row in frame.iterrows()
    ]
    frame["evidence_key"] = [evidence_key_for(row, ordinal=i) for i, (_, row) in enumerate(frame.iterrows())]
    frame["metadata_json"] = [_metadata_json(row, EVIDENCE_BASE_COLUMNS) for _, row in frame.iterrows()]
    return frame[["evidence_key", *EVIDENCE_BASE_COLUMNS, "metadata_json"]], total


def ensure_sqlite_schema(con: sqlite3.Connection) -> None:
    con.executescript(
        """
        create table if not exists kg_edge (
            edge_key text primary key,
            x_id text not null,
            x_type text not null,
            y_id text not null,
            y_type text not null,
            relation text not null,
            display_relation text,
            source text,
            credibility integer,
            metadata_json text
        );
        create index if not exists idx_kg_edge_relation_x on kg_edge(relation, x_id);
        create index if not exists idx_kg_edge_relation_y on kg_edge(relation, y_id);
        create index if not exists idx_kg_edge_x on kg_edge(x_type, x_id);
        create index if not exists idx_kg_edge_y on kg_edge(y_type, y_id);
        create index if not exists idx_kg_edge_relation_types on kg_edge(relation, x_type, y_type);

        create table if not exists kg_edge_evidence (
            evidence_key text primary key,
            edge_key text not null,
            relation text not null,
            x_id text not null,
            x_type text not null,
            y_id text not null,
            y_type text not null,
            evidence_type text,
            source text,
            source_dataset text,
            source_record_id text,
            paper_id text,
            dataset_id text,
            study_id text,
            evidence_score real,
            predicate text,
            direction text,
            metadata_json text
        );
        create index if not exists idx_kg_evidence_edge_key on kg_edge_evidence(edge_key);
        create index if not exists idx_kg_evidence_relation_x on kg_edge_evidence(relation, x_id);
        create index if not exists idx_kg_evidence_relation_y on kg_edge_evidence(relation, y_id);
        create index if not exists idx_kg_evidence_source on kg_edge_evidence(source, source_dataset);
        """
    )


def sync_relation_to_sqlite(
    *,
    root: str | Path,
    sqlite_path: str | Path,
    relation: str,
    edge_limit: int,
    evidence_limit: int,
) -> SyncResult:
    edge_frame, edge_total = build_edge_frame(root, relation, limit=edge_limit)
    evidence_frame, evidence_total = build_evidence_frame(root, relation, limit=evidence_limit)
    sqlite_path = Path(sqlite_path)
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(sqlite_path) as con:
        ensure_sqlite_schema(con)
        edge_rows = edge_frame.where(pd.notna(edge_frame), None).to_dict(orient="records")
        ev_rows = evidence_frame.where(pd.notna(evidence_frame), None).to_dict(orient="records")
        con.executemany(
            """
            insert into kg_edge(edge_key, x_id, x_type, y_id, y_type, relation, display_relation, source, credibility, metadata_json)
            values(:edge_key, :x_id, :x_type, :y_id, :y_type, :relation, :display_relation, :source, :credibility, :metadata_json)
            on conflict(edge_key) do update set
              display_relation=excluded.display_relation,
              source=excluded.source,
              credibility=excluded.credibility,
              metadata_json=excluded.metadata_json
            """,
            edge_rows,
        )
        con.executemany(
            """
            insert into kg_edge_evidence(evidence_key, edge_key, relation, x_id, x_type, y_id, y_type, evidence_type, source, source_dataset, source_record_id, paper_id, dataset_id, study_id, evidence_score, predicate, direction, metadata_json)
            values(:evidence_key, :edge_key, :relation, :x_id, :x_type, :y_id, :y_type, :evidence_type, :source, :source_dataset, :source_record_id, :paper_id, :dataset_id, :study_id, :evidence_score, :predicate, :direction, :metadata_json)
            on conflict(evidence_key) do update set
              source=excluded.source,
              source_dataset=excluded.source_dataset,
              source_record_id=excluded.source_record_id,
              evidence_score=excluded.evidence_score,
              predicate=excluded.predicate,
              direction=excluded.direction,
              metadata_json=excluded.metadata_json
            """,
            ev_rows,
        )
    return SyncResult(relation, edge_total, len(edge_frame), evidence_total, len(evidence_frame), len(edge_frame), len(evidence_frame))


def dry_run_relation(*, root: str | Path, relation: str, edge_limit: int, evidence_limit: int) -> SyncResult:
    edge_frame, edge_total = build_edge_frame(root, relation, limit=edge_limit)
    evidence_frame, evidence_total = build_evidence_frame(root, relation, limit=evidence_limit)
    return SyncResult(relation, edge_total, len(edge_frame), evidence_total, len(evidence_frame))


def read_pilot_edges(sqlite_path: str | Path, *, relation: str, x_ids: Sequence[str] | None = None) -> pd.DataFrame:
    path = Path(sqlite_path)
    if not path.exists():
        return pd.DataFrame()
    where = ["relation = ?"]
    params: list[object] = [relation]
    if x_ids:
        placeholders = ",".join("?" for _ in x_ids)
        where.append(f"x_id in ({placeholders})")
        params.extend(x_ids)
    with sqlite3.connect(path) as con:
        return pd.read_sql_query(f"select * from kg_edge where {' and '.join(where)}", con, params=params)


def diseases_for_gene_from_pilot(
    *,
    genes: pd.DataFrame,
    kg_root: str | Path,
    sqlite_path: str | Path,
    include_evidence: bool,
    limit: int | None,
    result_columns: Sequence[str],
) -> pd.DataFrame:
    """Return diseases-for-gene from populated pilot SQLite rows, or empty."""

    if genes.empty:
        return pd.DataFrame(columns=result_columns)
    edges = read_pilot_edges(sqlite_path, relation="disease_associated_gene", x_ids=list(genes["id"]))
    if edges.empty:
        return pd.DataFrame(columns=result_columns)

    disease_file = kg_root_path(kg_root) / "nodes" / "disease.parquet"
    diseases = pd.read_parquet(disease_file)
    merged = edges.merge(genes[["id", "gene_name", "name"]], left_on="x_id", right_on="id", how="left")
    merged = merged.merge(diseases, left_on="y_id", right_on="id", how="left", suffixes=("_gene", "_disease"))

    evidence_summary = pd.DataFrame(columns=["edge_key", "evidence_count", "evidence_sources", "evidence_score_max"])
    if include_evidence:
        with sqlite3.connect(sqlite_path) as con:
            evidence_summary = pd.read_sql_query(
                """
                select edge_key,
                       count(*) as evidence_count,
                       group_concat(distinct source) as evidence_sources,
                       max(evidence_score) as evidence_score_max
                from kg_edge_evidence
                where relation = 'disease_associated_gene'
                group by edge_key
                """,
                con,
            )
    merged = merged.merge(evidence_summary, on="edge_key", how="left")
    out = pd.DataFrame(
        {
            "gene_id": merged["x_id"],
            "gene_name": merged["gene_name"],
            "gene_label": merged["name_gene"].fillna(merged["gene_name"]).fillna(merged["x_id"]),
            "disease_id": merged["y_id"],
            "disease_name": merged["name_disease"].fillna(merged["y_id"]),
            "disease_description": merged.get("description"),
            "mondo_id": merged.get("mondo_id"),
            "efo_id": merged.get("efo_id"),
            "mesh_id": merged.get("mesh_id"),
            "hp_id": merged.get("hp_id"),
            "omim_id": merged.get("omim_id"),
            "doid_id": merged.get("doid_id"),
            "icd10_code": merged.get("icd10_code"),
            "edge_source": merged["source_gene"],
            "credibility": merged["credibility"],
            "score": merged["metadata_json"].map(lambda s: json.loads(s).get("score") if s else None),
            "evidence_count": merged["evidence_count"].fillna(0).astype("int64") if "evidence_count" in merged else 0,
            "evidence_sources": merged.get("evidence_sources"),
            "evidence_score_max": merged.get("evidence_score_max"),
        }
    )
    out = out[list(result_columns)].sort_values(["evidence_count", "disease_name", "disease_id"], ascending=[False, True, True])
    if limit is not None:
        out = out.head(int(limit))
    return out.reset_index(drop=True)


def results_to_json(results: Sequence[SyncResult]) -> str:
    return json.dumps([r.__dict__ for r in results], indent=2, sort_keys=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bounded lnschema_txgnn KG edge/evidence sync pilot")
    parser.add_argument("--kg-root", default=str(DEFAULT_KG_ROOT), help="Canonical KG root; read-only")
    parser.add_argument("--relation", action="append", dest="relations", help="Relation to inspect/sync; may be repeated")
    parser.add_argument("--edge-limit", type=int, default=25, help="Maximum edge rows per relation")
    parser.add_argument("--evidence-limit", type=int, default=25, help="Maximum evidence rows per relation")
    parser.add_argument("--sqlite-path", help="Local review SQLite path for --sync-sqlite")
    parser.add_argument("--sync-sqlite", action="store_true", help="Populate local SQLite review fixture idempotently")
    parser.add_argument("--format", choices=["json", "table"], default="json")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    relations = args.relations or list(DEFAULT_RELATIONS)
    results: list[SyncResult] = []
    for relation in relations:
        if args.sync_sqlite:
            if not args.sqlite_path:
                raise SystemExit("--sync-sqlite requires --sqlite-path")
            results.append(
                sync_relation_to_sqlite(
                    root=args.kg_root,
                    sqlite_path=args.sqlite_path,
                    relation=relation,
                    edge_limit=args.edge_limit,
                    evidence_limit=args.evidence_limit,
                )
            )
        else:
            results.append(dry_run_relation(root=args.kg_root, relation=relation, edge_limit=args.edge_limit, evidence_limit=args.evidence_limit))
    if args.format == "json":
        print(results_to_json(results))
    else:
        print(pd.DataFrame([r.__dict__ for r in results]).to_string(index=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
