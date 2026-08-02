from __future__ import annotations

"""Ergonomic read-only query helpers for the canonical Jouvence KG Parquets.

These helpers are intentionally Parquet-backed. They do not pretend that
Lamin/Bionty ORM relation tables exist for every KG edge; instead they provide a
small, stable Python/CLI layer over the canonical KG files.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

import duckdb
import pandas as pd

DEFAULT_KG_ROOT = Path(
    os.environ.get("JOUVENCE_KG_ROOT", "/Users/jkobject/mnt/gcs/jouvencekb/main")
)
DISEASE_ASSOCIATED_GENE = "disease_associated_gene"

GENE_COLUMNS = [
    "id",
    "ncbi_gene_id",
    "hgnc_id",
    "uniprot_id",
    "gene_name",
    "name",
    "description",
    "source",
]

DISEASE_COLUMNS = [
    "id",
    "name",
    "description",
    "mondo_id",
    "efo_id",
    "mesh_id",
    "hp_id",
    "omim_id",
    "doid_id",
    "icd10_code",
    "source",
]

DISEASE_RESULT_COLUMNS = [
    "gene_id",
    "gene_name",
    "gene_label",
    "disease_id",
    "disease_name",
    "disease_description",
    "mondo_id",
    "efo_id",
    "mesh_id",
    "hp_id",
    "omim_id",
    "doid_id",
    "icd10_code",
    "edge_source",
    "credibility",
    "score",
    "evidence_count",
    "evidence_sources",
    "evidence_score_max",
]


def _kg_root(root: str | Path | None = None) -> Path:
    return Path(root) if root is not None else DEFAULT_KG_ROOT


def _parquet(root: str | Path | None, *parts: str) -> str:
    path = _kg_root(root).joinpath(*parts)
    if not path.exists():
        raise FileNotFoundError(f"Required KG parquet not found: {path}")
    return str(path)


def _connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(database=":memory:")
    con.execute("PRAGMA threads=2")
    return con


def _normalise_query(identifier: str | None = None, **kwargs: str | None) -> tuple[str, str]:
    supplied = [(name, value) for name, value in [("identifier", identifier), *kwargs.items()] if value]
    if len(supplied) != 1:
        names = ", ".join(name for name, _ in supplied) or "none"
        raise ValueError(f"Provide exactly one gene query value; got {names}")
    name, value = supplied[0]
    value = str(value).strip()
    if not value:
        raise ValueError("Gene query value cannot be empty")
    return name, value


def resolve_gene(
    identifier: str | None = None,
    *,
    gene_id: str | None = None,
    gene_name: str | None = None,
    symbol: str | None = None,
    kg_root: str | Path | None = None,
    human_only: bool = True,
    limit: int | None = 20,
) -> pd.DataFrame:
    """Resolve a gene canonical KG id from an id, symbol, or display name.

    Parameters
    ----------
    identifier:
        Convenience positional query. It is matched against canonical ``id`` and
        common gene metadata columns.
    gene_id:
        Canonical KG id or cross-reference id value (e.g. ``NCBI:672``, ``672``,
        ``ENSG00000012048``, ``HGNC:1100`` where present).
    gene_name / symbol:
        Symbol/name query, case-insensitive (e.g. ``BRCA1``).
    kg_root:
        Canonical KG root. Defaults to the FUSE/GCS mount, not legacy ``.omoc``.
    human_only:
        Prefer human-like canonical ids and suppress obvious ortholog rows for
        symbol/name queries. Exact ``gene_id`` queries are never suppressed.
    limit:
        Maximum rows to return. Pass ``None`` for all matches.

    Returns
    -------
    pandas.DataFrame
        Stable columns: ``id``, cross-reference columns, labels, ``match_kind``,
        and ``rank``. Empty DataFrame means no match.
    """

    query_kind, value = _normalise_query(
        identifier, gene_id=gene_id, gene_name=gene_name, symbol=symbol
    )
    gene_file = _parquet(kg_root, "nodes", "gene.parquet")
    limit_clause = "" if limit is None else "limit ?"
    params: list[object] = [value, value, value, value, value, value, value, value]

    # Exact canonical/cross-reference ids rank first, followed by exact symbol/name.
    # For BRCA1-like symbols the KG currently contains both NCBI and ENSG rows;
    # returning both lets downstream queries union their associated diseases.
    where = """
    where (
      id = ?
      or ncbi_gene_id = ?
      or hgnc_id = ?
      or uniprot_id = ?
      or upper(gene_name) = upper(?)
      or upper(name) = upper(?)
      or upper(id) = upper(?)
      or upper(coalesce(description, '')) = upper(?)
    )
    """
    if query_kind == "gene_id":
        # A caller that supplies an exact canonical id should be allowed to reach
        # any species/namespace row in the canonical KG.
        human_filter = ""
    elif human_only:
        human_filter = """
        and (
          id like 'ENSG%'
          or id like 'NCBI:%'
          or (
            coalesce(source, '') not ilike '%homologues%'
            and coalesce(description, '') not ilike '%ortholog%'
          )
        )
        """
    else:
        human_filter = ""

    order = """
    order by
      case
        when id = ? then 0
        when upper(id) = upper(?) then 1
        when id like 'ENSG%' and (upper(gene_name) = upper(?) or upper(name) = upper(?)) then 2
        when id like 'NCBI:%' and (upper(gene_name) = upper(?) or upper(name) = upper(?)) then 3
        when upper(gene_name) = upper(?) then 4
        when upper(name) = upper(?) then 5
        else 9
      end,
      id
    """
    params.extend([value] * 8)
    if limit is not None:
        params.append(int(limit))

    sql = f"""
    select
      {', '.join(GENE_COLUMNS)},
      case
        when id = ? or upper(id) = upper(?) then 'id'
        when ncbi_gene_id = ? then 'ncbi_gene_id'
        when hgnc_id = ? then 'hgnc_id'
        when uniprot_id = ? then 'uniprot_id'
        when upper(gene_name) = upper(?) then 'gene_name'
        when upper(name) = upper(?) then 'name'
        else 'description'
      end as match_kind,
      row_number() over ({order}) as rank
    from read_parquet('{gene_file}')
    {where}
    {human_filter}
    {order}
    {limit_clause}
    """

    # select CASE params (7) + row_number order params (8) + WHERE params (8) +
    # final ORDER params (8) + optional LIMIT.
    all_params: list[object] = [value, value, value, value, value, value, value]
    all_params.extend([value] * 8)
    all_params.extend(params[:8])
    all_params.extend([value] * 8)
    if limit is not None:
        all_params.append(int(limit))
    return _connect().execute(sql, all_params).fetchdf()


def diseases_for_gene(
    identifier: str | None = None,
    *,
    gene_id: str | None = None,
    gene_name: str | None = None,
    symbol: str | None = None,
    kg_root: str | Path | None = None,
    include_evidence: bool = True,
    human_only: bool = True,
    limit: int | None = None,
    pilot_db: str | Path | None = None,
) -> pd.DataFrame:
    """Return diseases associated with a gene in the canonical KG.

    If ``pilot_db`` points at a populated local KGEdge/KGEdgeEvidence pilot
    SQLite fixture, the helper uses those generic edge/evidence rows for the
    relation and only falls back to Parquet when the pilot has no matching rows.
    This keeps the ORM/pilot-vs-Parquet boundary explicit: the current default
    behavior remains Parquet-backed, while the migration pilot can exercise the
    relation/evidence layer where it has been deliberately populated.
    """

    genes = resolve_gene(
        identifier,
        gene_id=gene_id,
        gene_name=gene_name,
        symbol=symbol,
        kg_root=kg_root,
        human_only=human_only,
        limit=None,
    )
    if genes.empty:
        return pd.DataFrame(columns=DISEASE_RESULT_COLUMNS)

    root = _kg_root(kg_root)
    if pilot_db is not None:
        from manage_db.kg_edge_pilot import diseases_for_gene_from_pilot

        pilot_result = diseases_for_gene_from_pilot(
            genes=genes,
            kg_root=root,
            sqlite_path=pilot_db,
            include_evidence=include_evidence,
            limit=limit,
            result_columns=DISEASE_RESULT_COLUMNS,
        )
        if not pilot_result.empty:
            return pilot_result

    edge_file = _parquet(root, "edges", f"{DISEASE_ASSOCIATED_GENE}.parquet")
    disease_file = _parquet(root, "nodes", "disease.parquet")
    evidence_file = root / "evidence" / f"{DISEASE_ASSOCIATED_GENE}.parquet"

    con = _connect()
    con.register("resolved_genes", genes[["id", "gene_name", "name"]])

    evidence_join = """
    left join (
      select
        x_id,
        y_id,
        count(*)::bigint as evidence_count,
        string_agg(distinct source, '; ' order by source) as evidence_sources,
        max(evidence_score) as evidence_score_max
      from read_parquet(?)
      group by x_id, y_id
    ) ev on ev.x_id = e.x_id and ev.y_id = e.y_id
    """ if include_evidence and evidence_file.exists() else ""
    evidence_select = """
      coalesce(ev.evidence_count, 0) as evidence_count,
      ev.evidence_sources,
      ev.evidence_score_max
    """ if evidence_join else """
      0::bigint as evidence_count,
      null::varchar as evidence_sources,
      null::double as evidence_score_max
    """
    limit_clause = "" if limit is None else "limit ?"

    sql = f"""
    select distinct
      e.x_id as gene_id,
      g.gene_name as gene_name,
      coalesce(g.name, g.gene_name, e.x_id) as gene_label,
      e.y_id as disease_id,
      coalesce(nullif(d.name, ''), e.y_id) as disease_name,
      d.description as disease_description,
      d.mondo_id,
      d.efo_id,
      d.mesh_id,
      d.hp_id,
      d.omim_id,
      d.doid_id,
      d.icd10_code,
      e.source as edge_source,
      e.credibility,
      e.score,
      {evidence_select}
    from read_parquet(?) e
    inner join resolved_genes g on g.id = e.x_id
    left join read_parquet(?) d on d.id = e.y_id
    {evidence_join}
    where e.relation = '{DISEASE_ASSOCIATED_GENE}'
    order by
      case when evidence_count > 0 then 0 else 1 end,
      disease_name,
      disease_id,
      gene_id
    {limit_clause}
    """
    params: list[object] = [edge_file, disease_file]
    if evidence_join:
        params.append(str(evidence_file))
    if limit is not None:
        params.append(int(limit))
    return con.execute(sql, params).fetchdf()[DISEASE_RESULT_COLUMNS]


def _records_for_output(df: pd.DataFrame) -> list[dict[str, object]]:
    clean = df.astype(object).where(pd.notna(df), None)
    return clean.to_dict(orient="records")


def _print_frame(df: pd.DataFrame, fmt: str) -> None:
    if fmt == "json":
        print(json.dumps(_records_for_output(df), indent=2, default=str, allow_nan=False))
    elif fmt == "jsonl":
        for record in _records_for_output(df):
            print(json.dumps(record, default=str, allow_nan=False))
    elif fmt == "tsv":
        print(df.to_csv(sep="\t", index=False), end="")
    else:  # table
        print(df.to_string(index=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Query canonical Jouvence KG Parquets")
    parser.add_argument(
        "--kg-root",
        default=str(DEFAULT_KG_ROOT),
        help="KG root containing nodes/, edges/, evidence/ (default: FUSE/GCS root or JOUVENCE_KG_ROOT)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_gene_args(p: argparse.ArgumentParser) -> None:
        group = p.add_mutually_exclusive_group(required=True)
        group.add_argument("--gene", help="Gene id, symbol, or name convenience query")
        group.add_argument("--gene-id", help="Canonical/cross-reference gene id")
        group.add_argument("--gene-name", help="Gene symbol/name, e.g. BRCA1")
        p.add_argument("--include-non-human", action="store_true", help="Do not filter obvious ortholog rows")
        p.add_argument("--limit", type=int, default=None)
        p.add_argument("--format", choices=["table", "tsv", "json", "jsonl"], default="table")

    resolve = sub.add_parser("resolve-gene", help="Resolve a gene query to canonical KG ids")
    add_gene_args(resolve)

    diseases = sub.add_parser("diseases-for-gene", help="List diseases associated with a gene")
    add_gene_args(diseases)
    diseases.add_argument("--no-evidence", action="store_true", help="Skip evidence summary join")
    diseases.add_argument(
        "--pilot-db",
        help="Optional local KGEdge/KGEdgeEvidence pilot SQLite; falls back to Parquet when unpopulated",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    common = {
        "identifier": args.gene,
        "gene_id": args.gene_id,
        "gene_name": args.gene_name,
        "kg_root": args.kg_root,
        "human_only": not args.include_non_human,
        "limit": args.limit,
    }
    if args.command == "resolve-gene":
        df = resolve_gene(**common)
    elif args.command == "diseases-for-gene":
        df = diseases_for_gene(**common, include_evidence=not args.no_evidence, pilot_db=args.pilot_db)
    else:  # pragma: no cover - argparse enforces choices
        parser.error(f"unknown command: {args.command}")
    _print_frame(df, args.format)
    return 0 if not df.empty else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
