from __future__ import annotations

"""Shard-aware readers for the canonical full ReMap CRM support sidecar.

The promoted full sidecar is support-only feature/QA material under
``features/remap_crm_tf_enhancer_support_full/``.  It is deliberately not graph
topology: not ``edges/tf_binds_enhancer.parquet``, not evidence, not observed
binding, and not inferred edges.  These helpers keep reads chromosome-sharded and
bounded so downstream users do not need, or accidentally create, a monolithic
Parquet.
"""

import argparse
import glob
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import fsspec
import pandas as pd
import pyarrow.parquet as pq

DEFAULT_KG_ROOT = os.environ.get("JOUVENCE_KG_ROOT", "/Users/jkobject/mnt/gcs/jouvencekb/main")
DEFAULT_GCS_PREFIX = "gs://jouvencekb/main/features/remap_crm_tf_enhancer_support_full"
DEFAULT_FUSE_PREFIX = str(Path(DEFAULT_KG_ROOT) / "features" / "remap_crm_tf_enhancer_support_full")
CHROMOSOME_ORDER = [str(i) for i in range(1, 23)] + ["X", "Y"]
SUMMARY_COLUMNS = [
    "feature_table",
    "support_entity_type",
    "tf_gene_id",
    "tf_symbol_sample",
    "enhancer_id",
    "enhancer_chromosome",
    "enhancer_start",
    "enhancer_end",
    "support_entity_id",
    "support_entity_label",
    "crm_support_rows",
    "crm_interval_count",
    "evidence_type",
    "support_semantics",
    "relation_under_review",
    "support_scope",
    "source",
    "source_release",
    "source_url",
    "genome_build",
    "liftover_performed",
    "aggregation_policy",
    "provenance_caveat",
    "source_task_id",
    "promotion_task_id",
    "source_report",
    "readiness_decision_doc",
]
TF_GLOBAL_COLUMNS = SUMMARY_COLUMNS
SEMANTICS_NOTE = (
    "canonical promoted full support sidecar; support-only feature/QA material; "
    "not edges/tf_binds_enhancer.parquet; not evidence; not observed binding; not inferred edge"
)


@dataclass(frozen=True)
class SidecarLocation:
    """Resolved location for sidecar reads."""

    prefix: str
    source: str


def default_prefix() -> str:
    """Return the healthy local FUSE prefix when present, otherwise the GCS URI."""

    if Path(DEFAULT_FUSE_PREFIX).exists():
        return DEFAULT_FUSE_PREFIX
    return DEFAULT_GCS_PREFIX


def _strip_slash(value: str | Path) -> str:
    return str(value).rstrip("/")


def _is_cloud_uri(value: str) -> bool:
    return "://" in value


def _url_to_fs(path: str):
    fs, fs_path = fsspec.core.url_to_fs(path)
    return fs, fs_path


def _exists(path: str) -> bool:
    if _is_cloud_uri(path):
        fs, fs_path = _url_to_fs(path)
        return fs.exists(fs_path)
    return Path(path).exists()


def _glob(path: str) -> list[str]:
    if _is_cloud_uri(path):
        fs, fs_path = _url_to_fs(path)
        return sorted(fs.glob(fs_path))
    return sorted(glob.glob(path))


def _format_path(prefix: str, name: str) -> str:
    prefix = _strip_slash(prefix)
    sep = "/"
    return f"{prefix}{sep}{name}"


def _apply_filters(df: pd.DataFrame, filters) -> pd.DataFrame:
    if not filters or df.empty:
        return df
    mask = pd.Series(True, index=df.index)
    for col, op, value in filters:
        if op == "==":
            mask &= df[col] == value
        elif op == "in":
            mask &= df[col].isin(list(value))
        else:  # pragma: no cover - internal helper only emits supported ops
            raise ValueError(f"Unsupported filter operator: {op}")
    return df[mask]


def _limited_read_parquet(
    path: str,
    *,
    columns: Sequence[str] | None = None,
    filters=None,
    limit: int,
) -> pd.DataFrame:
    if limit < 0:
        raise ValueError("limit must be non-negative")
    if limit == 0:
        return pd.DataFrame(columns=list(columns) if columns is not None else None)
    if _is_cloud_uri(path):
        fs, fs_path = _url_to_fs(path)
        parquet_file = pq.ParquetFile(fs.open(fs_path, "rb"))
    else:
        parquet_file = pq.ParquetFile(path)
    frames: list[pd.DataFrame] = []
    rows = 0
    for batch in parquet_file.iter_batches(columns=list(columns) if columns is not None else None, batch_size=65536):
        filtered = _apply_filters(batch.to_pandas(), filters)
        if filtered.empty:
            continue
        need = limit - rows
        frames.append(filtered.head(need))
        rows += len(frames[-1])
        if rows >= limit:
            break
    if not frames:
        return pd.DataFrame(columns=list(columns) if columns is not None else None)
    return pd.concat(frames, ignore_index=True)


def _read_parquet(
    path: str,
    *,
    columns: Sequence[str] | None = None,
    filters=None,
    limit: int | None = None,
) -> pd.DataFrame:
    if not _exists(path):
        raise FileNotFoundError(f"Parquet not found: {path}")
    if limit is not None:
        return _limited_read_parquet(path, columns=columns, filters=filters, limit=limit)
    return pd.read_parquet(path, columns=list(columns) if columns is not None else None, filters=filters, engine="pyarrow")


def _limit_frame(df: pd.DataFrame, limit: int | None) -> pd.DataFrame:
    if limit is None:
        return df
    if limit < 0:
        raise ValueError("limit must be non-negative")
    return df.head(limit).reset_index(drop=True)


def _normalise_chromosome(chromosome: str | int) -> str:
    value = str(chromosome).removeprefix("chr").removeprefix("CHR")
    if value not in set(CHROMOSOME_ORDER):
        raise ValueError(f"Unsupported chromosome {chromosome!r}; expected one of {CHROMOSOME_ORDER}")
    return value


def chromosome_shard_path(chromosome: str | int, *, prefix: str | Path | None = None) -> str:
    """Return the path/URI for one chromosome summary shard."""

    chrom = _normalise_chromosome(chromosome)
    return _format_path(_strip_slash(prefix or default_prefix()), f"summary_chr{chrom}.parquet")


def tf_global_summary_path(*, prefix: str | Path | None = None) -> str:
    """Return the path/URI for ``tf_global_summary.parquet``."""

    return _format_path(_strip_slash(prefix or default_prefix()), "tf_global_summary.parquet")


def list_chromosomes(*, prefix: str | Path | None = None) -> list[str]:
    """List available chromosome shards without reading their contents."""

    root = _strip_slash(prefix or default_prefix())
    matches = _glob(_format_path(root, "summary_chr*.parquet"))
    chroms: set[str] = set()
    for match in matches:
        name = Path(str(match)).name
        found = re.fullmatch(r"summary_chr(.+)\.parquet", name)
        if found:
            chroms.add(found.group(1))
    return [chrom for chrom in CHROMOSOME_ORDER if chrom in chroms]


def location_status(*, prefix: str | Path | None = None) -> SidecarLocation:
    """Return the prefix selected for reads and whether it is local/FUSE or cloud."""

    resolved = _strip_slash(prefix or default_prefix())
    source = "gcs" if _is_cloud_uri(resolved) else "local_or_fuse"
    return SidecarLocation(prefix=resolved, source=source)


def _summary_filters(
    *,
    tf_gene_id: str | None = None,
    tf_symbol: str | None = None,
    enhancer_id: str | None = None,
    support_entity_type: str | None = None,
):
    filters: list[tuple[str, str, object]] = []
    if tf_gene_id:
        filters.append(("tf_gene_id", "==", tf_gene_id))
    if tf_symbol:
        filters.append(("tf_symbol_sample", "==", tf_symbol))
    if enhancer_id:
        filters.append(("enhancer_id", "==", enhancer_id))
    if support_entity_type:
        filters.append(("support_entity_type", "==", support_entity_type))
    return filters or None


def read_chromosome_support(
    chromosome: str | int,
    *,
    prefix: str | Path | None = None,
    tf_gene_id: str | None = None,
    tf_symbol: str | None = None,
    enhancer_id: str | None = None,
    support_entity_type: str | None = None,
    columns: Sequence[str] | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    """Read one chromosome shard with optional TF/enhancer filters.

    This function never reads all 24 shards.  Use ``limit`` during exploration;
    omit it only when a full single-chromosome read is intentional.
    """

    path = chromosome_shard_path(chromosome, prefix=prefix)
    wanted_columns = list(columns) if columns is not None else None
    filter_columns = ["tf_gene_id", "tf_symbol_sample", "enhancer_id", "support_entity_type"]
    if wanted_columns is not None:
        # PyArrow needs filtered columns present in the scan even if the caller
        # does not want them in the final output.
        for col in filter_columns:
            if col not in wanted_columns:
                wanted_columns.append(col)
    df = _read_parquet(
        path,
        columns=wanted_columns,
        filters=_summary_filters(
            tf_gene_id=tf_gene_id,
            tf_symbol=tf_symbol,
            enhancer_id=enhancer_id,
            support_entity_type=support_entity_type,
        ),
        limit=limit,
    )
    if columns is not None:
        df = df[list(columns)]
    return df.reset_index(drop=True)


def read_tf_global_summary(
    *,
    prefix: str | Path | None = None,
    tf_gene_id: str | None = None,
    tf_symbol: str | None = None,
    columns: Sequence[str] | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    """Read the small global TF summary table, optionally filtered."""

    filters: list[tuple[str, str, object]] = []
    if tf_gene_id:
        filters.append(("tf_gene_id", "==", tf_gene_id))
    if tf_symbol:
        filters.append(("tf_symbol_sample", "==", tf_symbol))
    wanted_columns = list(columns) if columns is not None else None
    if wanted_columns is not None:
        # Manual bounded reads apply filters after reading batches, so filter
        # columns must be present internally even when callers request a
        # narrower output projection.
        for col in ("tf_gene_id", "tf_symbol_sample"):
            if col not in wanted_columns:
                wanted_columns.append(col)
    df = _read_parquet(tf_global_summary_path(prefix=prefix), columns=wanted_columns, filters=filters or None, limit=limit)
    if columns is not None:
        df = df[list(columns)]
    return df.reset_index(drop=True)


def _endpoint_path(kg_root: str | Path, endpoint: str) -> str:
    return _format_path(_strip_slash(kg_root), f"nodes/{endpoint}.parquet")


def check_loaded_endpoint_membership(
    support_df: pd.DataFrame,
    *,
    kg_root: str | Path | None = None,
    check_tfs: bool = True,
    check_enhancers: bool = True,
) -> dict[str, object]:
    """Bounded endpoint check for IDs already present in a loaded result.

    The function only checks distinct IDs in ``support_df``. It is intended for
    sampled reads or one filtered shard, not all-shard validation.
    """

    root = _strip_slash(kg_root or DEFAULT_KG_ROOT)
    result: dict[str, object] = {
        "semantics": SEMANTICS_NOTE,
        "loaded_rows": int(len(support_df)),
        "kg_root": root,
    }
    if check_tfs:
        tf_ids = sorted(str(x) for x in support_df.get("tf_gene_id", pd.Series(dtype=object)).dropna().unique())
        if tf_ids:
            genes = _read_parquet(_endpoint_path(root, "gene"), columns=["id"], filters=[("id", "in", tf_ids)])
            found = set(genes["id"].astype(str))
            missing = [x for x in tf_ids if x not in found]
            result["tf_gene_ids_checked"] = len(tf_ids)
            result["tf_gene_endpoint_antijoin"] = len(missing)
            result["tf_gene_missing_sample"] = missing[:10]
        else:
            result["tf_gene_ids_checked"] = 0
            result["tf_gene_endpoint_antijoin"] = 0
            result["tf_gene_missing_sample"] = []
    if check_enhancers:
        enhancer_ids = sorted(str(x) for x in support_df.get("enhancer_id", pd.Series(dtype=object)).dropna().unique())
        if enhancer_ids:
            enhancers = _read_parquet(_endpoint_path(root, "enhancer"), columns=["id"], filters=[("id", "in", enhancer_ids)])
            found = set(enhancers["id"].astype(str))
            missing = [x for x in enhancer_ids if x not in found]
            result["enhancer_ids_checked"] = len(enhancer_ids)
            result["enhancer_endpoint_antijoin"] = len(missing)
            result["enhancer_missing_sample"] = missing[:10]
        else:
            result["enhancer_ids_checked"] = 0
            result["enhancer_endpoint_antijoin"] = 0
            result["enhancer_missing_sample"] = []
    return result


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
    else:
        print(df.to_string(index=False))


def _parse_columns(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [part.strip() for part in value.split(",") if part.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Shard-aware read-only queries for the canonical ReMap CRM full support sidecar"
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help=f"Sidecar prefix (default: FUSE if healthy, else {DEFAULT_GCS_PREFIX})",
    )
    parser.add_argument("--kg-root", default=DEFAULT_KG_ROOT, help="Canonical KG root for bounded endpoint checks")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="Show selected prefix and support-only semantics")
    sub.add_parser("list-chromosomes", help="List available summary_chr*.parquet shards")

    chrom = sub.add_parser("read-chromosome", help="Read one chromosome shard with optional filters")
    chrom.add_argument("--chromosome", required=True, help="Chromosome shard, e.g. 1, chr1, X")
    chrom.add_argument("--tf-gene-id")
    chrom.add_argument("--tf-symbol")
    chrom.add_argument("--enhancer-id")
    chrom.add_argument("--support-entity-type", choices=["tf", "enhancer"])
    chrom.add_argument("--columns", help="Comma-separated output columns")
    chrom.add_argument("--limit", type=int, default=20)
    chrom.add_argument("--format", choices=["table", "tsv", "json", "jsonl"], default="table")

    summary = sub.add_parser("tf-global-summary", help="Read tf_global_summary.parquet")
    summary.add_argument("--tf-gene-id")
    summary.add_argument("--tf-symbol")
    summary.add_argument("--columns", help="Comma-separated output columns")
    summary.add_argument("--limit", type=int, default=20)
    summary.add_argument("--format", choices=["table", "tsv", "json", "jsonl"], default="table")

    check = sub.add_parser("check-endpoints", help="Bounded endpoint membership check over one filtered chromosome read")
    check.add_argument("--chromosome", required=True)
    check.add_argument("--tf-gene-id")
    check.add_argument("--tf-symbol")
    check.add_argument("--enhancer-id")
    check.add_argument("--support-entity-type", choices=["tf", "enhancer"])
    check.add_argument("--limit", type=int, default=1000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    prefix = args.prefix or default_prefix()
    if args.command == "status":
        status = location_status(prefix=prefix)
        print(json.dumps({"prefix": status.prefix, "source": status.source, "semantics": SEMANTICS_NOTE}, indent=2))
        return 0
    if args.command == "list-chromosomes":
        print(json.dumps({"prefix": _strip_slash(prefix), "chromosomes": list_chromosomes(prefix=prefix)}, indent=2))
        return 0
    if args.command == "read-chromosome":
        df = read_chromosome_support(
            args.chromosome,
            prefix=prefix,
            tf_gene_id=args.tf_gene_id,
            tf_symbol=args.tf_symbol,
            enhancer_id=args.enhancer_id,
            support_entity_type=args.support_entity_type,
            columns=_parse_columns(args.columns),
            limit=args.limit,
        )
        _print_frame(df, args.format)
        return 0 if not df.empty else 1
    if args.command == "tf-global-summary":
        df = read_tf_global_summary(
            prefix=prefix,
            tf_gene_id=args.tf_gene_id,
            tf_symbol=args.tf_symbol,
            columns=_parse_columns(args.columns),
            limit=args.limit,
        )
        _print_frame(df, args.format)
        return 0 if not df.empty else 1
    if args.command == "check-endpoints":
        df = read_chromosome_support(
            args.chromosome,
            prefix=prefix,
            tf_gene_id=args.tf_gene_id,
            tf_symbol=args.tf_symbol,
            enhancer_id=args.enhancer_id,
            support_entity_type=args.support_entity_type,
            columns=["tf_gene_id", "enhancer_id"],
            limit=args.limit,
        )
        print(json.dumps(check_loaded_endpoint_membership(df, kg_root=args.kg_root), indent=2, default=str))
        return 0
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
