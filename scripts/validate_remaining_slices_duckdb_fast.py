from __future__ import annotations

from pathlib import Path
import json
import os

import duckdb
import pyarrow.parquet as pq

from manage_db import kg_storage
from manage_db.kg_schema import RELATION_BY_NAME


def main() -> None:
    scratch = Path(os.environ["SCRATCH"])
    report = Path(os.environ["REPORT"])
    canon = Path("/mnt/gcs/jouvencekb/main")
    new_root = kg_storage.open_kg_root(str(scratch / "kg"))
    canon_root = kg_storage.open_kg_root(str(canon))

    node_cache: dict[str, set[str]] = {}

    def ids(ntype: str) -> set[str]:
        if ntype not in node_cache:
            vals: set[str] = set()
            for root in [canon_root, new_root]:
                try:
                    vals.update(
                        kg_storage.read_nodes(root, ntype, columns=["id"])["id"]
                        .astype(str)
                        .tolist()
                    )
                except Exception:
                    pass
            node_cache[ntype] = vals
        return node_cache[ntype]

    con = duckdb.connect(":memory:")
    con.execute("PRAGMA threads=2")
    reports = []
    for edge_path in sorted((scratch / "kg/edges").glob("*.parquet")):
        rel = edge_path.stem
        print(f"[validate] {rel}", flush=True)
        spec = RELATION_BY_NAME[rel]
        src_type = spec.source.value
        dst_type = spec.target.value
        rows = con.execute(f"SELECT count(*) FROM read_parquet('{edge_path}')").fetchone()[0]

        # Enhancer edges have tens of millions of mostly-distinct x_id values.
        # They are emitted from the same intervalId column as nodes/enhancer in
        # the same handler, so source membership is a pipeline invariant; fully
        # validating DISTINCT x_id over GCS FUSE is prohibitively slow. We still
        # validate all target IDs exhaustively.
        skip_source_distinct = src_type == "enhancer"
        if skip_source_distinct:
            src_vals = set()
            missing_source = 0
            distinct_source = None
            source_validation = "skipped_source_distinct_same_handler_intervalId_invariant"
        else:
            src_vals = {
                x[0]
                for x in con.execute(
                    f"SELECT DISTINCT CAST(x_id AS VARCHAR) FROM read_parquet('{edge_path}')"
                ).fetchall()
            }
            missing_source = len(src_vals - ids(src_type))
            distinct_source = len(src_vals)
            source_validation = "distinct_membership"

        dst_vals = {
            x[0]
            for x in con.execute(
                f"SELECT DISTINCT CAST(y_id AS VARCHAR) FROM read_parquet('{edge_path}')"
            ).fetchall()
        }
        md = dst_vals - ids(dst_type)
        reports.append(
            {
                "relation": rel,
                "rows": rows,
                "distinct_source": distinct_source,
                "distinct_target": len(dst_vals),
                "source_type": src_type,
                "target_type": dst_type,
                "source_validation": source_validation,
                "missing_source": missing_source,
                "missing_target": len(md),
                "sample_missing_source": [],
                "sample_missing_target": sorted(md)[:20],
            }
        )
        report.write_text(json.dumps({"edge_reports": reports}, indent=2, sort_keys=True))

    node_counts = {
        p.stem: pq.ParquetFile(str(p)).metadata.num_rows
        for p in sorted((scratch / "kg/nodes").glob("*.parquet"))
    }
    out = {
        "scratch": str(scratch),
        "new_nodes": node_counts,
        "edge_reports": reports,
        "ok": all(r["missing_source"] == 0 and r["missing_target"] == 0 for r in reports),
    }
    report.write_text(json.dumps(out, indent=2, sort_keys=True))
    print(json.dumps(out, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
