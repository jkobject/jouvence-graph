#!/usr/bin/env python3
"""Build bounded Template B inferred disease_associated_protein artifacts.

Template B:
    mutation_causes_protein_change(mutation, protein)
    + mutation_associated_disease(mutation, disease)
    => inferred/support-only disease_associated_protein(protein, disease)

The output is staging-only and deliberately uses edges_inferred/evidence_inferred
semantics. It never writes to canonical v2/edges or v2/evidence.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb

DEFAULT_KG_ROOT = Path("/Users/jkobject/mnt/gcs/jouvencekb/main")
DEFAULT_OUTPUT_ROOT = Path("artifacts/staged/t_38eef2b7")
RELATION = "disease_associated_protein"
DISPLAY_RELATION = "associated with"
TEMPLATE_ID = "mutation_protein_disease_v1"
TEMPLATE_VERSION = "1.0.0"
INFERENCE_LABEL = "inferred_weak"
SUPPORT_RELATIONS = "mutation_causes_protein_change|mutation_associated_disease"


def parquet_path(root: Path, subdir: str, name: str) -> Path:
    path = root / subdir / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def sql_path(path: Path) -> str:
    return str(path).replace("'", "''")


def scalar(con: duckdb.DuckDBPyConnection, sql: str) -> Any:
    row = con.execute(sql).fetchone()
    if row is None:
        raise RuntimeError(f"Scalar query returned no rows: {sql}")
    return row[0]


def rows(con: duckdb.DuckDBPyConnection, sql: str) -> list[dict[str, Any]]:
    return con.execute(sql).fetchdf().to_dict(orient="records")


def build(args: argparse.Namespace) -> dict[str, Any]:
    kg_root = args.kg_root.resolve()
    output_root = args.output_root.resolve()
    edge_out = output_root / "edges_inferred" / RELATION / f"{TEMPLATE_ID}.parquet"
    evidence_out = output_root / "evidence_inferred" / RELATION / f"{TEMPLATE_ID}.parquet"
    manifest_out = output_root / "manifest" / f"{TEMPLATE_ID}.json"
    report_out = Path(args.report).resolve() if args.report else output_root / "reports" / f"{TEMPLATE_ID}_report.json"

    for path in [edge_out.parent, evidence_out.parent, manifest_out.parent, report_out.parent]:
        path.mkdir(parents=True, exist_ok=True)

    paths = {
        "mp_edge": parquet_path(kg_root, "edges", "mutation_causes_protein_change"),
        "md_edge": parquet_path(kg_root, "edges", "mutation_associated_disease"),
        "mp_evidence": parquet_path(kg_root, "evidence", "mutation_causes_protein_change"),
        "md_evidence": parquet_path(kg_root, "evidence", "mutation_associated_disease"),
        "protein_nodes": parquet_path(kg_root, "nodes", "protein"),
        "disease_nodes": parquet_path(kg_root, "nodes", "disease"),
    }
    observed_target = kg_root / "edges" / f"{RELATION}.parquet"
    observed_target_exists = observed_target.exists()

    created_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    derivation_query = (
        "anchor_mutations = first N mutation_associated_disease.x_id ordered by x_id; "
        "join mutation_causes_protein_change.x_id = mutation_associated_disease.x_id; "
        "group by protein_id,disease_id; retain support mutation/evidence rows"
    )

    con = duckdb.connect(database=":memory:")
    con.execute(f"PRAGMA threads={int(args.threads)}")
    con.execute("PRAGMA preserve_insertion_order=false")

    con.execute(
        f"""
        CREATE TEMP TABLE anchor_mutations AS
        SELECT DISTINCT x_id AS mutation_id
        FROM read_parquet('{sql_path(paths['md_edge'])}')
        WHERE x_id IS NOT NULL AND y_id IS NOT NULL
        ORDER BY x_id
        LIMIT {int(args.mutation_limit)}
        """
    )

    con.execute(
        f"""
        CREATE TEMP TABLE support_paths AS
        SELECT DISTINCT
            mp.y_id::VARCHAR AS protein_id,
            md.y_id::VARCHAR AS disease_id,
            mp.x_id::VARCHAR AS support_mutation_id,
            mp.x_type::VARCHAR AS mutation_x_type,
            mp.y_type::VARCHAR AS protein_y_type,
            md.y_type::VARCHAR AS disease_y_type,
            mp.source::VARCHAR AS protein_change_source,
            md.source::VARCHAR AS mutation_disease_source,
            mp.credibility::BIGINT AS protein_change_credibility,
            md.credibility::BIGINT AS mutation_disease_credibility,
            mp.amino_acid_change::VARCHAR AS amino_acid_change,
            mp.uniprot_id::VARCHAR AS uniprot_id,
            md.score::DOUBLE AS mutation_disease_score,
            md.datatype::VARCHAR AS mutation_disease_datatype,
            md.studyLocusId::VARCHAR AS mutation_disease_study_locus_id,
            coalesce(mpe.edge_key, sha256('mutation_causes_protein_change|' || mp.x_id || '|' || mp.y_id))::VARCHAR AS protein_change_edge_key,
            mpe.source_record_id::VARCHAR AS protein_change_source_record_id,
            mpe.predicate::VARCHAR AS protein_change_predicate,
            coalesce(mde.edge_key, sha256('mutation_associated_disease|' || md.x_id || '|' || md.y_id))::VARCHAR AS mutation_disease_edge_key,
            mde.source_record_id::VARCHAR AS mutation_disease_source_record_id,
            mde.evidence_score::DOUBLE AS mutation_disease_evidence_score,
            mde.predicate::VARCHAR AS mutation_disease_predicate,
            sha256(
                '{TEMPLATE_ID}|' || mp.y_id || '|' || md.y_id || '|' || mp.x_id || '|' ||
                coalesce(mpe.edge_key, '') || '|' || coalesce(mde.edge_key, '')
            )::VARCHAR AS support_path_hash
        FROM read_parquet('{sql_path(paths['mp_edge'])}') mp
        JOIN anchor_mutations am ON am.mutation_id = mp.x_id
        JOIN read_parquet('{sql_path(paths['md_edge'])}') md ON md.x_id = mp.x_id
        LEFT JOIN read_parquet('{sql_path(paths['mp_evidence'])}') mpe
          ON mpe.relation = 'mutation_causes_protein_change'
         AND mpe.x_id = mp.x_id AND mpe.y_id = mp.y_id
        LEFT JOIN read_parquet('{sql_path(paths['md_evidence'])}') mde
          ON mde.relation = 'mutation_associated_disease'
         AND mde.x_id = md.x_id AND mde.y_id = md.y_id
        WHERE mp.y_id IS NOT NULL AND md.y_id IS NOT NULL
        """
    )

    overlap_select = "false AS canonical_observed_overlap"
    overlap_join = ""
    if observed_target_exists:
        overlap_select = "(obs.x_id IS NOT NULL) AS canonical_observed_overlap"
        overlap_join = (
            f"LEFT JOIN read_parquet('{sql_path(observed_target)}') obs "
            "ON obs.x_id = agg.protein_id AND obs.y_id = agg.disease_id"
        )

    con.execute(
        f"""
        CREATE TEMP TABLE inferred_edges AS
        WITH agg AS (
            SELECT
                protein_id,
                disease_id,
                count(DISTINCT support_mutation_id) AS support_count,
                string_agg(DISTINCT support_path_hash, '|' ORDER BY support_path_hash) AS support_edge_ids_or_hashes,
                string_agg(DISTINCT protein_change_source, '|' ORDER BY protein_change_source) || ' + ' ||
                    string_agg(DISTINCT mutation_disease_source, '|' ORDER BY mutation_disease_source) AS support_sources,
                min(least(protein_change_credibility, mutation_disease_credibility)) AS support_min_credibility,
                max(mutation_disease_score) AS support_max_mutation_disease_score,
                count(DISTINCT mutation_disease_source_record_id) AS support_mutation_disease_evidence_count,
                count(DISTINCT protein_change_source_record_id) AS support_protein_change_evidence_count
            FROM support_paths
            GROUP BY 1, 2
        )
        SELECT
            sha256('{RELATION}|' || agg.protein_id || '|' || agg.disease_id || '|{TEMPLATE_ID}')::VARCHAR AS edge_key,
            agg.protein_id AS x_id,
            'protein' AS x_type,
            agg.disease_id AS y_id,
            'disease' AS y_type,
            '{RELATION}' AS relation,
            '{DISPLAY_RELATION}' AS display_relation,
            '{INFERENCE_LABEL}' AS inference_label,
            '{TEMPLATE_ID}' AS inference_template_id,
            '{TEMPLATE_VERSION}' AS inference_template_version,
            agg.support_edge_ids_or_hashes,
            '{SUPPORT_RELATIONS}' AS support_relations,
            agg.support_count,
            agg.support_sources,
            agg.support_min_credibility,
            agg.support_max_mutation_disease_score,
            agg.support_mutation_disease_evidence_count,
            agg.support_protein_change_evidence_count,
            sha256('{derivation_query}')::VARCHAR AS derivation_query_hash,
            {overlap_select},
            'support_only_inferred_not_observed' AS layer_semantics,
            'exclude_from_observed_train_val_test_labels_by_default' AS leakage_policy,
            'bounded_first_100000_ordered_mutation_associated_disease_anchors' AS split_policy,
            {int(args.mutation_limit)}::BIGINT AS mutation_anchor_limit,
            '{created_at}' AS created_at,
            '{sql_path(kg_root)}' AS kg_snapshot_id
        FROM agg
        {overlap_join}
        ORDER BY support_count DESC, x_id, y_id
        """
    )

    con.execute(
        f"""
        CREATE TEMP TABLE inferred_evidence AS
        SELECT
            sha256('{RELATION}|' || protein_id || '|' || disease_id || '|{TEMPLATE_ID}')::VARCHAR AS inferred_edge_key,
            '{RELATION}' AS relation,
            protein_id AS x_id,
            'protein' AS x_type,
            disease_id AS y_id,
            'disease' AS y_type,
            'inferred_support_path' AS evidence_type,
            'inferred_from_canonical_kg' AS source,
            'Template B: mutation_causes_protein_change + mutation_associated_disease' AS source_dataset,
            support_path_hash AS source_record_id,
            '{INFERENCE_LABEL}' AS inference_label,
            '{TEMPLATE_ID}' AS inference_template_id,
            '{TEMPLATE_VERSION}' AS inference_template_version,
            support_mutation_id,
            protein_change_edge_key,
            mutation_disease_edge_key,
            protein_change_source_record_id,
            mutation_disease_source_record_id,
            amino_acid_change,
            uniprot_id,
            protein_change_predicate,
            mutation_disease_predicate,
            mutation_disease_score,
            mutation_disease_evidence_score,
            mutation_disease_datatype,
            mutation_disease_study_locus_id,
            protein_change_source,
            mutation_disease_source,
            protein_change_credibility,
            mutation_disease_credibility,
            support_path_hash,
            '{SUPPORT_RELATIONS}' AS support_relations,
            'support_only_inferred_not_observed' AS layer_semantics,
            'exclude_from_observed_train_val_test_labels_by_default' AS leakage_policy,
            '{created_at}' AS created_at,
            '{sql_path(kg_root)}' AS kg_snapshot_id
        FROM support_paths
        ORDER BY protein_id, disease_id, support_mutation_id, support_path_hash
        """
    )

    # Validation tables before writing.
    con.execute(
        f"""
        CREATE TEMP TABLE endpoint_misses AS
        SELECT 'protein' AS endpoint, x_id AS id
        FROM inferred_edges ie
        ANTI JOIN read_parquet('{sql_path(paths['protein_nodes'])}') p ON p.id = ie.x_id
        UNION ALL
        SELECT 'disease' AS endpoint, y_id AS id
        FROM inferred_edges ie
        ANTI JOIN read_parquet('{sql_path(paths['disease_nodes'])}') d ON d.id = ie.y_id
        """
    )

    con.execute(
        f"COPY inferred_edges TO '{sql_path(edge_out)}' (FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    con.execute(
        f"COPY inferred_evidence TO '{sql_path(evidence_out)}' (FORMAT PARQUET, COMPRESSION ZSTD)"
    )

    counts = {
        "anchor_mutation_count": scalar(con, "SELECT count(*) FROM anchor_mutations"),
        "support_path_rows": scalar(con, "SELECT count(*) FROM support_paths"),
        "edge_rows": scalar(con, "SELECT count(*) FROM inferred_edges"),
        "evidence_rows": scalar(con, "SELECT count(*) FROM inferred_evidence"),
        "distinct_support_mutations": scalar(con, "SELECT count(DISTINCT support_mutation_id) FROM support_paths"),
        "duplicate_edge_keys": scalar(
            con,
            "SELECT count(*) FROM (SELECT edge_key FROM inferred_edges GROUP BY edge_key HAVING count(*) > 1)",
        ),
        "protein_endpoint_misses": scalar(con, "SELECT count(*) FROM endpoint_misses WHERE endpoint = 'protein'"),
        "disease_endpoint_misses": scalar(con, "SELECT count(*) FROM endpoint_misses WHERE endpoint = 'disease'"),
        "canonical_observed_target_file_exists": observed_target_exists,
        "canonical_observed_overlap_rows": scalar(
            con, "SELECT count(*) FROM inferred_edges WHERE canonical_observed_overlap"
        ),
    }

    top_examples = rows(
        con,
        f"""
        SELECT x_id AS protein_id, y_id AS disease_id, support_count,
               support_sources, support_max_mutation_disease_score,
               canonical_observed_overlap
        FROM inferred_edges
        ORDER BY support_count DESC, protein_id, disease_id
        LIMIT {int(args.sample_limit)}
        """,
    )

    manifest = {
        "task_id": args.task_id,
        "status": "staged-only/support-only/review-required",
        "kg_root": str(kg_root),
        "created_at": created_at,
        "template": {
            "id": TEMPLATE_ID,
            "version": TEMPLATE_VERSION,
            "chain": "mutation_causes_protein_change(mutation,protein) + mutation_associated_disease(mutation,disease) => inferred disease_associated_protein(protein,disease)",
            "confidence_label": INFERENCE_LABEL,
            "support_relations": SUPPORT_RELATIONS.split("|"),
            "derivation_query_hash": scalar(con, f"SELECT sha256('{derivation_query}')"),
        },
        "bounds": {
            "mutation_anchor_policy": "first N distinct mutation_associated_disease.x_id ordered by x_id",
            "mutation_limit": int(args.mutation_limit),
            "anchor_mutation_count": counts["anchor_mutation_count"],
        },
        "artifacts": {
            "edges_inferred": str(edge_out),
            "evidence_inferred": str(evidence_out),
            "manifest": str(manifest_out),
            "report": str(report_out),
        },
        "gnn_leakage_policy": {
            "default_use": "do_not_mix_into_observed_train_val_test_labels",
            "allowed_use": ["inferred-only ablation", "observed+inferred explicit ablation", "candidate ranking/support feature"],
            "target_relation_family": RELATION,
            "support_relations": SUPPORT_RELATIONS.split("|"),
            "split_note": "This bounded artifact is generated from the canonical snapshot before any split. For label-sensitive GNN evaluation, regenerate split-specific inferred artifacts from training-permitted support edges only.",
        },
        "observed_overlap_handling": {
            "canonical_observed_target_file": str(observed_target),
            "canonical_observed_target_file_exists": observed_target_exists,
            "canonical_observed_overlap_rows": counts["canonical_observed_overlap_rows"],
            "policy": "direct protein-native observed edges win; inferred candidates remain separate and are not canonical observed evidence",
        },
        "counts": counts,
        "top_examples": top_examples,
        "validation": {
            "endpoint_validation": {
                "protein_endpoint_misses": counts["protein_endpoint_misses"],
                "disease_endpoint_misses": counts["disease_endpoint_misses"],
            },
            "duplicate_edge_keys": counts["duplicate_edge_keys"],
        },
        "commands": args.commands,
        "risks": [
            "inferred_weak graph-derived hypotheses, not source-native protein-disease observations",
            "bounded 100k ordered mutation-associated-disease anchor policy; not a full unbounded inferred layer",
            "split-specific leakage-safe GNN artifacts must be regenerated from training-permitted support edges for formal evaluation",
        ],
    }

    manifest_out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    report_out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kg-root", type=Path, default=DEFAULT_KG_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report", default=None)
    parser.add_argument("--task-id", default="t_38eef2b7")
    parser.add_argument("--mutation-limit", type=int, default=100_000)
    parser.add_argument("--sample-limit", type=int, default=10)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--commands", action="append", default=[])
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    manifest = build(args)
    if args.json:
        print(json.dumps(manifest, indent=2, sort_keys=True))
    else:
        print(f"Wrote {manifest['artifacts']['edges_inferred']}")
        print(f"Wrote {manifest['artifacts']['evidence_inferred']}")
        print(f"Wrote {manifest['artifacts']['manifest']}")
        print(f"Wrote {manifest['artifacts']['report']}")
        print(json.dumps(manifest["counts"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
