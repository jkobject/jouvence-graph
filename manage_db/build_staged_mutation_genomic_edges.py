"""Stage policy-bounded mutation genomic direct relations.

This builder is intentionally staged-only.  It implements the promotion policy in
``docs/mutation_genomic_relations_promotion_policy.md`` instead of blindly
promoting the earlier dense pilot:

* ``mutation_in_gene`` is limited to transcript/gene-local VEP consequence
  classes, canonical mutation/gene endpoints, and an independent OpenTargets
  ``target.genomicLocation`` point-in-gene containment proof.
* ``mutation_affects_transcript`` is limited to allowed transcript-local
  consequence classes, canonical mutation/transcript endpoints, and canonical
  transcript consequence rows when that flag is available.
* ``mutation_overlaps_enhancer`` keeps the downstream-support gate and performs
  exact mutation point -> enhancer interval overlap against current canonical
  enhancer nodes.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import tempfile
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from manage_db import kg_evidence
from manage_db.credibility import Credibility
from manage_db.kg_schema import NodeType
from manage_db.kg_storage import EDGE_PARQUET_COLUMNS

OT_RELEASE = "26.03"
OT_OUTPUT = f"https://ftp.ebi.ac.uk/pub/databases/opentargets/platform/{OT_RELEASE}/output"
ASSOCIATION_RELATIONS = (
    "mutation_associated_disease",
    "mutation_associated_phenotype",
    "mutation_affects_molecule_response",
    "mutation_associated_gene",
    "mutation_causes_protein_change",
)
EDGE_COLUMNS = [name for name, _ in EDGE_PARQUET_COLUMNS]
EVIDENCE_COLUMNS = [name for name, _ in kg_evidence.EVIDENCE_PARQUET_COLUMNS]

# Sequence Ontology classes allowed by the t_60b3e504 policy.  OpenTargets VEP
# rows provide SO IDs; labels are kept here for audit/report readability.
ALLOWED_CONSEQUENCE_LABELS = {
    "transcript_ablation",
    "splice_acceptor_variant",
    "splice_donor_variant",
    "stop_gained",
    "frameshift_variant",
    "stop_lost",
    "start_lost",
    "transcript_amplification",
    "inframe_insertion",
    "inframe_deletion",
    "missense_variant",
    "protein_altering_variant",
    "splice_region_variant",
    "incomplete_terminal_codon_variant",
    "start_retained_variant",
    "stop_retained_variant",
    "synonymous_variant",
    "coding_sequence_variant",
    "5_prime_UTR_variant",
    "3_prime_UTR_variant",
    "non_coding_transcript_exon_variant",
    "intron_variant",
    "NMD_transcript_variant",
    "non_coding_transcript_variant",
    "mature_miRNA_variant",
}
ALLOWED_CONSEQUENCE_IDS = {
    "SO_0001567",  # stop_retained_variant
    "SO_0001574",  # splice_acceptor_variant
    "SO_0001575",  # splice_donor_variant
    "SO_0001578",  # stop_lost
    "SO_0001580",  # coding_sequence_variant
    "SO_0001583",  # missense_variant
    "SO_0001587",  # stop_gained
    "SO_0001589",  # frameshift_variant
    "SO_0001619",  # non_coding_transcript_variant
    "SO_0001620",  # mature_miRNA_variant
    "SO_0001621",  # NMD_transcript_variant
    "SO_0001623",  # 5_prime_UTR_variant
    "SO_0001624",  # 3_prime_UTR_variant
    "SO_0001626",  # incomplete_terminal_codon_variant
    "SO_0001627",  # intron_variant
    "SO_0001630",  # splice_region_variant
    "SO_0001792",  # non_coding_transcript_exon_variant
    "SO_0001818",  # protein_altering_variant
    "SO_0001819",  # synonymous_variant
    "SO_0001821",  # inframe_insertion
    "SO_0001822",  # inframe_deletion
    "SO_0001889",  # transcript_amplification
    "SO_0001893",  # transcript_ablation
    "SO_0002012",  # start_lost
    "SO_0002019",  # start_retained_variant
}
EXCLUDED_CONSEQUENCE_IDS = {
    "SO_0001628",  # intergenic_variant
    "SO_0001631",  # upstream_gene_variant
    "SO_0001632",  # downstream_gene_variant
    "SO_0001566",  # regulatory_region_variant
    "SO_0001782",  # TF_binding_site_variant
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _list_ftp_parquets(dataset: str) -> list[str]:
    url = f"{OT_OUTPUT}/{dataset}/"
    try:
        html = urllib.request.urlopen(url, timeout=30).read().decode("utf-8", "ignore")
    except Exception:
        return []
    return [url + name for name in re.findall(r'href="([^"]+\.parquet)"', html)]


def is_url(value: str) -> bool:
    return urllib.parse.urlparse(value).scheme in {"http", "https"}


def materialize_variant_file(source: str, cache_dir: Path) -> Path:
    """Return a local Parquet path, downloading HTTP(S) sources if necessary."""
    if not is_url(source):
        return Path(source)
    cache_dir.mkdir(parents=True, exist_ok=True)
    name = Path(urllib.parse.urlparse(source).path).name
    out = cache_dir / name
    if not out.exists() or out.stat().st_size == 0:
        with urllib.request.urlopen(source, timeout=120) as response, out.open("wb") as fh:
            shutil.copyfileobj(response, fh)
    return out


def write_source_audit(stage_root: Path, variant_files: list[str], target_files: list[str], kg_cache_root: Path) -> dict[str, object]:
    datasets = [
        "variant",
        "enhancer_to_gene",
        "evidence_eva",
        "evidence_eva_somatic",
        "evidence_gwas_credible_sets",
        "evidence_uniprot_variants",
        "pharmacogenomics",
        "credible_set",
        "so",
    ]
    audit: dict[str, object] = {
        "created_at": _now(),
        "release": f"OpenTargets Platform {OT_RELEASE}",
        "policy_document": "docs/mutation_genomic_relations_promotion_policy.md",
        "source_policy": {
            "mutation_in_gene": "Filtered OpenTargets VEP transcriptConsequences targetId only after allowed transcript/gene-local SO classes, canonical mutation/gene endpoints, and independent point-in-gene containment against OpenTargets target.genomicLocation intervals from the same Platform release. Rows without a source-native target interval or with variant position outside the interval are rejected.",
            "mutation_affects_transcript": "Filtered OpenTargets VEP transcriptConsequences transcriptId only; allowed transcript-local SO classes; canonical transcript rows required when isEnsemblCanonical is present; canonical mutation/transcript endpoints required.",
            "mutation_overlaps_enhancer": "Exact coordinate overlap against current enhancer interval nodes only for variants with current downstream mutation support; no genome-wide unbounded overlap graph.",
        },
        "allowed_consequence_ids": sorted(ALLOWED_CONSEQUENCE_IDS),
        "allowed_consequence_labels": sorted(ALLOWED_CONSEQUENCE_LABELS),
        "excluded_consequence_ids": sorted(EXCLUDED_CONSEQUENCE_IDS),
        "downstream_support_relations": list(ASSOCIATION_RELATIONS),
        "official_opentargets_ftp": {dataset: len(_list_ftp_parquets(dataset)) for dataset in datasets},
        "variant_files": variant_files,
        "target_files": target_files,
        "kg_cache_root": str(kg_cache_root),
        "local_kg_cache": {},
    }
    for rel in ASSOCIATION_RELATIONS:
        p = kg_cache_root / "edges" / f"{rel}.parquet"
        entry: dict[str, object] = {"path": str(p), "exists": p.exists()}
        if p.exists():
            entry["rows"] = pq.ParquetFile(p).metadata.num_rows
        audit["local_kg_cache"][f"edges/{rel}.parquet"] = entry
    for node in ("gene", "transcript", "mutation", "enhancer"):
        p = kg_cache_root / "nodes" / f"{node}.parquet"
        entry = {"path": str(p), "exists": p.exists()}
        if p.exists():
            entry["rows"] = pq.ParquetFile(p).metadata.num_rows
            entry["columns"] = pq.ParquetFile(p).schema_arrow.names
        audit["local_kg_cache"][f"nodes/{node}.parquet"] = entry
    (stage_root / "source_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True))
    return audit


def _scalar_list(value: object) -> list[object]:
    if value is None or value is pd.NA:
        return []
    if isinstance(value, list):
        return value
    if hasattr(value, "tolist"):
        out = value.tolist()
        return out if isinstance(out, list) else [out]
    return [value]


def _first_rs(rs_ids: object) -> str | None:
    for rs in _scalar_list(rs_ids):
        if rs:
            return str(rs)
    return None


def _consequence_ids(consequence: dict[str, object]) -> list[str]:
    vals = consequence.get("variantFunctionalConsequenceIds")
    return [str(x) for x in _scalar_list(vals) if x]


def _variant_coord(row: dict[str, object]) -> tuple[str | None, int | None]:
    chrom = row.get("chromosome")
    pos = row.get("position")
    if chrom is None or pos is None or pd.isna(pos):
        vid = str(row.get("variantId") or "")
        m = re.match(r"^(?:chr)?([^_]+)_(\d+)_", vid)
        if not m:
            return None, None
        return m.group(1), int(m.group(2))
    return str(chrom).removeprefix("chr"), int(pos)


def _normalize_chromosome(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    return str(value).removeprefix("chr")


def _gene_interval_contains(gene_intervals: pd.DataFrame | None, gene_id: str, chrom: str | None, pos: int | None) -> tuple[bool, dict[str, object] | None, str]:
    """Check point-in-gene containment against source-native target intervals."""
    if gene_intervals is None or gene_intervals.empty:
        return False, None, "no_gene_interval_table"
    if chrom is None or pos is None:
        return False, None, "missing_variant_coordinate"
    matches = gene_intervals[gene_intervals["id"] == gene_id]
    if matches.empty:
        return False, None, "missing_gene_interval"
    norm_chrom = _normalize_chromosome(chrom)
    for row in matches.itertuples(index=False):
        row_chrom = _normalize_chromosome(getattr(row, "chromosome"))
        start = int(getattr(row, "start"))
        end = int(getattr(row, "end"))
        if row_chrom == norm_chrom and start <= int(pos) <= end:
            proof = {
                "gene_id": gene_id,
                "chromosome": row_chrom,
                "start": start,
                "end": end,
                "strand": getattr(row, "strand", None),
                "source_dataset": "OpenTargets/target.genomicLocation",
            }
            return True, proof, "contained"
    return False, None, "outside_gene_interval"


def _is_allowed_consequence(cons_ids: Iterable[str]) -> bool:
    """Return true only when every emitted SO id is explicitly allowlisted.

    OpenTargets/VEP rows may carry multiple SO ids.  A row with one allowed
    parent term plus an unreviewed child term is not policy-safe for direct graph
    edges: the source_audit allowlist is the complete approval surface for this
    builder, not a loose "any allowed id" matcher.
    """
    ids = {str(cid) for cid in cons_ids if cid}
    return bool(ids) and ids.issubset(ALLOWED_CONSEQUENCE_IDS)


def _is_canonical_transcript_consequence(consequence: dict[str, object]) -> bool:
    """Policy endpoint density filter.

    OpenTargets uses ``isEnsemblCanonical`` in observed rows.  If future source
    files omit this field entirely for a consequence row, do not invent a graph
    edge; keep such broad context out of staged canonical candidates.
    """
    return consequence.get("isEnsemblCanonical") is True


def _edge(x_id: str, x_type: str, y_id: str, y_type: str, relation: str, display: str, source: str) -> dict[str, object]:
    return {
        "x_id": x_id,
        "x_type": x_type,
        "y_id": y_id,
        "y_type": y_type,
        "relation": relation,
        "display_relation": display,
        "source": source,
        "credibility": int(Credibility.ESTABLISHED_FACT),
    }


def _evidence(edge: dict[str, object], source_dataset: str, source_record_id: str, predicate: str, *, text: dict[str, object] | None = None, score: float | None = None, method: str = "policy_bounded_source_native_builder") -> dict[str, object]:
    return {
        "relation": edge["relation"],
        "x_id": edge["x_id"],
        "x_type": edge["x_type"],
        "y_id": edge["y_id"],
        "y_type": edge["y_type"],
        "evidence_type": "database_record",
        "source": "OpenTargets",
        "source_dataset": source_dataset,
        "source_record_id": source_record_id,
        "evidence_score": score,
        "predicate": predicate,
        "text_span": json.dumps(text or {}, sort_keys=True),
        "extraction_method": method,
        "license": "OpenTargets Platform data",
        "release": OT_RELEASE,
        "created_at": _now(),
    }


def _empty_edges() -> dict[str, list[dict[str, object]]]:
    return {"mutation_in_gene": [], "mutation_affects_transcript": [], "mutation_overlaps_enhancer": []}


def build_edges_from_variants(
    variant_table: pa.Table,
    supported_for_enhancer: set[str],
    enhancer_nodes: pd.DataFrame | None,
    gene_intervals: pd.DataFrame | None = None,
    max_variants: int | None = None,
    *,
    canonical_mutations: set[str] | None = None,
    canonical_genes: set[str] | None = None,
    canonical_transcripts: set[str] | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], pd.DataFrame, dict[str, object]]:
    rows = variant_table.to_pylist()
    if max_variants is not None:
        rows = rows[:max_variants]

    edge_rows = _empty_edges()
    ev_rows: dict[str, list[dict[str, object]]] = {k: [] for k in edge_rows}
    mutation_rows: list[dict[str, object]] = []
    variant_positions: list[dict[str, object]] = []
    counts: dict[str, int] = {
        "input_variants": len(rows),
        "canonical_mutation_matches": 0,
        "exploded_transcript_consequence_rows": 0,
        "rows_after_allowed_consequence_filter": 0,
        "rows_after_canonical_transcript_filter": 0,
        "excluded_upstream_downstream_intergenic_or_regulatory_rows": 0,
        "rejected_non_allowlisted_consequence_rows": 0,
        "gene_endpoint_rejects": 0,
        "gene_interval_missing_or_unproven_rejects": 0,
        "gene_coordinate_containment_rejects": 0,
        "gene_coordinate_containment_passes": 0,
        "transcript_endpoint_rejects": 0,
        "supported_variants_for_enhancer_gate": len(supported_for_enhancer),
    }

    for row in rows:
        variant_id = row.get("variantId")
        if not isinstance(variant_id, str) or not variant_id:
            continue
        if canonical_mutations is not None and variant_id not in canonical_mutations:
            continue
        counts["canonical_mutation_matches"] = int(counts["canonical_mutation_matches"]) + 1
        chrom, pos = _variant_coord(row)
        mutation_rows.append({
            "id": variant_id,
            "hgvs": row.get("hgvsId"),
            "clinvar_id": _first_rs(row.get("rsIds")),
            "gnomad_id": variant_id,
            "name": _first_rs(row.get("rsIds")) or row.get("hgvsId") or variant_id,
            "source": "OpenTargets/variant",
        })
        if chrom and pos:
            variant_positions.append({"variant_id": variant_id, "chromosome": chrom, "position": pos, "ref": row.get("referenceAllele"), "alt": row.get("alternateAllele")})
        for cons_idx, consequence in enumerate(_scalar_list(row.get("transcriptConsequences"))):
            if not isinstance(consequence, dict):
                continue
            counts["exploded_transcript_consequence_rows"] = int(counts["exploded_transcript_consequence_rows"]) + 1
            cons_ids = _consequence_ids(consequence)
            if set(cons_ids) & EXCLUDED_CONSEQUENCE_IDS:
                counts["excluded_upstream_downstream_intergenic_or_regulatory_rows"] = int(counts["excluded_upstream_downstream_intergenic_or_regulatory_rows"]) + 1
            if not _is_allowed_consequence(cons_ids):
                counts["rejected_non_allowlisted_consequence_rows"] = int(counts["rejected_non_allowlisted_consequence_rows"]) + 1
                continue
            counts["rows_after_allowed_consequence_filter"] = int(counts["rows_after_allowed_consequence_filter"]) + 1
            gene_id = consequence.get("targetId")
            transcript_id = consequence.get("transcriptId")
            base_text = {
                "variant_id": variant_id,
                "chromosome": chrom,
                "position": pos,
                "reference_allele": row.get("referenceAllele"),
                "alternate_allele": row.get("alternateAllele"),
                "hgvs_id": row.get("hgvsId"),
                "consequence_ids": cons_ids,
                "impact": consequence.get("impact"),
                "biotype": consequence.get("biotype"),
                "is_ensembl_canonical": consequence.get("isEnsemblCanonical"),
                "distance_from_tss": consequence.get("distanceFromTss"),
                "distance_from_footprint": consequence.get("distanceFromFootprint"),
                "approved_symbol": consequence.get("approvedSymbol"),
                "policy": "t_60b3e504 allowed transcript/gene-local consequence filter",
            }
            if isinstance(gene_id, str) and gene_id.startswith("ENSG"):
                if canonical_genes is not None and gene_id not in canonical_genes:
                    counts["gene_endpoint_rejects"] = int(counts["gene_endpoint_rejects"]) + 1
                else:
                    contained, containment_proof, containment_reason = _gene_interval_contains(gene_intervals, gene_id, chrom, pos)
                    if not contained:
                        if containment_reason in {"no_gene_interval_table", "missing_variant_coordinate", "missing_gene_interval"}:
                            counts["gene_interval_missing_or_unproven_rejects"] = int(counts["gene_interval_missing_or_unproven_rejects"]) + 1
                        else:
                            counts["gene_coordinate_containment_rejects"] = int(counts["gene_coordinate_containment_rejects"]) + 1
                        continue
                    counts["gene_coordinate_containment_passes"] = int(counts["gene_coordinate_containment_passes"]) + 1
                    e = _edge(variant_id, NodeType.MUTATION.value, gene_id, NodeType.GENE.value, "mutation_in_gene", "in gene", "OpenTargets/variant transcriptConsequences policy-filtered target-contained")
                    edge_rows["mutation_in_gene"].append(e)
                    ev_rows["mutation_in_gene"].append(_evidence(e, "variant+target", f"variant:{variant_id}:tc:{cons_idx}:gene:{gene_id}", "policy_filtered_variant_transcript_consequence_target_gene_with_target_genomic_location_containment", text={**base_text, "containment_proof": containment_proof}, score=consequence.get("consequenceScore"), method="policy_bounded_source_native_builder_with_target_genomic_location_containment"))
            if not _is_canonical_transcript_consequence(consequence):
                continue
            counts["rows_after_canonical_transcript_filter"] = int(counts["rows_after_canonical_transcript_filter"]) + 1
            if cons_ids and isinstance(transcript_id, str) and transcript_id.startswith("ENST"):
                if canonical_transcripts is not None and transcript_id not in canonical_transcripts:
                    counts["transcript_endpoint_rejects"] = int(counts["transcript_endpoint_rejects"]) + 1
                else:
                    e = _edge(variant_id, NodeType.MUTATION.value, transcript_id, NodeType.TRANSCRIPT.value, "mutation_affects_transcript", "affects transcript", "OpenTargets/variant transcriptConsequences policy-filtered canonical-transcript")
                    edge_rows["mutation_affects_transcript"].append(e)
                    ev_rows["mutation_affects_transcript"].append(_evidence(e, "variant", f"variant:{variant_id}:tc:{cons_idx}:transcript:{transcript_id}", "policy_filtered_canonical_variant_transcript_consequence", text={**base_text, "transcript_id": transcript_id}, score=consequence.get("consequenceScore")))

    if enhancer_nodes is not None and not enhancer_nodes.empty and variant_positions:
        vp = pd.DataFrame(variant_positions)
        vp = vp[vp["variant_id"].isin(supported_for_enhancer)]
        if not vp.empty:
            enh = enhancer_nodes.copy()
            enh["chromosome"] = enh["chromosome"].astype(str).str.removeprefix("chr")
            enh["start"] = pd.to_numeric(enh["start"], errors="coerce")
            enh["end"] = pd.to_numeric(enh["end"], errors="coerce")
            for chrom, variants_chr in vp.groupby("chromosome"):
                enh_chr = enh[enh["chromosome"] == str(chrom)]
                if enh_chr.empty:
                    continue
                for v in variants_chr.itertuples(index=False):
                    overlaps = enh_chr[(enh_chr["start"] <= int(v.position)) & (enh_chr["end"] >= int(v.position))]
                    for h in overlaps.itertuples(index=False):
                        e = _edge(v.variant_id, NodeType.MUTATION.value, str(h.id), NodeType.ENHANCER.value, "mutation_overlaps_enhancer", "overlaps enhancer", "OpenTargets/variant + current enhancer intervals downstream-gated")
                        edge_rows["mutation_overlaps_enhancer"].append(e)
                        text = {"variant_id": v.variant_id, "chromosome": chrom, "position": int(v.position), "reference_allele": v.ref, "alternate_allele": v.alt, "enhancer_id": str(h.id), "enhancer_start": int(h.start), "enhancer_end": int(h.end), "downstream_association_gate": True, "support_relations": list(ASSOCIATION_RELATIONS)}
                        ev_rows["mutation_overlaps_enhancer"].append(_evidence(e, "variant+enhancer_interval", f"variant_enhancer_overlap:{v.variant_id}:{h.id}", "bounded_variant_overlaps_enhancer_interval", text=text, method="coordinate_overlap_after_downstream_association_gate"))

    edges = {rel: pd.DataFrame(rows).drop_duplicates(subset=["x_id", "y_id", "relation", "source"]) if rows else pd.DataFrame(columns=EDGE_COLUMNS) for rel, rows in edge_rows.items()}
    evidence = {rel: pd.DataFrame(rows) if rows else pd.DataFrame(columns=EVIDENCE_COLUMNS) for rel, rows in ev_rows.items()}
    nodes = pd.DataFrame(mutation_rows).drop_duplicates(subset=["id"]) if mutation_rows else pd.DataFrame(columns=["id", "hgvs", "clinvar_id", "gnomad_id", "name", "source"])
    summary = {
        **counts,
        "mutation_nodes": len(nodes),
        "edge_rows": {rel: len(df) for rel, df in edges.items()},
        "evidence_rows": {rel: len(df) for rel, df in evidence.items()},
    }
    return edges, evidence, nodes, summary


def _read_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return set(pq.read_table(path, columns=["id"]).column("id").to_pylist())


def _endpoint_missing_ids(edges: pd.DataFrame, node_path: Path, id_column: str) -> list[str]:
    if edges.empty:
        return []
    if not node_path.exists():
        return sorted(set(edges[id_column].astype(str)))
    try:
        import duckdb  # type: ignore[import-not-found]

        con = duckdb.connect(database=":memory:")
        vals = pd.DataFrame({"id": sorted(set(edges[id_column].astype(str)))})
        con.register("edge_ids", vals)
        return [str(row[0]) for row in con.execute(
            """
            SELECT edge_ids.id
            FROM edge_ids
            ANTI JOIN read_parquet(?) nodes
              ON edge_ids.id = CAST(nodes.id AS VARCHAR)
            ORDER BY edge_ids.id
            """,
            [str(node_path)],
        ).fetchall()]
    except Exception:
        node_ids = _read_ids(node_path)
        return sorted(set(edges[id_column].astype(str)) - node_ids)


def load_supported_variants(kg_cache_root: Path, candidate_variants: set[str]) -> set[str]:
    supported: set[str] = set()
    for rel in ASSOCIATION_RELATIONS:
        p = kg_cache_root / "edges" / f"{rel}.parquet"
        if not p.exists():
            continue
        table = pq.read_table(p, columns=["x_id"])
        ids = set(table.column("x_id").to_pylist())
        supported |= (ids & candidate_variants)
    return supported


def load_enhancer_slice(enhancer_path: Path, positions: pd.DataFrame, padding: int = 0) -> pd.DataFrame:
    if not enhancer_path.exists() or positions.empty:
        return pd.DataFrame(columns=["id", "chromosome", "start", "end"])
    cols = [c for c in ["id", "chromosome", "start", "end"] if c in pq.ParquetFile(enhancer_path).schema_arrow.names]
    pos = positions[["variant_id", "chromosome", "position"]].dropna().copy()
    if pos.empty:
        return pd.DataFrame(columns=cols)
    pos["chromosome"] = pos["chromosome"].astype(str).str.removeprefix("chr")
    pos["position"] = pd.to_numeric(pos["position"], errors="coerce")
    pos = pos.dropna(subset=["position"])
    pos["position"] = pos["position"].astype("int64")
    try:
        import duckdb  # type: ignore[import-not-found]

        con = duckdb.connect(database=":memory:")
        con.register("positions", pos)
        query = f"""
            SELECT DISTINCT enh.id, enh.chromosome, enh.start, enh."end"
            FROM read_parquet(?) AS enh
            JOIN positions AS pos
              ON regexp_replace(CAST(enh.chromosome AS VARCHAR), '^chr', '') = CAST(pos.chromosome AS VARCHAR)
             AND CAST(enh.start AS BIGINT) <= pos.position + ?
             AND CAST(enh."end" AS BIGINT) >= pos.position - ?
        """
        return con.execute(query, [str(enhancer_path), padding, padding]).df()
    except Exception:
        pieces = []
        full = pq.read_table(enhancer_path, columns=cols).to_pandas()
        full["chromosome"] = full["chromosome"].astype(str).str.removeprefix("chr")
        full["start"] = pd.to_numeric(full["start"], errors="coerce")
        full["end"] = pd.to_numeric(full["end"], errors="coerce")
        for chrom, grp in pos.groupby("chromosome"):
            lo = int(grp["position"].min()) - padding
            hi = int(grp["position"].max()) + padding
            pieces.append(full[(full["chromosome"] == str(chrom)) & (full["start"] <= hi) & (full["end"] >= lo)])
        return pd.concat(pieces, ignore_index=True).drop_duplicates(subset=["id"]) if pieces else pd.DataFrame(columns=cols)


def load_gene_intervals(target_files: list[str], cache_dir: Path) -> pd.DataFrame:
    """Load OpenTargets target.genomicLocation intervals keyed by Ensembl gene id."""
    rows: list[dict[str, object]] = []
    for source in target_files:
        local_path = materialize_variant_file(source, cache_dir)
        table = pq.read_table(local_path, columns=["id", "approvedSymbol", "genomicLocation"])
        for row in table.to_pylist():
            gene_id = row.get("id")
            loc = row.get("genomicLocation")
            if not isinstance(gene_id, str) or not gene_id.startswith("ENSG") or not isinstance(loc, dict):
                continue
            chrom = _normalize_chromosome(loc.get("chromosome"))
            start = loc.get("start")
            end = loc.get("end")
            if chrom is None or start is None or end is None:
                continue
            rows.append({
                "id": gene_id,
                "approved_symbol": row.get("approvedSymbol"),
                "chromosome": chrom,
                "start": int(start),
                "end": int(end),
                "strand": loc.get("strand"),
                "source": "OpenTargets/target.genomicLocation",
            })
    if not rows:
        return pd.DataFrame(columns=["id", "approved_symbol", "chromosome", "start", "end", "strand", "source"])
    return pd.DataFrame(rows).drop_duplicates(subset=["id", "chromosome", "start", "end"]).reset_index(drop=True)


def write_outputs(stage_root: Path, edges: dict[str, pd.DataFrame], evidence: dict[str, pd.DataFrame], nodes: pd.DataFrame, summary: dict[str, object]) -> None:
    (stage_root / "edges").mkdir(parents=True, exist_ok=True)
    (stage_root / "evidence").mkdir(parents=True, exist_ok=True)
    (stage_root / "nodes").mkdir(parents=True, exist_ok=True)
    if not nodes.empty:
        pq.write_table(pa.Table.from_pandas(nodes, preserve_index=False), stage_root / "nodes" / "mutation.parquet")
    for rel, df in edges.items():
        df = df.reindex(columns=EDGE_COLUMNS)
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), stage_root / "edges" / f"{rel}.parquet")
        ev = evidence[rel]
        ev = kg_evidence._coerce_evidence_frame(ev, rel) if not ev.empty else ev.reindex(columns=EVIDENCE_COLUMNS)
        pq.write_table(pa.Table.from_pandas(ev, preserve_index=False), stage_root / "evidence" / f"{rel}.parquet")
    (stage_root / "manifest.json").write_text(json.dumps({"created_at": _now(), **summary}, indent=2, sort_keys=True))


def validate_outputs(stage_root: Path, kg_cache_root: Path) -> dict[str, object]:
    out: dict[str, object] = {"relations": {}, "passed": True}
    target_nodes = {
        "mutation_in_gene": kg_cache_root / "nodes" / "gene.parquet",
        "mutation_affects_transcript": kg_cache_root / "nodes" / "transcript.parquet",
        "mutation_overlaps_enhancer": kg_cache_root / "nodes" / "enhancer.parquet",
    }
    mutation_node_path = kg_cache_root / "nodes" / "mutation.parquet"
    for rel in ("mutation_in_gene", "mutation_affects_transcript", "mutation_overlaps_enhancer"):
        ep = stage_root / "edges" / f"{rel}.parquet"
        vp = stage_root / "evidence" / f"{rel}.parquet"
        edges = pq.read_table(ep).to_pandas() if ep.exists() else pd.DataFrame(columns=EDGE_COLUMNS)
        ev = pq.read_table(vp).to_pandas() if vp.exists() else pd.DataFrame(columns=EVIDENCE_COLUMNS)
        edge_keys = set((edges["relation"] + "|" + edges["x_id"] + "|" + edges["y_id"]).astype(str)) if not edges.empty else set()
        ev_keys = set(ev["edge_key"].astype(str)) if not ev.empty and "edge_key" in ev.columns else set()
        missing_x_all = _endpoint_missing_ids(edges, mutation_node_path, "x_id")
        missing_y_all = _endpoint_missing_ids(edges, target_nodes[rel], "y_id")
        info = {
            "edges": len(edges),
            "evidence": len(ev),
            "edges_without_evidence": len(edge_keys - ev_keys),
            "evidence_without_edge": len(ev_keys - edge_keys),
            "missing_x_count": len(missing_x_all),
            "missing_y_count": len(missing_y_all),
            "missing_x_examples": missing_x_all[:10],
            "missing_y_examples": missing_y_all[:10],
        }
        info["passed"] = info["missing_x_count"] == 0 and info["missing_y_count"] == 0 and info["edges_without_evidence"] == 0 and info["evidence_without_edge"] == 0
        out["relations"][rel] = info
        out["passed"] = out["passed"] and info["passed"]
    (stage_root / "validation.json").write_text(json.dumps(out, indent=2, sort_keys=True))
    return out


def _merge_frames(parts: list[pd.DataFrame], columns: list[str]) -> pd.DataFrame:
    if not parts:
        return pd.DataFrame(columns=columns)
    return pd.concat(parts, ignore_index=True).drop_duplicates().reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant-file", action="append", required=True, help="Local Parquet path or HTTP(S) URL. Repeat for multiple OpenTargets variant parts.")
    ap.add_argument("--kg-cache-root", type=Path, default=Path("/Users/jkobject/mnt/gcs/jouvencekb/main"))
    ap.add_argument("--stage-root", type=Path, required=True)
    ap.add_argument("--max-variants", type=int, default=None)
    ap.add_argument("--skip-enhancer-overlap", action="store_true", help="Skip enhancer overlap when running a fast gene/transcript-only policy smoke build.")
    ap.add_argument("--download-cache", type=Path, default=Path("artifacts/cache/opentargets-26.03/variant"))
    ap.add_argument("--target-file", action="append", default=None, help="OpenTargets target Parquet path or URL containing target.genomicLocation. Defaults to all official target parts for the configured release.")
    ap.add_argument("--target-download-cache", type=Path, default=Path("artifacts/cache/opentargets-26.03/target"))
    args = ap.parse_args(argv)

    args.stage_root.mkdir(parents=True, exist_ok=True)
    target_files = args.target_file or _list_ftp_parquets("target")
    write_source_audit(args.stage_root, args.variant_file, target_files, args.kg_cache_root)
    gene_intervals = load_gene_intervals(target_files, args.target_download_cache)
    canonical_mutations = _read_ids(args.kg_cache_root / "nodes" / "mutation.parquet")
    canonical_genes = _read_ids(args.kg_cache_root / "nodes" / "gene.parquet")
    canonical_transcripts = _read_ids(args.kg_cache_root / "nodes" / "transcript.parquet")

    all_edges: dict[str, list[pd.DataFrame]] = {rel: [] for rel in _empty_edges()}
    all_evidence: dict[str, list[pd.DataFrame]] = {rel: [] for rel in _empty_edges()}
    all_nodes: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    remaining = args.max_variants

    with tempfile.TemporaryDirectory(prefix="mutation-genomic-build-") as tmp:
        cache_dir = args.download_cache if args.download_cache else Path(tmp)
        for source in args.variant_file:
            if remaining is not None and remaining <= 0:
                break
            local_path = materialize_variant_file(source, cache_dir)
            columns = ["variantId", "chromosome", "position", "referenceAllele", "alternateAllele", "hgvsId", "rsIds", "transcriptConsequences"]
            table = pq.read_table(local_path, columns=columns)
            if remaining is not None:
                table = table.slice(0, remaining)
                remaining -= table.num_rows
            candidate_ids = set(table.column("variantId").to_pylist()) & canonical_mutations
            supported = load_supported_variants(args.kg_cache_root, candidate_ids)
            if args.skip_enhancer_overlap or not supported:
                enhancer_slice = pd.DataFrame(columns=["id", "chromosome", "start", "end"])
            else:
                pos = pd.DataFrame([{"variant_id": r.get("variantId"), "chromosome": _variant_coord(r)[0], "position": _variant_coord(r)[1]} for r in table.to_pylist() if r.get("variantId") in supported])
                enhancer_slice = load_enhancer_slice(args.kg_cache_root / "nodes" / "enhancer.parquet", pos.dropna()) if not pos.empty else pd.DataFrame(columns=["id", "chromosome", "start", "end"])
            edges, evidence, nodes, summary = build_edges_from_variants(
                table,
                supported,
                enhancer_slice,
                gene_intervals,
                max_variants=None,
                canonical_mutations=canonical_mutations,
                canonical_genes=canonical_genes,
                canonical_transcripts=canonical_transcripts,
            )
            summary["variant_file"] = str(source)
            summary["local_variant_file"] = str(local_path)
            summary["enhancer_slice_rows"] = len(enhancer_slice)
            summary["gene_interval_rows"] = len(gene_intervals)
            summaries.append(summary)
            for rel in all_edges:
                all_edges[rel].append(edges[rel])
                all_evidence[rel].append(evidence[rel])
            all_nodes.append(nodes)

    merged_edges = {rel: _merge_frames(parts, EDGE_COLUMNS) for rel, parts in all_edges.items()}
    merged_evidence = {rel: _merge_frames(parts, EVIDENCE_COLUMNS) for rel, parts in all_evidence.items()}
    merged_nodes = _merge_frames(all_nodes, ["id", "hgvs", "clinvar_id", "gnomad_id", "name", "source"])
    summary = {
        "policy": "t_60b3e504 bounded rebuild; no pilot-as-is promotion; staged only",
        "variant_file_count": len(summaries),
        "per_file": summaries,
        "totals": {
            "input_variants": sum(int(s.get("input_variants", 0)) for s in summaries),
            "canonical_mutation_matches": sum(int(s.get("canonical_mutation_matches", 0)) for s in summaries),
            "exploded_transcript_consequence_rows": sum(int(s.get("exploded_transcript_consequence_rows", 0)) for s in summaries),
            "rows_after_allowed_consequence_filter": sum(int(s.get("rows_after_allowed_consequence_filter", 0)) for s in summaries),
            "rows_after_canonical_transcript_filter": sum(int(s.get("rows_after_canonical_transcript_filter", 0)) for s in summaries),
            "excluded_upstream_downstream_intergenic_or_regulatory_rows": sum(int(s.get("excluded_upstream_downstream_intergenic_or_regulatory_rows", 0)) for s in summaries),
            "rejected_non_allowlisted_consequence_rows": sum(int(s.get("rejected_non_allowlisted_consequence_rows", 0)) for s in summaries),
            "gene_endpoint_rejects": sum(int(s.get("gene_endpoint_rejects", 0)) for s in summaries),
            "gene_interval_missing_or_unproven_rejects": sum(int(s.get("gene_interval_missing_or_unproven_rejects", 0)) for s in summaries),
            "gene_coordinate_containment_rejects": sum(int(s.get("gene_coordinate_containment_rejects", 0)) for s in summaries),
            "gene_coordinate_containment_passes": sum(int(s.get("gene_coordinate_containment_passes", 0)) for s in summaries),
            "transcript_endpoint_rejects": sum(int(s.get("transcript_endpoint_rejects", 0)) for s in summaries),
            "enhancer_slice_rows": sum(int(s.get("enhancer_slice_rows", 0)) for s in summaries),
            "gene_interval_rows": len(gene_intervals),
            "mutation_nodes": len(merged_nodes),
            "edge_rows": {rel: len(df) for rel, df in merged_edges.items()},
            "evidence_rows": {rel: len(df) for rel, df in merged_evidence.items()},
        },
    }
    write_outputs(args.stage_root, merged_edges, merged_evidence, merged_nodes, summary)
    validation = validate_outputs(args.stage_root, args.kg_cache_root)
    print(json.dumps({"stage_root": str(args.stage_root), "summary": summary["totals"], "validation": validation}, indent=2, sort_keys=True))
    return 0 if validation["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
