"""Build staged Cell Ontology cell-type context relations.

This staged pilot materializes only source-native, reviewable assertions:

- ``cell_type_subtype_of_cell_type`` from Cell Ontology ``is_a`` edges.
- ``cell_type_found_in_tissue`` from explicit CL relationships to UBERON
  anatomical entities (currently ``part_of`` / ``located_in`` only).
- ``cell_type_involved_in_disease`` is intentionally not synthesized from CL
  disease-like labels or RNA-expression context.  A future run must provide an
  explicit disease-cell enrichment/annotation source.

Outputs mirror KG edge/evidence layout and add staged-only metadata columns to
retain source release, predicate, mapping confidence/context, and source record
IDs for review before promotion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

try:  # package execution
    from .kg_schema import EDGE_PARQUET_COLUMNS
except ImportError:  # pragma: no cover - script execution fallback
    from kg_schema import EDGE_PARQUET_COLUMNS  # type: ignore[no-redef]

log = logging.getLogger(__name__)

CL_OBO_URL = "https://purl.obolibrary.org/obo/cl.obo"
REL_SUBTYPE = "cell_type_subtype_of_cell_type"
REL_TISSUE = "cell_type_found_in_tissue"
REL_DISEASE = "cell_type_involved_in_disease"
EDGE_COLUMNS = [name for name, _ in EDGE_PARQUET_COLUMNS]
BASE_EVIDENCE_COLUMNS = [
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
    "effect_size",
    "p_value",
    "direction",
    "confidence_interval",
    "predicate",
    "text_span",
    "section",
    "extraction_method",
    "license",
    "release",
    "created_at",
]
STAGED_EVIDENCE_COLUMNS = BASE_EVIDENCE_COLUMNS + [
    "source_term_id",
    "source_term_name",
    "target_term_id",
    "target_term_name",
    "mapping_confidence",
    "context",
    "source_record_json",
]
_ACCEPTED_TISSUE_RELATIONSHIPS = {
    "part_of",
    "BFO:0000050",
    "BFO:0000050 ! part of",
    "RO:0001025",
    "RO:0001025 ! located in",
    "located_in",
}
_RELATIONSHIP_LABELS = {
    "part_of": "part_of",
    "BFO:0000050": "part_of",
    "RO:0001025": "located_in",
    "located_in": "located_in",
}
_CURIE_RE = re.compile(r"\b([A-Z][A-Za-z0-9_]*:\d{7,})\b")


@dataclass
class Term:
    id: str = ""
    name: str = ""
    is_obsolete: bool = False
    is_a: list[str] = field(default_factory=list)
    relationships: list[tuple[str, str, str]] = field(default_factory=list)
    raw_lines: list[str] = field(default_factory=list)


def _clean_curie(value: str) -> str:
    match = _CURIE_RE.search(value.replace("_", ":", 1) if value.startswith(("CL_", "UBERON_", "EFO_", "MONDO_")) else value)
    return match.group(1) if match else ""


def _stable_id(parts: dict[str, Any]) -> str:
    payload = json.dumps(parts, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


def _download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    req = urllib.request.Request(url, headers={"User-Agent": "hermes-agent"})
    with urllib.request.urlopen(req, timeout=120) as response:
        data = response.read()
    dest.write_bytes(data)
    return dest


def _parse_obo(path: Path) -> tuple[dict[str, Term], dict[str, Any]]:
    terms: dict[str, Term] = {}
    header: list[str] = []
    current: Term | None = None
    in_term = False

    def finish() -> None:
        nonlocal current
        if current and current.id:
            terms[current.id] = current
        current = None

    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if line == "[Term]":
            finish()
            current = Term()
            in_term = True
            continue
        if line.startswith("[") and line != "[Term]":
            finish()
            in_term = False
            continue
        if not in_term:
            if line:
                header.append(line)
            continue
        if current is None:
            continue
        current.raw_lines.append(line)
        if line.startswith("id: "):
            current.id = line[4:].strip()
        elif line.startswith("name: "):
            current.name = line[6:].strip()
        elif line.startswith("is_obsolete: true"):
            current.is_obsolete = True
        elif line.startswith("is_a: "):
            parent = _clean_curie(line[6:])
            if parent:
                current.is_a.append(parent)
        elif line.startswith("relationship: "):
            payload = line[len("relationship: ") :]
            parts = payload.split()
            if len(parts) >= 2:
                predicate = parts[0]
                target = _clean_curie(" ".join(parts[1:]))
                if target:
                    current.relationships.append((predicate, target, line))
    finish()

    metadata: dict[str, Any] = {"header_lines": header[:200]}
    for item in header:
        if item.startswith("data-version: "):
            metadata["data_version"] = item[len("data-version: ") :].strip()
        elif item.startswith("date: "):
            metadata["date"] = item[len("date: ") :].strip()
        elif item.startswith("ontology: "):
            metadata["ontology"] = item[len("ontology: ") :].strip()
    return terms, metadata


def _load_ids(path: str, id_column: str = "id") -> set[str]:
    table = pq.read_table(path, columns=[id_column])
    return {str(v) for v in table.column(id_column).to_pylist() if v is not None}


def _load_labels(path: str, id_column: str = "id", label_column: str = "name") -> dict[str, str]:
    schema_names = set(pq.ParquetFile(path).schema_arrow.names)
    if label_column not in schema_names:
        return {}
    table = pq.read_table(path, columns=[id_column, label_column])
    labels: dict[str, str] = {}
    for item in table.to_pylist():
        term_id = item.get(id_column)
        label = item.get(label_column)
        if term_id is not None and label:
            labels[str(term_id)] = str(label)
    return labels


def _term_name(terms: dict[str, Term], term_id: str) -> str:
    term = terms.get(term_id)
    return term.name if term else ""


def _edge_row(x_id: str, x_type: str, y_id: str, y_type: str, relation: str, display: str, source: str) -> dict[str, Any]:
    return {
        "x_id": x_id,
        "x_type": x_type,
        "y_id": y_id,
        "y_type": y_type,
        "relation": relation,
        "display_relation": display,
        "source": source,
        "credibility": 3,
    }


def _evidence_row(
    *,
    relation: str,
    x_id: str,
    x_type: str,
    y_id: str,
    y_type: str,
    predicate: str,
    source_record_id: str,
    release: str,
    created_at: str,
    source_term_name: str,
    target_term_name: str,
    context: str,
    source_record_json: dict[str, Any],
) -> dict[str, Any]:
    return {
        "edge_key": f"{relation}|{x_id}|{y_id}",
        "relation": relation,
        "x_id": x_id,
        "x_type": x_type,
        "y_id": y_id,
        "y_type": y_type,
        "evidence_type": "ontology_axiom",
        "source": "Cell Ontology",
        "source_dataset": "cl.obo",
        "source_record_id": source_record_id,
        "paper_id": "",
        "dataset_id": f"CellOntology:{release}",
        "study_id": "",
        "evidence_score": None,
        "effect_size": None,
        "p_value": None,
        "direction": "source_to_target",
        "confidence_interval": "",
        "predicate": predicate,
        "text_span": "",
        "section": "[Term]",
        "extraction_method": "OBO is_a/relationship parser",
        "license": "Cell Ontology license / OBO Foundry terms",
        "release": release,
        "created_at": created_at,
        "source_term_id": x_id,
        "source_term_name": source_term_name,
        "target_term_id": y_id,
        "target_term_name": target_term_name,
        "mapping_confidence": "exact_ontology_curie_endpoint",
        "context": context,
        "source_record_json": json.dumps(source_record_json, sort_keys=True),
    }


def build(
    *,
    cl_obo_path: Path,
    canonical_cell_type_path: str,
    canonical_tissue_path: str,
    canonical_disease_path: str,
    output_dir: Path,
) -> dict[str, Any]:
    created_at = datetime.now(timezone.utc).isoformat()
    terms, cl_metadata = _parse_obo(cl_obo_path)
    release = str(cl_metadata.get("data_version") or cl_metadata.get("date") or "unknown")
    cell_ids = _load_ids(canonical_cell_type_path)
    tissue_ids = _load_ids(canonical_tissue_path)
    disease_ids = _load_ids(canonical_disease_path)
    tissue_labels = _load_labels(canonical_tissue_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    for subdir in ("edges", "evidence", "reports"):
        (output_dir / subdir).mkdir(exist_ok=True)

    active_cl_terms = {tid: t for tid, t in terms.items() if tid.startswith("CL:") and not t.is_obsolete}
    source_terms_in_canonical = set(active_cl_terms) & cell_ids

    subtype_edges: dict[tuple[str, str], dict[str, Any]] = {}
    subtype_evidence: list[dict[str, Any]] = []
    tissue_edges: dict[tuple[str, str], dict[str, Any]] = {}
    tissue_evidence: list[dict[str, Any]] = []
    rejected = {
        "subtype_noncanonical_child": 0,
        "subtype_noncanonical_parent": 0,
        "subtype_obsolete_term": 0,
        "tissue_noncanonical_cell_type": 0,
        "tissue_non_uberon_target": 0,
        "tissue_unaccepted_predicate": 0,
        "tissue_noncanonical_tissue": 0,
        "disease_edges_not_built_source_gap": 0,
    }
    tissue_predicate_counts: dict[str, int] = {}
    all_uberon_relationship_predicate_counts: dict[str, int] = {}

    for term_id, term in active_cl_terms.items():
        for parent_id in term.is_a:
            if term_id not in cell_ids:
                rejected["subtype_noncanonical_child"] += 1
                continue
            parent = terms.get(parent_id)
            if parent and parent.is_obsolete:
                rejected["subtype_obsolete_term"] += 1
                continue
            if parent_id not in cell_ids:
                rejected["subtype_noncanonical_parent"] += 1
                continue
            key = (term_id, parent_id)
            subtype_edges.setdefault(
                key,
                _edge_row(term_id, "cell_type", parent_id, "cell_type", REL_SUBTYPE, "subtype of", "Cell Ontology"),
            )
            source_record = {"term_id": term_id, "parent_id": parent_id, "axiom": f"is_a: {parent_id}"}
            subtype_evidence.append(
                _evidence_row(
                    relation=REL_SUBTYPE,
                    x_id=term_id,
                    x_type="cell_type",
                    y_id=parent_id,
                    y_type="cell_type",
                    predicate="is_a",
                    source_record_id="CL:is_a:" + _stable_id(source_record),
                    release=release,
                    created_at=created_at,
                    source_term_name=term.name,
                    target_term_name=_term_name(terms, parent_id),
                    context="Cell Ontology asserted is_a hierarchy",
                    source_record_json=source_record,
                )
            )

        for predicate, target_id, raw_line in term.relationships:
            if target_id.startswith("UBERON:"):
                all_uberon_relationship_predicate_counts[predicate] = all_uberon_relationship_predicate_counts.get(predicate, 0) + 1
            if not target_id.startswith("UBERON:"):
                rejected["tissue_non_uberon_target"] += 1
                continue
            if predicate not in _ACCEPTED_TISSUE_RELATIONSHIPS:
                rejected["tissue_unaccepted_predicate"] += 1
                continue
            if term_id not in cell_ids:
                rejected["tissue_noncanonical_cell_type"] += 1
                continue
            if target_id not in tissue_ids:
                rejected["tissue_noncanonical_tissue"] += 1
                continue
            normalized_predicate = _RELATIONSHIP_LABELS.get(predicate, predicate)
            tissue_predicate_counts[normalized_predicate] = tissue_predicate_counts.get(normalized_predicate, 0) + 1
            key = (term_id, target_id)
            tissue_edges.setdefault(
                key,
                _edge_row(term_id, "cell_type", target_id, "tissue", REL_TISSUE, "found in tissue", "Cell Ontology/UBERON"),
            )
            source_record = {"term_id": term_id, "target_id": target_id, "predicate": predicate, "axiom": raw_line}
            tissue_evidence.append(
                _evidence_row(
                    relation=REL_TISSUE,
                    x_id=term_id,
                    x_type="cell_type",
                    y_id=target_id,
                    y_type="tissue",
                    predicate=normalized_predicate,
                    source_record_id="CL:relationship:" + _stable_id(source_record),
                    release=release,
                    created_at=created_at,
                    source_term_name=term.name,
                    target_term_name=tissue_labels.get(target_id, target_id),
                    context="Explicit CL relationship to UBERON anatomical entity",
                    source_record_json=source_record,
                )
            )

    rejected["disease_edges_not_built_source_gap"] = len(source_terms_in_canonical)

    artifacts: dict[str, str] = {}
    relation_counts: dict[str, dict[str, int]] = {}
    for relation, edge_records, evidence_rows in [
        (REL_SUBTYPE, subtype_edges, subtype_evidence),
        (REL_TISSUE, tissue_edges, tissue_evidence),
    ]:
        edges = pd.DataFrame(edge_records.values(), columns=EDGE_COLUMNS)
        evidence = pd.DataFrame(evidence_rows, columns=STAGED_EVIDENCE_COLUMNS)
        if not evidence.empty:
            evidence = evidence.drop_duplicates(
                subset=["relation", "x_id", "y_id", "source", "source_dataset", "source_record_id", "predicate"],
                keep="first",
            ).reset_index(drop=True)
        edge_path = output_dir / "edges" / f"{relation}.parquet"
        evidence_path = output_dir / "evidence" / f"{relation}.parquet"
        edges.to_parquet(edge_path, index=False)
        evidence.to_parquet(evidence_path, index=False)
        artifacts[f"edges/{relation}"] = str(edge_path)
        artifacts[f"evidence/{relation}"] = str(evidence_path)
        edge_keys = set(relation + "|" + edges["x_id"].astype(str) + "|" + edges["y_id"].astype(str)) if not edges.empty else set()
        relation_counts[relation] = {
            "edges": int(len(edges)),
            "evidence_rows": int(len(evidence)),
            "duplicate_edges": int(edges.duplicated(subset=["x_id", "y_id", "relation"]).sum()) if not edges.empty else 0,
            "evidence_without_edge": int((~evidence["edge_key"].isin(edge_keys)).sum()) if not evidence.empty else 0,
        }

    disease_gap_report = {
        "relation": REL_DISEASE,
        "status": "blocked_source_gap",
        "reason": "No explicit disease-cell enrichment/annotation source was provided or present in local KG cache; do not infer disease involvement from Cell Ontology labels, cell type names, or RNA-expression context.",
        "acceptable_future_sources": [
            "curated disease-cell enrichment table with CL/EFO-or-MONDO endpoints and source record IDs",
            "single-cell disease atlas annotations that explicitly link disease/context to enriched cell types with statistics",
        ],
        "canonical_disease_nodes_available": len(disease_ids),
    }
    disease_gap_path = output_dir / "reports" / f"{REL_DISEASE}_source_gap.json"
    disease_gap_path.write_text(json.dumps(disease_gap_report, indent=2, sort_keys=True) + "\n")
    artifacts["reports/disease_source_gap"] = str(disease_gap_path)

    subtype_missing = sorted((set(x for pair in subtype_edges for x in pair)) - cell_ids)
    tissue_missing_cell = sorted({x for x, _ in tissue_edges} - cell_ids)
    tissue_missing_tissue = sorted({y for _, y in tissue_edges} - tissue_ids)
    report: dict[str, Any] = {
        "title": "Cell Ontology cell_type context staged pilot",
        "created_at": created_at,
        "source_audit": {
            "cell_ontology": {
                "url": CL_OBO_URL,
                "local_path": str(cl_obo_path),
                "release": release,
                "terms_total": len(terms),
                "active_cl_terms": len(active_cl_terms),
                "active_cl_terms_in_canonical_nodes": len(source_terms_in_canonical),
                "accepted_for_subtype": "is_a axioms where child and parent are canonical CL cell_type nodes",
                "accepted_for_tissue": "explicit CL relationship to UBERON with predicate part_of or located_in and canonical endpoints",
            },
            "uberon_cl_bridge": {
                "source": "CL OBO relationship axioms that target UBERON CURIEs",
                "all_uberon_relationship_predicate_counts": dict(sorted(all_uberon_relationship_predicate_counts.items())),
                "accepted_predicate_counts": dict(sorted(tissue_predicate_counts.items())),
            },
            "hca_cellxgene_tissue_metadata": {
                "status": "not_used_in_this_pilot",
                "reason": "No local source-native HCA/CellxGene dataset metadata table with paired CL and UBERON endpoints was present in the KG cache; this pilot avoids deriving tissue from expression edges.",
            },
            "disease_cell_enrichment_resources": disease_gap_report,
        },
        "relations": relation_counts,
        "rejected_counts": rejected,
        "validation": {
            "subtype_endpoint_antijoin_pass": len(subtype_missing) == 0,
            "subtype_missing_endpoint_ids_sample": subtype_missing[:50],
            "tissue_cell_endpoint_antijoin_pass": len(tissue_missing_cell) == 0,
            "tissue_tissue_endpoint_antijoin_pass": len(tissue_missing_tissue) == 0,
            "tissue_missing_cell_ids_sample": tissue_missing_cell[:50],
            "tissue_missing_tissue_ids_sample": tissue_missing_tissue[:50],
            "evidence_support_pass": all(v["evidence_without_edge"] == 0 for v in relation_counts.values()),
            "duplicate_edge_pass": all(v["duplicate_edges"] == 0 for v in relation_counts.values()),
        },
        "artifacts": artifacts,
        "promotion_recommendation": "Review subtype and tissue staged edges for semantic scope before canonical promotion; keep disease relation blocked until an explicit disease-cell enrichment source is selected.",
    }
    report_path = output_dir / "reports" / "cell_type_context_relations_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    artifacts["reports/build_report"] = str(report_path)
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cl-obo-path", default="artifacts/cache/raw/cell_ontology/cl.obo")
    parser.add_argument("--download", action="store_true", help="Download cl.obo if missing")
    parser.add_argument("--canonical-cell-type-path", default="/Users/jkobject/mnt/gcs/jouvencekb/main/nodes/cell_type.parquet")
    parser.add_argument("--canonical-tissue-path", default="/Users/jkobject/mnt/gcs/jouvencekb/main/nodes/tissue.parquet")
    parser.add_argument("--canonical-disease-path", default="/Users/jkobject/mnt/gcs/jouvencekb/main/nodes/disease.parquet")
    parser.add_argument("--output-dir", default="artifacts/staged/cell-type-context-relations")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cl_obo_path = Path(args.cl_obo_path)
    if args.download:
        _download(CL_OBO_URL, cl_obo_path)
    if not cl_obo_path.exists():
        raise FileNotFoundError(f"Missing {cl_obo_path}; pass --download to fetch {CL_OBO_URL}")
    report = build(
        cl_obo_path=cl_obo_path,
        canonical_cell_type_path=args.canonical_cell_type_path,
        canonical_tissue_path=args.canonical_tissue_path,
        canonical_disease_path=args.canonical_disease_path,
        output_dir=Path(args.output_dir),
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
