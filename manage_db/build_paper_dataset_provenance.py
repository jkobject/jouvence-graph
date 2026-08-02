"""Stage paper/dataset provenance, citation, and dataset containment edges.

This builder is intentionally staging-only. It creates metadata/provenance edges
for review and never turns paper co-mentions or citations into biological
assertions. The default pilot uses source metadata from OpenAlex/Crossref/Europe
PMC plus locally cached OpenTargets 26.03 parquet tables when available.
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

SOURCE_RELEASE = "OpenTargets 26.03"
USER_AGENT = "hermes-txgnn-paper-dataset-provenance/0.1 (mailto:jkobject@gmail.com)"

EDGE_COLUMNS = [
    "x_id",
    "x_type",
    "y_id",
    "y_type",
    "relation",
    "display_relation",
    "source",
    "credibility",
]

EVIDENCE_COLUMNS = [
    "edge_key",
    "relation",
    "x_id",
    "x_type",
    "y_id",
    "y_type",
    "evidence_type",
    "source",
    "source_dataset",
    "source_release",
    "source_record_id",
    "paper_id",
    "doi",
    "pmid",
    "openalex_work_id",
    "dataset_id",
    "source_dataset_id",
    "citation_direction",
    "mapping_confidence",
    "release_date",
    "accessed_at",
    "source_url",
    "license",
    "raw_json",
]

DATASET_COLUMNS = [
    "id",
    "name",
    "source",
    "source_dataset_id",
    "source_release",
    "release_date",
    "license",
    "source_url",
]

PAPER_COLUMNS = [
    "id",
    "pmid",
    "doi",
    "pmc_id",
    "arxiv_id",
    "title",
    "publication_date",
    "openalex_work_id",
    "source",
    "source_record_id",
    "license",
]

RELATION_TARGET_TYPE = {
    "dataset_contains_disease": "disease",
    "dataset_contains_molecule": "molecule",
    "dataset_contains_cell_type": "cell_type",
    "dataset_contains_cell_line": "cell_line",
    "dataset_contains_tissue": "tissue",
}


@dataclass(frozen=True)
class BuildResult:
    papers: pd.DataFrame
    datasets: pd.DataFrame
    edges: dict[str, pd.DataFrame]
    evidence: dict[str, pd.DataFrame]
    source_audit: list[dict[str, Any]]
    validation: dict[str, Any]


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _clean(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))


def _http_json(url: str, timeout: int = 30) -> dict[str, Any]:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def audit_metadata_sources(accessed_at: str) -> list[dict[str, Any]]:
    """Probe public metadata endpoints and record license/access notes.

    Semantic Scholar is audited but not used by the default pilot because API-key
    and redistribution terms can vary. OpenAlex is used for citation metadata;
    Crossref and Europe PMC are kept as DOI/PMID verification endpoints.
    """

    endpoints = [
        {
            "source": "OpenAlex",
            "url": "https://api.openalex.org/works?per-page=1",
            "license": "OpenAlex metadata snapshot/API are CC0; observe polite-pool User-Agent.",
            "used_for": "paper metadata and paper_cites_paper citation direction",
        },
        {
            "source": "Europe PMC",
            "url": "https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=OPEN_TARGETS&format=json&pageSize=1",
            "license": "Europe PMC REST metadata is publicly accessible; record only bibliographic metadata/IDs, not full text.",
            "used_for": "PMID/DOI access audit and possible cross-checks",
        },
        {
            "source": "Crossref",
            "url": "https://api.crossref.org/works?rows=1",
            "license": "Crossref metadata API is public; individual work/license fields must be preserved when used.",
            "used_for": "DOI metadata/license access audit",
        },
        {
            "source": "Semantic Scholar",
            "url": "https://api.semanticscholar.org/graph/v1/paper/search?query=Open%20Targets&limit=1&fields=paperId,title",
            "license": "Audited only; not used in this pilot because API-key/rate/redistribution constraints vary by deployment.",
            "used_for": "not used",
        },
    ]
    out: list[dict[str, Any]] = []
    for endpoint in endpoints:
        row = {**endpoint, "accessed_at": accessed_at, "ok": False, "status": None, "error": ""}
        try:
            req = urllib.request.Request(endpoint["url"], headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=20) as response:
                row["status"] = int(response.status)
                row["ok"] = 200 <= response.status < 300
        except urllib.error.HTTPError as exc:
            row["status"] = int(exc.code)
            row["error"] = str(exc)
        except Exception as exc:  # pragma: no cover - network dependent
            row["error"] = repr(exc)
        out.append(row)
    return out


def normalize_doi(value: Any) -> str:
    text = _clean(value)
    if not text:
        return ""
    text = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", text, flags=re.I)
    text = text.removeprefix("doi:").strip()
    return f"DOI:{text.lower()}" if text else ""


def normalize_pmid(value: Any) -> str:
    text = _clean(value)
    if not text:
        return ""
    match = re.search(r"(\d{4,})", text)
    return f"PMID:{match.group(1)}" if match else ""


def openalex_compact_id(value: Any) -> str:
    text = _clean(value)
    if not text:
        return ""
    return text.rstrip("/").rsplit("/", 1)[-1]


def paper_id_from_work(work: dict[str, Any]) -> str:
    ids = work.get("ids") or {}
    return normalize_pmid(ids.get("pmid")) or normalize_doi(work.get("doi") or ids.get("doi")) or f"OpenAlex:{openalex_compact_id(work.get('id'))}"


def paper_node_from_work(work: dict[str, Any], source: str = "OpenAlex") -> dict[str, Any]:
    ids = work.get("ids") or {}
    pmid = normalize_pmid(ids.get("pmid"))
    doi = normalize_doi(work.get("doi") or ids.get("doi"))
    openalex_id = openalex_compact_id(work.get("id"))
    return {
        "id": paper_id_from_work(work),
        "pmid": pmid,
        "doi": doi,
        "pmc_id": "",
        "arxiv_id": "",
        "title": _clean(work.get("title") or work.get("display_name")),
        "publication_date": _clean(work.get("publication_date")),
        "openalex_work_id": openalex_id,
        "source": source,
        "source_record_id": openalex_id,
        "license": "OpenAlex CC0 metadata; article full text/license not imported",
    }


def fetch_openalex_work(identifier: str) -> dict[str, Any]:
    ident = identifier.strip()
    if ident.startswith("http") or ident.startswith("W"):
        url = f"https://api.openalex.org/works/{openalex_compact_id(ident)}"
    elif ident.upper().startswith("PMID:"):
        url = f"https://api.openalex.org/works/pmid:{ident.split(':', 1)[1]}"
    elif ident.upper().startswith("DOI:"):
        url = f"https://api.openalex.org/works/doi:{urllib.parse.quote(ident.split(':', 1)[1])}"
    else:
        url = f"https://api.openalex.org/works/{urllib.parse.quote(ident)}"
    return _http_json(url)


def edge_key(relation: str, x_id: str, y_id: str) -> str:
    return f"{relation}|{x_id}|{y_id}"


def make_edge(x_id: str, x_type: str, y_id: str, y_type: str, relation: str, display: str, source: str, credibility: int = 3, **extra: Any) -> dict[str, Any]:
    row = {
        "x_id": x_id,
        "x_type": x_type,
        "y_id": y_id,
        "y_type": y_type,
        "relation": relation,
        "display_relation": display,
        "source": source,
        "credibility": int(credibility),
    }
    row.update(extra)
    return row


def evidence_for_edge(edge: dict[str, Any], *, evidence_type: str, source_dataset: str, source_release: str, source_record_id: str, accessed_at: str, raw: Any, **extra: Any) -> dict[str, Any]:
    row = {
        "edge_key": edge_key(edge["relation"], edge["x_id"], edge["y_id"]),
        "relation": edge["relation"],
        "x_id": edge["x_id"],
        "x_type": edge["x_type"],
        "y_id": edge["y_id"],
        "y_type": edge["y_type"],
        "evidence_type": evidence_type,
        "source": edge["source"],
        "source_dataset": source_dataset,
        "source_release": source_release,
        "source_record_id": source_record_id,
        "paper_id": "",
        "doi": "",
        "pmid": "",
        "openalex_work_id": "",
        "dataset_id": edge["x_id"] if edge["x_type"] == "dataset" else edge["y_id"] if edge["y_type"] == "dataset" else "",
        "source_dataset_id": "",
        "citation_direction": "",
        "mapping_confidence": "source_native_metadata",
        "release_date": "",
        "accessed_at": accessed_at,
        "source_url": "",
        "license": "",
        "raw_json": _json(raw),
    }
    row.update({k: _clean(v) for k, v in extra.items()})
    return row


def default_datasets() -> list[dict[str, Any]]:
    return [
        {
            "id": "OpenTargets:target_essentiality:26.03",
            "name": "OpenTargets target_essentiality 26.03 (DepMap gene essentiality context)",
            "source": "OpenTargets/target_essentiality",
            "source_dataset_id": "target_essentiality",
            "source_release": SOURCE_RELEASE,
            "release_date": "2026-03",
            "license": "Open Targets Platform data; source metadata staging only, review upstream terms before canonical promotion",
            "source_url": "gs://open-targets-data-releases/26.03/output/etl/parquet/target_essentiality/",
        },
        {
            "id": "OpenTargets:drug_molecule:26.03",
            "name": "OpenTargets drug_molecule 26.03",
            "source": "OpenTargets/drug_molecule",
            "source_dataset_id": "drug_molecule",
            "source_release": SOURCE_RELEASE,
            "release_date": "2026-03",
            "license": "Open Targets Platform data; source metadata staging only, review upstream terms before canonical promotion",
            "source_url": "gs://open-targets-data-releases/26.03/output/etl/parquet/drug_molecule/",
        },
        {
            "id": "OpenTargets:biosample:26.03",
            "name": "OpenTargets biosample 26.03 cell-type metadata",
            "source": "OpenTargets/biosample",
            "source_dataset_id": "biosample",
            "source_release": SOURCE_RELEASE,
            "release_date": "2026-03",
            "license": "Open Targets Platform data; raw table not downloaded in default pilot if requester-pays blocks access",
            "source_url": "gs://open-targets-data-releases/26.03/output/etl/parquet/biosample/",
        },
    ]


def build_paper_edges(accessed_at: str, max_refs_per_paper: int = 8) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    seed_specs = [
        {
            "identifier": "PMID:39657122",
            "dataset_ids": ["OpenTargets:target_essentiality:26.03", "OpenTargets:drug_molecule:26.03", "OpenTargets:biosample:26.03"],
            "source_record_id": "OpenTargetsPlatformPaper:2024:gkae1128",
            "mapping_confidence": "platform_paper_to_release_dataset_metadata; review_required",
        },
        {
            "identifier": "PMID:28753430",
            "dataset_ids": ["OpenTargets:target_essentiality:26.03"],
            "source_record_id": "DepMapPaper:2017:cell_2017_06_010",
            "mapping_confidence": "depmap_primary_paper_to_downstream_opentargets_dataset_metadata; review_required",
        },
    ]
    papers: list[dict[str, Any]] = []
    produced_edges: list[dict[str, Any]] = []
    produced_ev: list[dict[str, Any]] = []
    citation_edges: list[dict[str, Any]] = []
    citation_ev: list[dict[str, Any]] = []
    seen_papers: set[str] = set()

    for spec in seed_specs:
        work = fetch_openalex_work(spec["identifier"])
        paper = paper_node_from_work(work)
        if paper["id"] not in seen_papers:
            seen_papers.add(paper["id"])
            papers.append(paper)
        for dataset_id in spec["dataset_ids"]:
            edge = make_edge(
                paper["id"],
                "paper",
                dataset_id,
                "dataset",
                "paper_produced_dataset",
                "produced dataset",
                "OpenAlex/OpenTargets curated provenance audit",
                1,
                doi=paper["doi"],
                pmid=paper["pmid"],
                openalex_work_id=paper["openalex_work_id"],
                source_dataset_id=dataset_id,
                mapping_confidence=spec["mapping_confidence"],
            )
            produced_edges.append(edge)
            produced_ev.append(
                evidence_for_edge(
                    edge,
                    evidence_type="dataset_provenance_metadata",
                    source_dataset="openalex_works_plus_opentargets_release_metadata",
                    source_release=SOURCE_RELEASE,
                    source_record_id=spec["source_record_id"],
                    accessed_at=accessed_at,
                    raw={"work": work, "seed_spec": spec},
                    paper_id=paper["id"],
                    doi=paper["doi"],
                    pmid=paper["pmid"],
                    openalex_work_id=paper["openalex_work_id"],
                    dataset_id=dataset_id,
                    source_dataset_id=dataset_id,
                    mapping_confidence=spec["mapping_confidence"],
                    source_url=f"https://openalex.org/{paper['openalex_work_id']}",
                    license=paper["license"],
                )
            )

        for ref_url in (work.get("referenced_works") or [])[:max_refs_per_paper]:
            ref_id = openalex_compact_id(ref_url)
            try:
                ref_work = fetch_openalex_work(ref_id)
            except Exception:
                ref_work = {"id": ref_url, "ids": {}, "title": ""}
            ref_paper = paper_node_from_work(ref_work)
            if ref_paper["id"] not in seen_papers:
                seen_papers.add(ref_paper["id"])
                papers.append(ref_paper)
            edge = make_edge(
                paper["id"],
                "paper",
                ref_paper["id"],
                "paper",
                "paper_cites_paper",
                "cites",
                "OpenAlex",
                3,
                citation_direction="x_cites_y",
                openalex_work_id=paper["openalex_work_id"],
                cited_openalex_work_id=ref_paper["openalex_work_id"],
            )
            citation_edges.append(edge)
            citation_ev.append(
                evidence_for_edge(
                    edge,
                    evidence_type="citation_metadata",
                    source_dataset="openalex_referenced_works",
                    source_release="OpenAlex live API",
                    source_record_id=f"OpenAlex:{paper['openalex_work_id']}:references:{ref_paper['openalex_work_id']}",
                    accessed_at=accessed_at,
                    raw={"citing_work": work, "cited_work": ref_work},
                    paper_id=paper["id"],
                    doi=paper["doi"],
                    pmid=paper["pmid"],
                    openalex_work_id=paper["openalex_work_id"],
                    citation_direction="x_cites_y",
                    source_url=f"https://openalex.org/{paper['openalex_work_id']}",
                    license="OpenAlex CC0 metadata",
                )
            )
    edges = {
        "paper_produced_dataset": pd.DataFrame(produced_edges),
        "paper_cites_paper": pd.DataFrame(citation_edges),
    }
    evidence = {
        "paper_produced_dataset": pd.DataFrame(produced_ev),
        "paper_cites_paper": pd.DataFrame(citation_ev),
    }
    return pd.DataFrame(papers), edges, evidence


def _first_parquet_files(path: Path) -> list[Path]:
    if not path.exists():
        return []
    if path.is_file() and path.suffix == ".parquet":
        return [path]
    return sorted(path.glob("*.parquet")) or sorted(path.glob("*.snappy.parquet"))


def _read_parquet_limited(path: Path, columns: list[str] | None = None, max_rows: int = 1000) -> pd.DataFrame:
    frames = []
    total = 0
    for file in _first_parquet_files(path):
        df = pd.read_parquet(file, columns=columns)
        if max_rows:
            df = df.head(max_rows - total)
        frames.append(df)
        total += len(df)
        if max_rows and total >= max_rows:
            break
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _to_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes, dict)):
        converted = value.tolist()
        return converted if isinstance(converted, list) else [converted]
    try:
        if pd.isna(value):
            return []
    except (TypeError, ValueError):
        pass
    return [value]


def normalize_curie(value: Any) -> str:
    text = _clean(value)
    if not text:
        return ""
    if "_" in text and ":" not in text:
        prefix, rest = text.split("_", 1)
        if prefix in {"EFO", "MONDO", "HP", "DOID", "UBERON", "CL"}:
            return f"{prefix}:{rest}"
    return text


def build_containment_edges(raw_root: Path, kg_root: Path | None, accessed_at: str, max_rows: int = 1000) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    edges: dict[str, list[dict[str, Any]]] = {r: [] for r in RELATION_TARGET_TYPE}
    evidence: dict[str, list[dict[str, Any]]] = {r: [] for r in RELATION_TARGET_TYPE}

    drug_dir = raw_root / "opentargets" / "26.03" / "drug_molecule"
    drugs = _read_parquet_limited(drug_dir, columns=["id", "name"], max_rows=max_rows)
    for idx, row in drugs.iterrows():
        chembl = _clean(row.get("id"))
        if not chembl.startswith("CHEMBL"):
            continue
        edge = make_edge("OpenTargets:drug_molecule:26.03", "dataset", chembl, "molecule", "dataset_contains_molecule", "contains molecule", "OpenTargets/drug_molecule", 3)
        edges[edge["relation"]].append(edge)
        evidence[edge["relation"]].append(evidence_for_edge(edge, evidence_type="source_table_membership", source_dataset="drug_molecule", source_release=SOURCE_RELEASE, source_record_id=f"drug_molecule:{chembl}", accessed_at=accessed_at, raw=row.to_dict(), source_dataset_id="drug_molecule", mapping_confidence="source_native_chembl_id", source_url="gs://open-targets-data-releases/26.03/output/etl/parquet/drug_molecule/"))

    te_dir = raw_root / "opentargets-26.03" / "target_essentiality"
    te = _read_parquet_limited(te_dir, columns=["id", "geneEssentiality"], max_rows=max_rows)
    for _, row in te.iterrows():
        for essentiality in _to_list(row.get("geneEssentiality")):
            if not isinstance(essentiality, dict):
                continue
            for depmap in _to_list(essentiality.get("depMapEssentiality")):
                if not isinstance(depmap, dict):
                    continue
                tissue_id = normalize_curie(depmap.get("tissueId"))
                if tissue_id.startswith("UBERON:"):
                    edge = make_edge("OpenTargets:target_essentiality:26.03", "dataset", tissue_id, "tissue", "dataset_contains_tissue", "contains tissue", "OpenTargets/target_essentiality", 3)
                    edges[edge["relation"]].append(edge)
                    evidence[edge["relation"]].append(evidence_for_edge(edge, evidence_type="source_table_membership", source_dataset="target_essentiality", source_release=SOURCE_RELEASE, source_record_id=f"target_essentiality:tissue:{tissue_id}", accessed_at=accessed_at, raw=depmap, source_dataset_id="target_essentiality", mapping_confidence="source_native_uberon_id", source_url="gs://open-targets-data-releases/26.03/output/etl/parquet/target_essentiality/"))
                for screen in _to_list(depmap.get("screens")):
                    if not isinstance(screen, dict):
                        continue
                    depmap_id = _clean(screen.get("depmapId"))
                    if depmap_id:
                        edge = make_edge("OpenTargets:target_essentiality:26.03", "dataset", depmap_id, "cell_line", "dataset_contains_cell_line", "contains cell line", "OpenTargets/target_essentiality", 3)
                        edges[edge["relation"]].append(edge)
                        evidence[edge["relation"]].append(evidence_for_edge(edge, evidence_type="source_table_membership", source_dataset="target_essentiality", source_release=SOURCE_RELEASE, source_record_id=f"target_essentiality:depmap:{depmap_id}", accessed_at=accessed_at, raw=screen, source_dataset_id="target_essentiality", mapping_confidence="source_native_depmap_id", source_url="gs://open-targets-data-releases/26.03/output/etl/parquet/target_essentiality/"))
                    disease_id = normalize_curie(screen.get("diseaseCellLineId"))
                    if disease_id.startswith(("EFO:", "MONDO:", "Orphanet:", "HP:", "DOID:")):
                        edge = make_edge("OpenTargets:target_essentiality:26.03", "dataset", disease_id, "disease", "dataset_contains_disease", "contains disease", "OpenTargets/target_essentiality", 3)
                        edges[edge["relation"]].append(edge)
                        evidence[edge["relation"]].append(evidence_for_edge(edge, evidence_type="source_table_membership", source_dataset="target_essentiality", source_release=SOURCE_RELEASE, source_record_id=f"target_essentiality:disease:{disease_id}", accessed_at=accessed_at, raw=screen, source_dataset_id="target_essentiality", mapping_confidence="source_native_disease_curie", source_url="gs://open-targets-data-releases/26.03/output/etl/parquet/target_essentiality/"))

    # Cell-type pilot uses existing node provenance when raw biosample is not cached.
    if kg_root is not None:
        cell_type_nodes = kg_root / "nodes" / "cell_type.parquet"
        if cell_type_nodes.exists():
            cells = pd.read_parquet(cell_type_nodes, columns=["id", "name", "source"]).head(max(1, min(max_rows, 100)))
            for _, row in cells.iterrows():
                cell_id = _clean(row.get("id"))
                if cell_id.startswith("CL:") and _clean(row.get("source")) == "OpenTargets":
                    edge = make_edge("OpenTargets:biosample:26.03", "dataset", cell_id, "cell_type", "dataset_contains_cell_type", "contains cell type", "OpenTargets/biosample node provenance", 1)
                    edges[edge["relation"]].append(edge)
                    evidence[edge["relation"]].append(evidence_for_edge(edge, evidence_type="node_provenance_membership", source_dataset="biosample", source_release=SOURCE_RELEASE, source_record_id=f"cell_type_node:{cell_id}", accessed_at=accessed_at, raw=row.to_dict(), source_dataset_id="biosample", mapping_confidence="node_source_only_raw_biosample_not_cached", source_url="gs://open-targets-data-releases/26.03/output/etl/parquet/biosample/", license="Open Targets Platform data; requester-pays raw access not used in default pilot"))

    out_edges: dict[str, pd.DataFrame] = {}
    out_ev: dict[str, pd.DataFrame] = {}
    for relation in RELATION_TARGET_TYPE:
        df = pd.DataFrame(edges[relation])
        ev = pd.DataFrame(evidence[relation])
        if not df.empty:
            df = df.drop_duplicates(subset=["x_id", "y_id", "relation", "source"]).reset_index(drop=True)
        if not ev.empty:
            ev = ev.drop_duplicates(subset=["edge_key", "source_record_id"]).reset_index(drop=True)
        out_edges[relation] = df
        out_ev[relation] = ev
    return out_edges, out_ev


def validate(result: BuildResult, kg_root: Path | None) -> dict[str, Any]:
    validation: dict[str, Any] = {
        "canonical_promotion": False,
        "only_metadata_or_literature_relations": True,
        "no_paper_comention_biological_assertions": True,
        "required_metadata_columns_present": True,
        "counts": {},
        "endpoint_anti_joins": {},
        "warnings": [],
    }
    allowed = {"paper_produced_dataset", "paper_cites_paper", *RELATION_TARGET_TYPE.keys()}
    for relation, df in result.edges.items():
        validation["counts"][relation] = int(len(df))
        if relation not in allowed:
            validation["only_metadata_or_literature_relations"] = False
        if not df.empty:
            missing = [c for c in EDGE_COLUMNS if c not in df.columns]
            if missing:
                validation["required_metadata_columns_present"] = False
                validation["warnings"].append({"relation": relation, "missing_edge_columns": missing})
        ev = result.evidence.get(relation, pd.DataFrame())
        if not ev.empty:
            missing_ev = [c for c in ["doi", "pmid", "source_record_id", "mapping_confidence", "source_release", "accessed_at"] if c not in ev.columns]
            if missing_ev:
                validation["required_metadata_columns_present"] = False
                validation["warnings"].append({"relation": relation, "missing_evidence_columns": missing_ev})

    if kg_root is not None and kg_root.exists():
        for relation, target_type in RELATION_TARGET_TYPE.items():
            df = result.edges.get(relation, pd.DataFrame())
            node_path = kg_root / "nodes" / f"{target_type}.parquet"
            if df.empty or not node_path.exists():
                continue
            nodes = pd.read_parquet(node_path, columns=["id"])
            missing = sorted(set(df["y_id"].astype(str)) - set(nodes["id"].astype(str)))
            validation["endpoint_anti_joins"][relation] = {"missing_y_count": len(missing), "sample": missing[:20]}
    return validation


def build_paper_dataset_provenance(raw_root: Path, kg_root: Path | None, max_rows: int = 1000, max_refs_per_paper: int = 8) -> BuildResult:
    accessed_at = now_utc()
    source_audit = audit_metadata_sources(accessed_at)
    papers, paper_edges, paper_evidence = build_paper_edges(accessed_at, max_refs_per_paper=max_refs_per_paper)
    containment_edges, containment_evidence = build_containment_edges(raw_root, kg_root, accessed_at, max_rows=max_rows)
    edges = {**paper_edges, **containment_edges}
    evidence = {**paper_evidence, **containment_evidence}
    result = BuildResult(
        papers=papers.drop_duplicates(subset=["id"]).reset_index(drop=True),
        datasets=pd.DataFrame(default_datasets()).drop_duplicates(subset=["id"]).reset_index(drop=True),
        edges=edges,
        evidence=evidence,
        source_audit=source_audit,
        validation={},
    )
    validation = validate(result, kg_root)
    return BuildResult(result.papers, result.datasets, result.edges, result.evidence, result.source_audit, validation)


def _write_df(df: pd.DataFrame, path: Path, columns: list[str] | None = None) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    if df.empty:
        if columns is None:
            columns = []
        df = pd.DataFrame(columns=columns)
    df.to_parquet(path, index=False)
    return int(len(df))


def write_outputs(result: BuildResult, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    counts["paper_nodes"] = _write_df(result.papers, output_dir / "nodes" / "paper.parquet", PAPER_COLUMNS)
    counts["dataset_nodes"] = _write_df(result.datasets, output_dir / "nodes" / "dataset.parquet", DATASET_COLUMNS)
    for relation, df in sorted(result.edges.items()):
        counts[f"edges/{relation}"] = _write_df(df, output_dir / "edges" / f"{relation}.parquet", EDGE_COLUMNS)
    for relation, df in sorted(result.evidence.items()):
        counts[f"evidence/{relation}"] = _write_df(df, output_dir / "evidence" / f"{relation}.parquet", EVIDENCE_COLUMNS)
    (output_dir / "validation").mkdir(exist_ok=True)
    (output_dir / "validation" / "source_license_access_audit.json").write_text(json.dumps(result.source_audit, indent=2, sort_keys=True) + "\n")
    manifest = {
        "ok": bool(result.validation.get("only_metadata_or_literature_relations") and result.validation.get("required_metadata_columns_present")),
        "canonical_promotion": False,
        "counts": counts,
        "validation": result.validation,
        "relations": sorted(result.edges),
        "notes": [
            "Staging-only pilot; no canonical promotion without review.",
            "Paper citations and paper/dataset links are metadata/provenance only, not biological assertions.",
            "OpenAlex is used for citation direction; Crossref/Europe PMC/Semantic Scholar are audited for access/licensing.",
        ],
    }
    (output_dir / "validation" / "paper_dataset_provenance_build_report.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=Path("artifacts/cache/raw"))
    parser.add_argument("--kg-root", type=Path, default=Path("/Users/jkobject/mnt/gcs/jouvencekb/main"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-rows", type=int, default=1000)
    parser.add_argument("--max-refs-per-paper", type=int, default=8)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    kg_root = args.kg_root if args.kg_root and args.kg_root.exists() else None
    result = build_paper_dataset_provenance(args.raw_root, kg_root, max_rows=args.max_rows, max_refs_per_paper=args.max_refs_per_paper)
    manifest = write_outputs(result, args.output_dir)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if manifest.get("ok") else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
