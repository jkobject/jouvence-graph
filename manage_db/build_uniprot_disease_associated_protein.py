"""Build staged source-native disease_associated_protein edges.

This module is intentionally staging-only. It emits ``disease_associated_protein``
only when the source row directly names a UniProt protein/isoform endpoint and a
disease assertion. It does not project gene disease associations to all proteins.

Accepted source-native inputs in this pilot:

- UniProtKB reviewed human DISEASE comments (protein entry -> disease)
- UniProt humsavar missense variants with disease names/MIM IDs (variant is
  tied to a UniProt accession/FTId; materialized as protein -> disease evidence)

OpenTargets disease evidence, Reactome disease events, Complex Portal disease
fields, and existing protein mapping files are audited but not materialized here
unless they expose a direct protein/isoform -> disease assertion.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import fsspec
import pandas as pd

RELATION = "disease_associated_protein"
DISPLAY_RELATION = "associated with"
SOURCE_UNIPROT = "UniProtKB"
SOURCE_HUMSAVAR = "UniProtKB/humsavar"
UNIPROT_DISEASE_DATASET = "reviewed_human_disease_comments"
HUMSAVAR_DATASET = "humsavar_missense_variants"
HUMSAVAR_URL = "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/variants/humsavar.txt"
CANONICAL_ROOTS = {"/mnt/gcs/jouvencekb/main", "gs://jouvencekb/main"}

EDGE_COLUMNS = ["x_id", "x_type", "y_id", "y_type", "relation", "display_relation", "source", "credibility"]
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
    "uniprot_accession",
    "uniprot_entry_id",
    "ensp_id",
    "isoform",
    "disease_source_id",
    "disease_name",
    "disease_acronym",
    "disease_description",
    "variant_ft_id",
    "aa_change",
    "variant_category",
    "dbsnp_id",
    "eco_codes",
    "pmids",
    "mapping_confidence",
    "mapping_method",
    "source_native_endpoint_policy",
]

REJECTED_COLUMNS = [
    "source",
    "source_dataset",
    "source_record_id",
    "uniprot_accession",
    "disease_source_id",
    "disease_name",
    "reason",
    "raw_text",
]

MIM_RE = re.compile(r"\[(?:MIM|OMIM):\s*([0-9]{6})\]", re.IGNORECASE)


def _clean(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _pipe(values: Iterable[str]) -> str:
    return "|".join(sorted({v for v in values if v}))


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _hash_id(payload: Mapping[str, Any]) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=_json_default)
    return hashlib.sha256(data.encode()).hexdigest()[:24]


def _protein_mapping(protein_nodes: pd.DataFrame) -> dict[str, str]:
    if "id" not in protein_nodes.columns or "uniprot_id" not in protein_nodes.columns:
        raise ValueError("protein nodes must contain id and uniprot_id columns")
    out: dict[str, str] = {}
    for _, row in protein_nodes[["id", "uniprot_id"]].dropna().iterrows():
        protein_id = _clean(row["id"])
        for token in _clean(row["uniprot_id"]).split("|"):
            accession = token.strip()
            if accession and accession not in out:
                out[accession] = protein_id
    return out


def _disease_mapping(disease_nodes: pd.DataFrame) -> dict[str, str]:
    if "id" not in disease_nodes.columns:
        raise ValueError("disease nodes must contain id column")
    out: dict[str, str] = {}
    for _, row in disease_nodes.iterrows():
        disease_id = _clean(row.get("id"))
        if not disease_id:
            continue
        for column, prefixes in {
            "id": [""],
            "mondo_id": [""],
            "omim_id": ["OMIM:", "MIM:"],
            "mesh_id": [""],
            "doid_id": [""],
            "hp_id": [""],
            "efo_id": [""],
        }.items():
            if column not in disease_nodes.columns:
                continue
            raw = _clean(row.get(column))
            if not raw:
                continue
            for token in re.split(r"[|,;]", raw):
                value = token.strip()
                if not value:
                    continue
                out.setdefault(value, disease_id)
                for prefix in prefixes:
                    if prefix and not value.upper().startswith(prefix):
                        out.setdefault(prefix + value, disease_id)
    return out


def _extract_evidence(evidences: Sequence[Mapping[str, Any]]) -> tuple[str, str]:
    eco: list[str] = []
    pmids: list[str] = []
    for ev in evidences:
        code = _clean(ev.get("evidenceCode"))
        if code:
            eco.append(code)
        if _clean(ev.get("source")).lower() == "pubmed":
            pmid = _clean(ev.get("id"))
            if pmid:
                pmids.append(f"PMID:{pmid}" if pmid.isdigit() else pmid)
    return _pipe(eco), _pipe(pmids)


def _entry_disease_comments(entry: Mapping[str, Any]) -> list[dict[str, Any]]:
    comments = entry.get("comments") or []
    return [c for c in comments if _clean(c.get("commentType")).upper() == "DISEASE"]


def _disease_comment_text(comment: Mapping[str, Any]) -> str:
    texts = comment.get("texts") or []
    return " ".join(_clean(t.get("value")) for t in texts if _clean(t.get("value")))


def _disease_comment_evidence(comment: Mapping[str, Any]) -> tuple[str, str]:
    evidences: list[Mapping[str, Any]] = []
    for t in comment.get("texts") or []:
        evidences.extend(cast(list[Mapping[str, Any]], t.get("evidences") or []))
    disease = comment.get("disease") or {}
    evidences.extend(cast(list[Mapping[str, Any]], disease.get("evidences") or []))
    return _extract_evidence(evidences)


def _comment_disease_source_id(comment: Mapping[str, Any]) -> tuple[str, str, str, str]:
    disease = comment.get("disease") or {}
    name = _clean(disease.get("diseaseId"))
    acronym = _clean(disease.get("acronym"))
    description = _clean(disease.get("description"))
    xref = disease.get("diseaseCrossReference") or {}
    db = _clean(xref.get("database"))
    xref_id = _clean(xref.get("id"))
    if db and xref_id:
        if db.upper() in {"MIM", "OMIM"} and xref_id.isdigit():
            return f"OMIM:{xref_id}", name, acronym, description
        return f"{db}:{xref_id}", name, acronym, description
    return "", name, acronym, description


def _parse_humsavar_release(text: str) -> str:
    for line in text.splitlines()[:40]:
        if line.startswith("Release:"):
            return line.split(":", 1)[1].strip()
    return ""


def parse_humsavar_text(text: str) -> tuple[list[dict[str, str]], str]:
    rows: list[dict[str, str]] = []
    release = _parse_humsavar_release(text)
    for line in text.splitlines():
        if not line or line.startswith("-") or line.startswith("_") or line.startswith(" "):
            continue
        if line.startswith(("Main", "gene", "Description", "Name:", "Release:", "Statistics", "This ", "The ")):
            continue
        parts = line.split()
        if len(parts) < 7:
            continue
        gene, accession, ftid, aa_change, category, dbsnp = parts[:6]
        disease_name = line[74:].strip() if len(line) >= 74 else " ".join(parts[6:]).strip()
        if not accession or not ftid.startswith("VAR_"):
            continue
        rows.append(
            {
                "gene": gene,
                "uniprot_accession": accession,
                "variant_ft_id": ftid,
                "aa_change": aa_change,
                "variant_category": category,
                "dbsnp_id": "" if dbsnp == "-" else dbsnp,
                "disease_name": "" if disease_name == "-" else disease_name,
            }
        )
    return rows, release


def _humsavar_disease_source_id(disease_name: str) -> str:
    match = MIM_RE.search(disease_name)
    return f"OMIM:{match.group(1)}" if match else ""


def build_rows(
    *,
    uniprot_entries: Sequence[Mapping[str, Any]],
    humsavar_rows: Sequence[Mapping[str, str]],
    protein_nodes: pd.DataFrame,
    disease_nodes: pd.DataFrame,
    uniprot_release: str,
    humsavar_release: str,
    created_at: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    created_at = created_at or datetime.now(timezone.utc).isoformat()
    accession_to_protein = _protein_mapping(protein_nodes)
    disease_to_node = _disease_mapping(disease_nodes)
    edge_records: dict[tuple[str, str], dict[str, Any]] = {}
    evidence_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    counts: dict[str, int] = {
        "uniprot_entries_seen": 0,
        "uniprot_disease_comments_seen": 0,
        "uniprot_comments_accepted": 0,
        "humsavar_rows_seen": 0,
        "humsavar_rows_with_disease": 0,
        "humsavar_rows_accepted": 0,
        "missing_protein_mapping": 0,
        "missing_disease_mapping": 0,
    }

    def accept(
        *,
        protein_id: str,
        disease_id: str,
        source: str,
        source_dataset: str,
        source_record_id: str,
        release: str,
        uniprot_accession: str,
        uniprot_entry_id: str = "",
        disease_source_id: str,
        disease_name: str,
        disease_acronym: str = "",
        disease_description: str = "",
        predicate: str,
        text_span: str = "",
        variant_ft_id: str = "",
        aa_change: str = "",
        variant_category: str = "",
        dbsnp_id: str = "",
        eco_codes: str = "",
        pmids: str = "",
        evidence_type: str = "database_record",
        paper_id: str = "",
    ) -> None:
        edge_key = f"{RELATION}|{protein_id}|{disease_id}"
        edge_records.setdefault(
            (protein_id, disease_id),
            {
                "x_id": protein_id,
                "x_type": "protein",
                "y_id": disease_id,
                "y_type": "disease",
                "relation": RELATION,
                "display_relation": DISPLAY_RELATION,
                "source": source,
                "credibility": 3,
            },
        )
        evidence_rows.append(
            {
                "edge_key": edge_key,
                "relation": RELATION,
                "x_id": protein_id,
                "x_type": "protein",
                "y_id": disease_id,
                "y_type": "disease",
                "evidence_type": evidence_type,
                "source": source,
                "source_dataset": source_dataset,
                "source_record_id": source_record_id,
                "paper_id": paper_id,
                "dataset_id": "",
                "study_id": "",
                "evidence_score": "",
                "effect_size": "",
                "p_value": "",
                "direction": "protein_to_disease",
                "confidence_interval": "",
                "predicate": predicate,
                "text_span": text_span,
                "section": "UniProt disease annotation" if source_dataset == UNIPROT_DISEASE_DATASET else "UniProt humsavar variant index",
                "extraction_method": "source_native_uniprot_accession_no_gene_projection",
                "license": "UniProt terms; verify before canonical promotion",
                "release": release,
                "created_at": created_at,
                "uniprot_accession": uniprot_accession,
                "uniprot_entry_id": uniprot_entry_id,
                "ensp_id": protein_id,
                "isoform": "canonical",
                "disease_source_id": disease_source_id,
                "disease_name": disease_name,
                "disease_acronym": disease_acronym,
                "disease_description": disease_description,
                "variant_ft_id": variant_ft_id,
                "aa_change": aa_change,
                "variant_category": variant_category,
                "dbsnp_id": dbsnp_id,
                "eco_codes": eco_codes,
                "pmids": pmids,
                "mapping_confidence": "exact_uniprot_accession_to_existing_protein_node;exact_disease_xref_to_existing_disease_node",
                "mapping_method": "nodes/protein.uniprot_id and nodes/disease xref columns",
                "source_native_endpoint_policy": "source directly names UniProt protein/isoform accession; no gene-to-protein projection",
            }
        )

    for entry in uniprot_entries:
        counts["uniprot_entries_seen"] += 1
        accession = _clean(entry.get("primaryAccession"))
        protein_id = accession_to_protein.get(accession, "")
        entry_id = _clean(entry.get("uniProtkbId"))
        for comment in _entry_disease_comments(entry):
            counts["uniprot_disease_comments_seen"] += 1
            disease_source_id, disease_name, acronym, description = _comment_disease_source_id(comment)
            disease_id = disease_to_node.get(disease_source_id, "") if disease_source_id else ""
            record_id = f"{accession}:DISEASE:{disease_source_id or _hash_id(comment)}"
            if not protein_id:
                counts["missing_protein_mapping"] += 1
                rejected_rows.append({"source": SOURCE_UNIPROT, "source_dataset": UNIPROT_DISEASE_DATASET, "source_record_id": record_id, "uniprot_accession": accession, "disease_source_id": disease_source_id, "disease_name": disease_name, "reason": "missing_uniprot_to_protein_node_mapping", "raw_text": _disease_comment_text(comment)})
                continue
            if not disease_id:
                counts["missing_disease_mapping"] += 1
                rejected_rows.append({"source": SOURCE_UNIPROT, "source_dataset": UNIPROT_DISEASE_DATASET, "source_record_id": record_id, "uniprot_accession": accession, "disease_source_id": disease_source_id, "disease_name": disease_name, "reason": "missing_source_disease_xref_or_disease_node_mapping", "raw_text": _disease_comment_text(comment)})
                continue
            eco_codes, pmids = _disease_comment_evidence(comment)
            accept(protein_id=protein_id, disease_id=disease_id, source=SOURCE_UNIPROT, source_dataset=UNIPROT_DISEASE_DATASET, source_record_id=record_id, release=uniprot_release, uniprot_accession=accession, uniprot_entry_id=entry_id, disease_source_id=disease_source_id, disease_name=disease_name, disease_acronym=acronym, disease_description=description, predicate="UniProt DISEASE comment", text_span=_disease_comment_text(comment), eco_codes=eco_codes, pmids=pmids)
            for pmid in [p for p in pmids.split("|") if p]:
                accept(protein_id=protein_id, disease_id=disease_id, source=SOURCE_UNIPROT, source_dataset=UNIPROT_DISEASE_DATASET, source_record_id=f"{record_id}:{pmid}", release=uniprot_release, uniprot_accession=accession, uniprot_entry_id=entry_id, disease_source_id=disease_source_id, disease_name=disease_name, disease_acronym=acronym, disease_description=description, predicate="UniProt DISEASE comment", text_span=_disease_comment_text(comment), eco_codes=eco_codes, pmids=pmids, evidence_type="paper", paper_id=pmid)
            counts["uniprot_comments_accepted"] += 1

    for row in humsavar_rows:
        counts["humsavar_rows_seen"] += 1
        disease_name = _clean(row.get("disease_name"))
        if not disease_name:
            continue
        counts["humsavar_rows_with_disease"] += 1
        accession = _clean(row.get("uniprot_accession"))
        protein_id = accession_to_protein.get(accession, "")
        disease_source_id = _humsavar_disease_source_id(disease_name)
        disease_id = disease_to_node.get(disease_source_id, "") if disease_source_id else ""
        source_record_id = _clean(row.get("variant_ft_id")) or f"{accession}:{_hash_id(row)}"
        if not protein_id:
            counts["missing_protein_mapping"] += 1
            rejected_rows.append({"source": SOURCE_HUMSAVAR, "source_dataset": HUMSAVAR_DATASET, "source_record_id": source_record_id, "uniprot_accession": accession, "disease_source_id": disease_source_id, "disease_name": disease_name, "reason": "missing_uniprot_to_protein_node_mapping", "raw_text": disease_name})
            continue
        if not disease_id:
            counts["missing_disease_mapping"] += 1
            rejected_rows.append({"source": SOURCE_HUMSAVAR, "source_dataset": HUMSAVAR_DATASET, "source_record_id": source_record_id, "uniprot_accession": accession, "disease_source_id": disease_source_id, "disease_name": disease_name, "reason": "missing_mim_or_disease_node_mapping", "raw_text": disease_name})
            continue
        accept(protein_id=protein_id, disease_id=disease_id, source=SOURCE_HUMSAVAR, source_dataset=HUMSAVAR_DATASET, source_record_id=source_record_id, release=humsavar_release, uniprot_accession=accession, disease_source_id=disease_source_id, disease_name=disease_name, predicate="UniProt humsavar disease variant", text_span=disease_name, variant_ft_id=_clean(row.get("variant_ft_id")), aa_change=_clean(row.get("aa_change")), variant_category=_clean(row.get("variant_category")), dbsnp_id=_clean(row.get("dbsnp_id")))
        counts["humsavar_rows_accepted"] += 1

    edges = pd.DataFrame(list(edge_records.values()), columns=EDGE_COLUMNS).sort_values(["x_id", "y_id"]).reset_index(drop=True)
    evidence = pd.DataFrame(evidence_rows, columns=EVIDENCE_COLUMNS).drop_duplicates(
        subset=["relation", "x_id", "y_id", "evidence_type", "source", "source_record_id", "paper_id"], keep="last"
    ).sort_values(["x_id", "y_id", "source", "source_record_id", "evidence_type"]).reset_index(drop=True)
    rejected = pd.DataFrame(rejected_rows, columns=REJECTED_COLUMNS).drop_duplicates().reset_index(drop=True)
    counts["edges"] = int(len(edges))
    counts["evidence_rows"] = int(len(evidence))
    counts["rejected_rows"] = int(len(rejected))
    validation = {
        "protein_endpoint_antijoin_pass": bool(edges.empty or set(edges["x_id"]).issubset(set(protein_nodes["id"].astype(str)))),
        "disease_endpoint_antijoin_pass": bool(edges.empty or set(edges["y_id"]).issubset(set(disease_nodes["id"].astype(str)))),
        "edges_without_evidence": int(0 if edges.empty else len(set(edges.apply(lambda r: f"{r.relation}|{r.x_id}|{r.y_id}", axis=1)) - set(evidence["edge_key"]))),
        "evidence_without_edge": int(0 if evidence.empty else len(set(evidence["edge_key"]) - set(edges.apply(lambda r: f"{r.relation}|{r.x_id}|{r.y_id}", axis=1)))),
        "source_native_policy": "only UniProt accession / humsavar UniProt FTId disease assertions accepted; OpenTargets gene disease associations are not projected to proteins",
    }
    return edges, evidence, rejected, {"counts": counts, "validation": validation}


def _next_link(headers: Mapping[str, str]) -> str | None:
    link = headers.get("Link") or headers.get("link") or ""
    for part in link.split(","):
        if 'rel="next"' in part:
            match = re.search(r"<([^>]+)>", part)
            if match:
                return match.group(1)
    return None


def _fetch_json_page(url: str, *, attempts: int = 4) -> tuple[dict[str, Any], Mapping[str, str]]:
    last_error: BaseException | None = None
    for attempt in range(1, attempts + 1):
        req = urllib.request.Request(url, headers={"User-Agent": "txgnn-disease-associated-protein/0.1"})
        try:
            with urllib.request.urlopen(req, timeout=120) as response:
                return json.loads(response.read()), dict(response.headers.items())
        except (urllib.error.URLError, TimeoutError, http.client.IncompleteRead, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt == attempts:
                break
            time.sleep(min(2**attempt, 10))
    assert last_error is not None
    raise last_error


def fetch_uniprot_disease_entries(*, size: int = 100, max_pages: int | None = None) -> tuple[list[dict[str, Any]], str, int]:
    query = "(reviewed:true) AND (organism_id:9606) AND (cc_disease:*)"
    params = {"query": query, "format": "json", "size": str(size)}
    url = "https://rest.uniprot.org/uniprotkb/search?" + urllib.parse.urlencode(params)
    entries: list[dict[str, Any]] = []
    release = ""
    total_results = 0
    pages = 0
    while url:
        pages += 1
        payload, headers = _fetch_json_page(url)
        release = headers.get("X-UniProt-Release") or release
        total_results = int(headers.get("X-Total-Results") or total_results or 0)
        entries.extend(cast(list[dict[str, Any]], payload.get("results") or []))
        url = _next_link(headers)
        if max_pages is not None and pages >= max_pages:
            break
    return entries, release, total_results


def fetch_humsavar_text() -> str:
    req = urllib.request.Request(HUMSAVAR_URL, headers={"User-Agent": "txgnn-disease-associated-protein/0.1"})
    with urllib.request.urlopen(req, timeout=180) as response:
        return response.read().decode("utf-8", "replace")


def _write_table(path: str, frame: pd.DataFrame) -> None:
    fs, fs_path = fsspec.core.url_to_fs(path)
    parent = fs_path.rsplit("/", 1)[0] if "/" in fs_path else ""
    if parent:
        fs.makedirs(parent, exist_ok=True)
    frame.to_parquet(path, index=False)


def _write_json(path: str, payload: Mapping[str, Any]) -> None:
    fs, fs_path = fsspec.core.url_to_fs(path)
    parent = fs_path.rsplit("/", 1)[0] if "/" in fs_path else ""
    if parent:
        fs.makedirs(parent, exist_ok=True)
    with fs.open(fs_path, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _join_uri(root: str, *parts: str) -> str:
    return root.rstrip("/") + "/" + "/".join(p.strip("/") for p in parts)


def _refuse_canonical(output_root: str) -> None:
    normalized = output_root.rstrip("/")
    if normalized in CANONICAL_ROOTS:
        raise ValueError("refusing to write canonical KG root; choose a staging root")
    if "://" not in normalized:
        try:
            if str(Path(normalized).resolve()) in CANONICAL_ROOTS:
                raise ValueError("refusing to write canonical KG root; choose a staging root")
        except FileNotFoundError:
            pass


def _read_json_payload(path: str) -> tuple[list[dict[str, Any]], str, int]:
    with fsspec.open(path, "r") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        entries = payload.get("entries") or payload.get("results") or []
        release = _clean(payload.get("release")) or "fixture"
        total = int(payload.get("total_results") or len(entries))
    else:
        entries = payload
        release = "fixture"
        total = len(entries)
    return cast(list[dict[str, Any]], entries), release, total


def audit_non_materialized_sources(output_root: str, created_at: str) -> dict[str, Any]:
    audit = {
        "created_at": created_at,
        "relation": RELATION,
        "sources": {
            "UniProtKB DISEASE comments": {
                "decision": "materialized_when_mapped",
                "reason": "UniProt entry/comment directly names a protein accession and disease cross-reference.",
            },
            "UniProtKB humsavar": {
                "decision": "materialized_when_mapped",
                "reason": "Each row directly names Swiss-Prot accession + variant FTId + disease name/MIM for missense variants.",
            },
            "ClinVar/UniProt variants": {
                "decision": "represented_by_humsavar_pilot_only",
                "reason": "Raw ClinVar variant_summary is variant/disease-native and gene/sequence-location centric; this pilot uses UniProt humsavar, which is explicitly UniProt protein accession/FTId native. Do not broaden ClinVar gene assertions to proteins.",
            },
            "OpenTargets disease evidence": {
                "decision": "not_materialized",
                "reason": "OpenTargets target-disease evidence is target/gene-centric; available local cache has clinical/drug datasets but no source-native protein/isoform disease endpoint suitable for this relation.",
            },
            "Reactome disease fields": {
                "decision": "not_materialized",
                "reason": "Reactome exposes disease events/pathways and participants, not a direct protein/isoform disease-association assertion; pathway disease context must not be projected onto every participant protein.",
            },
            "Complex Portal disease fields": {
                "decision": "not_materialized",
                "reason": "Complex Portal disease annotations are complex-level/context annotations; they do not assert every member protein is disease-associated.",
            },
            "Existing protein mapping sources": {
                "decision": "mapping_only",
                "reason": "nodes/protein.uniprot_id is used solely for endpoint resolution to ENSP IDs, not as disease evidence.",
            },
        },
    }
    _write_json(_join_uri(output_root, "reports", "source_native_audit.json"), audit)
    return audit


def build_uniprot_disease_associated_protein(
    *,
    output_root: str,
    protein_nodes_path: str,
    disease_nodes_path: str,
    max_pages: int | None = None,
    entries_json_path: str | None = None,
    humsavar_path: str | None = None,
) -> dict[str, Any]:
    _refuse_canonical(output_root)
    created_at = datetime.now(timezone.utc).isoformat()
    protein_nodes = pd.read_parquet(protein_nodes_path, columns=["id", "uniprot_id"])
    disease_cols = ["id", "name", "mondo_id", "omim_id", "mesh_id", "efo_id", "doid_id", "hp_id"]
    disease_nodes = pd.read_parquet(disease_nodes_path, columns=disease_cols)
    if entries_json_path:
        entries, uniprot_release, total_results = _read_json_payload(entries_json_path)
    else:
        entries, uniprot_release, total_results = fetch_uniprot_disease_entries(max_pages=max_pages)
    if humsavar_path:
        with fsspec.open(humsavar_path, "r") as handle:
            humsavar_text = handle.read()
    else:
        humsavar_text = fetch_humsavar_text()
    humsavar_rows, humsavar_release = parse_humsavar_text(humsavar_text)

    edges, evidence, rejected, stats = build_rows(
        uniprot_entries=entries,
        humsavar_rows=humsavar_rows,
        protein_nodes=protein_nodes,
        disease_nodes=disease_nodes,
        uniprot_release=uniprot_release,
        humsavar_release=humsavar_release,
        created_at=created_at,
    )
    _write_table(_join_uri(output_root, "edges", f"{RELATION}.parquet"), edges)
    _write_table(_join_uri(output_root, "evidence", f"{RELATION}.parquet"), evidence)
    _write_table(_join_uri(output_root, "diagnostics", "rejected_source_rows.parquet"), rejected)
    audit = audit_non_materialized_sources(output_root, created_at)
    manifest = {
        "created_at": created_at,
        "relation": RELATION,
        "staging_only": True,
        "canonical_promotion": False,
        "uniprot_release": uniprot_release,
        "uniprot_total_query_results": int(total_results),
        "humsavar_release": humsavar_release,
        "source_policy": "source-native protein/isoform disease assertions only; no gene-to-protein projection",
        "outputs": {
            "edges": _join_uri(output_root, "edges", f"{RELATION}.parquet"),
            "evidence": _join_uri(output_root, "evidence", f"{RELATION}.parquet"),
            "rejected_source_rows": _join_uri(output_root, "diagnostics", "rejected_source_rows.parquet"),
            "source_audit": _join_uri(output_root, "reports", "source_native_audit.json"),
            "manifest": _join_uri(output_root, "MANIFEST.json"),
        },
        "counts": stats["counts"],
        "validation": stats["validation"],
        "audit_summary": audit["sources"],
    }
    _write_json(_join_uri(output_root, "reports", "validation.json"), stats["validation"])
    _write_json(_join_uri(output_root, "MANIFEST.json"), manifest)
    return manifest


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, help="Local or gs:// staging root")
    parser.add_argument("--protein-nodes", default="gs://jouvencekb/main/nodes/protein.parquet")
    parser.add_argument("--disease-nodes", default="gs://jouvencekb/main/nodes/disease.parquet")
    parser.add_argument("--max-pages", type=int, default=None, help="Debug limit for UniProt pagination")
    parser.add_argument("--entries-json", default=None, help="Optional UniProt JSON fixture/cache")
    parser.add_argument("--humsavar", default=None, help="Optional humsavar.txt fixture/cache")
    args = parser.parse_args(list(argv) if argv is not None else None)
    manifest = build_uniprot_disease_associated_protein(
        output_root=args.output_root,
        protein_nodes_path=args.protein_nodes,
        disease_nodes_path=args.disease_nodes,
        max_pages=args.max_pages,
        entries_json_path=args.entries_json,
        humsavar_path=args.humsavar,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
