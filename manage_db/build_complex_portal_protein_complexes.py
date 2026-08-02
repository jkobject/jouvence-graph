"""Build staged Complex Portal protein_complex nodes and membership edges.

This module intentionally writes only to a staging directory. It downloads or
reads the Complex Portal human complextab file, preserves Complex Portal IDs as
staged ``protein_complex`` node identifiers, maps UniProt participants to
existing canonical protein ENSP IDs when a KG node root is supplied, and writes
lossy-free evidence/rejection reports for review before any canonical promotion.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from manage_db.kg_storage import open_kg_root, read_nodes

COMPLEX_PORTAL_HUMAN_URL = "https://ftp.ebi.ac.uk/pub/databases/intact/complex/current/complextab/9606.tsv"
SOURCE = "complex_portal"
SOURCE_DATASET = "complex_portal_complextab_9606"
NODE_TYPE = "protein_complex"
MEMBERSHIP_RELATION = "protein_part_of_protein_complex"
NESTED_RELATION = "protein_complex_part_of_protein_complex"
DISPLAY_MEMBERSHIP = "part of complex"
DISPLAY_NESTED = "complex part of complex"
HUMAN_TAXONOMY_ID = "9606"
HUMAN_ORGANISM_CURIE = "NCBITaxon:9606"
LICENSE_NOTE = "Complex Portal/EMBL-EBI; release-pinned cached complextab, review license before canonical promotion"

COMPLEXTAB_COLUMNS = [
    "#Complex ac",
    "Recommended name",
    "Aliases for complex",
    "Taxonomy identifier",
    "Identifiers (and stoichiometry) of molecules in complex",
    "Evidence Code",
    "Experimental evidence",
    "Go Annotations",
    "Cross references",
    "Description",
    "Complex properties",
    "Complex assembly",
    "Ligand",
    "Disease",
    "Agonist",
    "Antagonist",
    "Comment",
    "Source",
    "Expanded participant list",
]

NODE_COLUMNS = [
    "id",
    "name",
    "description",
    "organism_id",
    "source",
    "source_record_id",
    "complex_portal_id",
    "aliases",
    "stoichiometry_json",
    "assembly",
    "ligand_ids",
    "disease_xrefs",
    "go_annotations",
    "crossrefs",
    "release",
    "license",
]

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
    "source_complex_id",
    "source_complex_name",
    "source_participant_id",
    "source_participant_namespace",
    "mapped_protein_id",
    "mapping_method",
    "mapping_confidence",
    "stoichiometry",
    "participant_role",
    "evidence_code",
    "experimental_evidence",
    "pmids",
    "go_annotations",
    "crossrefs",
    "complex_assembly",
    "release",
    "license",
    "raw_json",
]

REJECTED_COLUMNS = [
    "source_row_number",
    "source_complex_id",
    "source_complex_name",
    "source_participant_id",
    "source_participant_namespace",
    "stoichiometry",
    "reason",
    "candidate_protein_ids",
    "raw_json",
]

PARTICIPANT_RE = re.compile(r"^\s*([^()\s]+)(?:\(([^()]*)\))?\s*$")
PMID_RE = re.compile(r"(?:pubmed:|PMID:)?(\d{4,})", re.IGNORECASE)
COMPLEX_ID_RE = re.compile(r"^(?:complex portal:)?(CPX-\d+)$", re.IGNORECASE)


@dataclass(frozen=True)
class Participant:
    namespace: str
    identifier: str
    stoichiometry: str
    raw: str


@dataclass(frozen=True)
class BuildResult:
    nodes: pd.DataFrame
    edges: pd.DataFrame
    evidence: pd.DataFrame
    nested_edges: pd.DataFrame
    nested_evidence: pd.DataFrame
    rejected: pd.DataFrame
    validation: dict[str, Any]


def _clean(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return "" if text == "-" else text


def _split_pipe(value: Any) -> list[str]:
    text = _clean(value)
    if not text:
        return []
    return [part.strip() for part in text.split("|") if part.strip() and part.strip() != "-"]


def _parse_xref_token(token: str) -> tuple[str, str]:
    token = token.strip()
    if not token:
        return "", ""
    if ":" not in token:
        return "", token.split("(", 1)[0].strip()
    namespace, identifier = token.split(":", 1)
    identifier = identifier.split("(", 1)[0].strip().strip('"')
    return namespace.lower().strip(), identifier


def _normalize_complex_id(identifier: str) -> str:
    match = COMPLEX_ID_RE.match(identifier.strip())
    return match.group(1).upper() if match else ""


def parse_participant_token(token: str) -> Participant | None:
    token = token.strip()
    if not token or token == "-":
        return None
    match = PARTICIPANT_RE.match(token)
    if not match:
        return Participant(namespace="unknown", identifier=token, stoichiometry="", raw=token)
    identifier = match.group(1).strip()
    stoichiometry = (match.group(2) or "").strip()
    namespace, parsed_identifier = _parse_xref_token(identifier)
    identifier = parsed_identifier or identifier
    complex_id = _normalize_complex_id(identifier)
    if complex_id:
        return Participant(namespace="complex_portal", identifier=complex_id, stoichiometry=stoichiometry, raw=token)
    if namespace in {"complex portal", "complex_portal"}:
        complex_id = _normalize_complex_id(identifier)
        if complex_id:
            return Participant(namespace="complex_portal", identifier=complex_id, stoichiometry=stoichiometry, raw=token)
    if not namespace:
        namespace = "uniprotkb" if re.match(r"^[A-Z0-9]{6,10}(?:-\d+)?$", identifier) else "unknown"
    return Participant(namespace=namespace, identifier=identifier, stoichiometry=stoichiometry, raw=token)


def parse_participants(value: Any) -> list[Participant]:
    participants: list[Participant] = []
    for token in _split_pipe(value):
        participant = parse_participant_token(token)
        if participant is not None:
            participants.append(participant)
    return participants


def parse_pmids(*values: Any) -> str:
    """Normalize explicit PubMed identifiers without mining digits from other xrefs."""
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        for token in _split_pipe(value):
            namespace, identifier = _parse_xref_token(token)
            candidates: list[str] = []
            if namespace == "pubmed" and identifier.isdigit():
                candidates = [identifier]
            elif token.upper().startswith("PMID:"):
                maybe = token.split(":", 1)[1].split("(", 1)[0].strip()
                candidates = [maybe] if maybe.isdigit() else []
            elif token.isdigit():
                candidates = [token]
            for candidate in candidates:
                pmid = f"PMID:{candidate}"
                if pmid not in seen:
                    seen.add(pmid)
                    out.append(pmid)
    return "|".join(out)


def _json_default(value: Any) -> Any:
    if pd.isna(value):
        return None
    return value


def raw_json(row: pd.Series) -> str:
    return json.dumps({str(k): _json_default(v) for k, v in row.to_dict().items()}, sort_keys=True, default=str, separators=(",", ":"))


def participant_stoichiometry_json(participants: Iterable[Participant]) -> str:
    entries = [
        {"namespace": p.namespace, "id": p.identifier, "stoichiometry": p.stoichiometry}
        for p in participants
        if p.stoichiometry
    ]
    return json.dumps(entries, sort_keys=True, separators=(",", ":")) if entries else ""


def crossref_value(value: Any, prefix: str) -> str:
    vals = []
    for token in _split_pipe(value):
        namespace, identifier = _parse_xref_token(token)
        if namespace.replace(" ", "_") == prefix:
            vals.append(identifier)
    return "|".join(dict.fromkeys(vals))


def build_uniprot_to_protein_map(node_root: str | None = None, protein_nodes_path: str | Path | None = None) -> tuple[dict[str, str], dict[str, list[str]], dict[str, Any]]:
    if protein_nodes_path:
        proteins = pd.read_parquet(protein_nodes_path, columns=["id", "uniprot_id"])
        source = str(protein_nodes_path)
    elif node_root:
        root = open_kg_root(node_root)
        proteins = read_nodes(root, "protein", columns=["id", "uniprot_id"])
        source = f"{node_root.rstrip('/')}/nodes/protein.parquet"
    else:
        return {}, {}, {"mapping_supplied": False, "source": "", "protein_node_rows": 0, "unique_uniprot_accessions": 0, "ambiguous_uniprot_accessions": 0}

    proteins = proteins.dropna(subset=["uniprot_id", "id"]).copy()
    proteins["uniprot_id"] = proteins["uniprot_id"].astype(str).str.strip()
    proteins["id"] = proteins["id"].astype(str).str.strip()
    grouped = proteins[proteins["uniprot_id"].ne("")].groupby("uniprot_id")["id"].apply(lambda s: sorted(set(s)))
    unique = {acc: ids[0] for acc, ids in grouped.items() if len(ids) == 1}
    ambiguous = {acc: ids for acc, ids in grouped.items() if len(ids) > 1}
    stats = {
        "mapping_supplied": True,
        "source": source,
        "protein_node_rows": int(len(proteins)),
        "unique_uniprot_accessions": int(len(unique)),
        "ambiguous_uniprot_accessions": int(len(ambiguous)),
    }
    return unique, ambiguous, stats


def _edge_key(relation: str, x_id: str, y_id: str) -> str:
    return f"{relation}|{x_id}|{y_id}"


def _empty(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def build_from_complextab(
    complextab: pd.DataFrame,
    *,
    uniprot_to_protein: Mapping[str, str] | None = None,
    ambiguous_uniprot: Mapping[str, list[str]] | None = None,
    release: str = "",
    source_url: str = COMPLEX_PORTAL_HUMAN_URL,
) -> BuildResult:
    uniprot_to_protein = uniprot_to_protein or {}
    ambiguous_uniprot = ambiguous_uniprot or {}
    nodes: list[dict[str, Any]] = []
    edge_records: dict[tuple[str, str], dict[str, Any]] = {}
    evidence_rows: list[dict[str, Any]] = []
    nested_edge_records: dict[tuple[str, str], dict[str, Any]] = {}
    nested_evidence_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    source_counts: Counter[str] = Counter()
    now = datetime.now(timezone.utc).isoformat()

    for idx, row in complextab.iterrows():
        source_row_number = int(idx) + 2  # +1 for 0-index, +1 for header line
        complex_id = _clean(row.get("#Complex ac"))
        name = _clean(row.get("Recommended name")) or complex_id
        taxonomy_id = _clean(row.get("Taxonomy identifier"))
        participants = parse_participants(row.get("Identifiers (and stoichiometry) of molecules in complex"))
        if not complex_id:
            source_counts["rejected_missing_complex_id"] += 1
            continue
        if taxonomy_id != HUMAN_TAXONOMY_ID:
            source_counts["rejected_non_human_complex"] += 1
            continue

        nodes.append(
            {
                "id": complex_id,
                "name": name,
                "description": _clean(row.get("Description")),
                "organism_id": HUMAN_ORGANISM_CURIE,
                "source": SOURCE,
                "source_record_id": complex_id,
                "complex_portal_id": complex_id,
                "aliases": "|".join(_split_pipe(row.get("Aliases for complex"))),
                "stoichiometry_json": participant_stoichiometry_json(participants),
                "assembly": _clean(row.get("Complex assembly")),
                "ligand_ids": "|".join(_split_pipe(row.get("Ligand"))),
                "disease_xrefs": "|".join(_split_pipe(row.get("Disease"))),
                "go_annotations": "|".join(_split_pipe(row.get("Go Annotations"))),
                "crossrefs": "|".join(_split_pipe(row.get("Cross references"))),
                "release": release,
                "license": LICENSE_NOTE,
            }
        )
        source_counts["complex_rows"] += 1

        for participant in participants:
            participant_payload = {
                "source_row_number": source_row_number,
                "source_complex_id": complex_id,
                "source_complex_name": name,
                "source_participant_id": participant.identifier,
                "source_participant_namespace": participant.namespace,
                "stoichiometry": participant.stoichiometry,
                "candidate_protein_ids": "",
                "raw_json": raw_json(row),
            }
            if participant.namespace == "complex_portal":
                if participant.identifier == complex_id:
                    rejected_rows.append({**participant_payload, "reason": "nested_complex_self_reference"})
                    source_counts["rejected_nested_complex_self_reference"] += 1
                    continue
                nested_edge_records[(participant.identifier, complex_id)] = {
                    "x_id": participant.identifier,
                    "x_type": NODE_TYPE,
                    "y_id": complex_id,
                    "y_type": NODE_TYPE,
                    "relation": NESTED_RELATION,
                    "display_relation": DISPLAY_NESTED,
                    "source": SOURCE,
                    "credibility": 3,
                }
                nested_evidence_rows.append(
                    {
                        "edge_key": _edge_key(NESTED_RELATION, participant.identifier, complex_id),
                        "relation": NESTED_RELATION,
                        "x_id": participant.identifier,
                        "x_type": NODE_TYPE,
                        "y_id": complex_id,
                        "y_type": NODE_TYPE,
                        "evidence_type": "database_record",
                        "source": SOURCE,
                        "source_dataset": SOURCE_DATASET,
                        "source_release": release,
                        "source_record_id": f"{complex_id}|{participant.raw}",
                        "source_complex_id": complex_id,
                        "source_complex_name": name,
                        "source_participant_id": participant.identifier,
                        "source_participant_namespace": participant.namespace,
                        "mapped_protein_id": "",
                        "mapping_method": "source_native_complex_portal_id",
                        "mapping_confidence": "explicit_child_complex",
                        "stoichiometry": participant.stoichiometry,
                        "participant_role": "child_complex",
                        "evidence_code": _clean(row.get("Evidence Code")),
                        "experimental_evidence": _clean(row.get("Experimental evidence")),
                        "pmids": parse_pmids(row.get("Cross references"), row.get("Experimental evidence")),
                        "go_annotations": "|".join(_split_pipe(row.get("Go Annotations"))),
                        "crossrefs": "|".join(_split_pipe(row.get("Cross references"))),
                        "complex_assembly": _clean(row.get("Complex assembly")),
                        "release": release,
                        "license": LICENSE_NOTE,
                        "raw_json": raw_json(row),
                    }
                )
                source_counts["nested_complex_participants"] += 1
                continue

            if participant.namespace not in {"uniprotkb", "uniprot", "uniprot/swiss-prot", "uniprot/trembl"}:
                rejected_rows.append({**participant_payload, "reason": "unsupported_participant_namespace"})
                source_counts["rejected_unsupported_participant_namespace"] += 1
                continue

            accession = participant.identifier
            protein_id = uniprot_to_protein.get(accession)
            if not protein_id:
                if accession in ambiguous_uniprot:
                    rejected_rows.append(
                        {
                            **participant_payload,
                            "reason": "uniprot_maps_to_multiple_protein_nodes",
                            "candidate_protein_ids": "|".join(ambiguous_uniprot[accession]),
                        }
                    )
                    source_counts["rejected_uniprot_maps_to_multiple_protein_nodes"] += 1
                else:
                    rejected_rows.append({**participant_payload, "reason": "uniprot_unmapped_to_protein_node"})
                    source_counts["rejected_uniprot_unmapped_to_protein_node"] += 1
                continue

            edge_records[(protein_id, complex_id)] = {
                "x_id": protein_id,
                "x_type": "protein",
                "y_id": complex_id,
                "y_type": NODE_TYPE,
                "relation": MEMBERSHIP_RELATION,
                "display_relation": DISPLAY_MEMBERSHIP,
                "source": SOURCE,
                "credibility": 3,
            }
            evidence_rows.append(
                {
                    "edge_key": _edge_key(MEMBERSHIP_RELATION, protein_id, complex_id),
                    "relation": MEMBERSHIP_RELATION,
                    "x_id": protein_id,
                    "x_type": "protein",
                    "y_id": complex_id,
                    "y_type": NODE_TYPE,
                    "evidence_type": "database_record",
                    "source": SOURCE,
                    "source_dataset": SOURCE_DATASET,
                    "source_release": release,
                    "source_record_id": f"{complex_id}|{participant.raw}",
                    "source_complex_id": complex_id,
                    "source_complex_name": name,
                    "source_participant_id": accession,
                    "source_participant_namespace": participant.namespace,
                    "mapped_protein_id": protein_id,
                    "mapping_method": "nodes/protein.uniprot_id exact unique xref",
                    "mapping_confidence": "exact_unique_uniprot_xref",
                    "stoichiometry": participant.stoichiometry,
                    "participant_role": "component",
                    "evidence_code": _clean(row.get("Evidence Code")),
                    "experimental_evidence": _clean(row.get("Experimental evidence")),
                    "pmids": parse_pmids(row.get("Cross references"), row.get("Experimental evidence")),
                    "go_annotations": "|".join(_split_pipe(row.get("Go Annotations"))),
                    "crossrefs": "|".join(_split_pipe(row.get("Cross references"))),
                    "complex_assembly": _clean(row.get("Complex assembly")),
                    "release": release,
                    "license": LICENSE_NOTE,
                    "raw_json": raw_json(row),
                }
            )
            source_counts["mapped_protein_participants"] += 1

    nodes_df = pd.DataFrame(nodes).drop_duplicates(subset=["id"], keep="last") if nodes else _empty(NODE_COLUMNS)
    nodes_df = _ensure_columns(nodes_df, NODE_COLUMNS)[NODE_COLUMNS]
    edges_df = pd.DataFrame(edge_records.values()) if edge_records else _empty(EDGE_COLUMNS)
    edges_df = _ensure_columns(edges_df, EDGE_COLUMNS)[EDGE_COLUMNS]
    evidence_df = pd.DataFrame(evidence_rows) if evidence_rows else _empty(EVIDENCE_COLUMNS)
    evidence_df = _ensure_columns(evidence_df, EVIDENCE_COLUMNS)[EVIDENCE_COLUMNS]
    nested_edges_df = pd.DataFrame(nested_edge_records.values()) if nested_edge_records else _empty(EDGE_COLUMNS)
    nested_edges_df = _ensure_columns(nested_edges_df, EDGE_COLUMNS)[EDGE_COLUMNS]
    nested_evidence_df = pd.DataFrame(nested_evidence_rows) if nested_evidence_rows else _empty(EVIDENCE_COLUMNS)
    nested_evidence_df = _ensure_columns(nested_evidence_df, EVIDENCE_COLUMNS)[EVIDENCE_COLUMNS]
    rejected_df = pd.DataFrame(rejected_rows) if rejected_rows else _empty(REJECTED_COLUMNS)
    rejected_df = _ensure_columns(rejected_df, REJECTED_COLUMNS)[REJECTED_COLUMNS]

    validation = validate_outputs(
        nodes_df,
        edges_df,
        evidence_df,
        nested_edges_df,
        nested_evidence_df,
        rejected_df,
        dict(source_counts),
        release=release,
        source_url=source_url,
        created_at=now,
        mapping_supplied=bool(uniprot_to_protein or ambiguous_uniprot),
    )
    return BuildResult(nodes_df, edges_df, evidence_df, nested_edges_df, nested_evidence_df, rejected_df, validation)


def _ensure_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            out[col] = pd.NA
    return out


def validate_outputs(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    evidence: pd.DataFrame,
    nested_edges: pd.DataFrame,
    nested_evidence: pd.DataFrame,
    rejected: pd.DataFrame,
    source_counts: dict[str, int],
    *,
    release: str,
    source_url: str,
    created_at: str,
    mapping_supplied: bool,
) -> dict[str, Any]:
    validation: dict[str, Any] = {
        "ok": True,
        "created_at": created_at,
        "staging_only": True,
        "canonical_promotion": False,
        "source": SOURCE,
        "source_url": source_url,
        "source_release": release,
        "counts": {
            "protein_complex_nodes": int(len(nodes)),
            "protein_part_of_protein_complex_edges": int(len(edges)),
            "protein_part_of_protein_complex_evidence": int(len(evidence)),
            "nested_complex_edges": int(len(nested_edges)),
            "nested_complex_evidence": int(len(nested_evidence)),
            "rejected_participants": int(len(rejected)),
        },
        "source_counts": dict(sorted(source_counts.items())),
        "checks": {},
        "warnings": [],
        "canonical_promotion_recommendation": "Do not promote canonically yet. Review UniProt→ENSP ambiguity rejects, add schema NodeType/Relation definitions, and approve relation naming before promotion.",
    }
    checks = validation["checks"]

    duplicate_nodes = int(nodes.duplicated(subset=["id"]).sum()) if not nodes.empty else 0
    checks["node_ids_unique"] = {"ok": duplicate_nodes == 0, "duplicate_rows": duplicate_nodes}
    validation["ok"] = validation["ok"] and duplicate_nodes == 0

    node_ids = set(nodes["id"].astype(str)) if not nodes.empty else set()
    edge_complex_missing = sorted(set(edges["y_id"].astype(str)) - node_ids) if not edges.empty else []
    checks["membership_complex_endpoint_antijoin"] = {"ok": not edge_complex_missing, "missing_complex_ids": edge_complex_missing[:20], "missing_count": len(edge_complex_missing)}
    validation["ok"] = validation["ok"] and not edge_complex_missing

    if not edges.empty:
        duplicate_edges = int(edges.duplicated(subset=["relation", "x_id", "y_id"]).sum())
        edge_keys = set(edges.apply(lambda r: _edge_key(str(r["relation"]), str(r["x_id"]), str(r["y_id"])), axis=1))
    else:
        duplicate_edges = 0
        edge_keys = set()
    evidence_keys = set(evidence["edge_key"].astype(str)) if not evidence.empty else set()
    unsupported_edges = sorted(edge_keys - evidence_keys)
    evidence_without_edge = sorted(evidence_keys - edge_keys)
    checks["membership_edges_unique"] = {"ok": duplicate_edges == 0, "duplicate_rows": duplicate_edges}
    checks["membership_edge_evidence_support"] = {
        "ok": not unsupported_edges and not evidence_without_edge,
        "edges_without_evidence": unsupported_edges[:20],
        "evidence_without_edge": evidence_without_edge[:20],
        "edges_without_evidence_count": len(unsupported_edges),
        "evidence_without_edge_count": len(evidence_without_edge),
    }
    validation["ok"] = validation["ok"] and duplicate_edges == 0 and not unsupported_edges and not evidence_without_edge

    if not nested_edges.empty:
        duplicate_nested = int(nested_edges.duplicated(subset=["relation", "x_id", "y_id"]).sum())
        nested_edge_keys = set(nested_edges.apply(lambda r: _edge_key(str(r["relation"]), str(r["x_id"]), str(r["y_id"])), axis=1))
        nested_missing = sorted((set(nested_edges["x_id"].astype(str)) | set(nested_edges["y_id"].astype(str))) - node_ids)
    else:
        duplicate_nested = 0
        nested_edge_keys = set()
        nested_missing = []
    nested_evidence_keys = set(nested_evidence["edge_key"].astype(str)) if not nested_evidence.empty else set()
    checks["nested_edges_explicit_only"] = {"ok": True, "note": "nested edges are emitted only from participant tokens whose parsed identifier is CPX-*"}
    checks["nested_complex_endpoint_antijoin"] = {"ok": not nested_missing, "missing_complex_ids": nested_missing[:20], "missing_count": len(nested_missing)}
    checks["nested_edge_evidence_support"] = {
        "ok": not (nested_edge_keys - nested_evidence_keys) and not (nested_evidence_keys - nested_edge_keys),
        "edges_without_evidence_count": len(nested_edge_keys - nested_evidence_keys),
        "evidence_without_edge_count": len(nested_evidence_keys - nested_edge_keys),
    }
    validation["ok"] = validation["ok"] and duplicate_nested == 0 and not nested_missing and checks["nested_edge_evidence_support"]["ok"]

    checks["protein_endpoint_mapping"] = {
        "ok": mapping_supplied,
        "mapping_supplied": mapping_supplied,
        "policy": "accepted only source UniProt accessions with exactly one nodes/protein.uniprot_id match; ambiguous and unmapped participants are materialized as rejects",
        "rejected_reason_counts": rejected["reason"].value_counts().to_dict() if not rejected.empty else {},
    }
    validation["ok"] = validation["ok"] and mapping_supplied
    if not mapping_supplied:
        validation["warnings"].append("No protein node mapping supplied; production staging must pass --node-root or --protein-nodes.")

    checks["no_member_disease_projection"] = {"ok": True, "note": "Disease field is preserved only as protein_complex node raw disease_xrefs; no disease/phenotype edges are emitted."}
    return validation


def _write_parquet(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.reset_index(drop=True).to_parquet(path, index=False)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def read_complextab(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def _release_from_headers(headers: Mapping[str, str]) -> str:
    last_modified = headers.get("Last-Modified") or headers.get("last-modified") or ""
    if last_modified:
        try:
            return parsedate_to_datetime(last_modified).date().isoformat()
        except Exception:
            return last_modified
    etag = (headers.get("ETag") or headers.get("etag") or "").strip('"')
    return f"etag-{etag}" if etag else date.today().isoformat()


def cache_complex_portal_human(url: str = COMPLEX_PORTAL_HUMAN_URL, raw_cache_dir: str | Path = "artifacts/cache/raw/complex_portal") -> tuple[Path, dict[str, Any]]:
    raw_cache = Path(raw_cache_dir)
    raw_cache.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request, timeout=60) as response:
        headers = dict(response.headers.items())
        release = _release_from_headers(headers)
    release_dir = raw_cache / release
    release_dir.mkdir(parents=True, exist_ok=True)
    target = release_dir / "9606.tsv"
    if not target.exists():
        with urllib.request.urlopen(url, timeout=180) as response, target.open("wb") as out:
            shutil.copyfileobj(response, out)
    manifest = {
        "source": SOURCE,
        "url": url,
        "cached_path": str(target),
        "release": release,
        "headers": {k: headers.get(k, "") for k in ["Last-Modified", "ETag", "Content-Length", "Content-Type"]},
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(release_dir / "9606.manifest.json", manifest)
    return target, manifest


def build_staged_complex_portal(
    *,
    input_path: str | Path | None = None,
    node_root: str | None = None,
    protein_nodes_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    raw_cache_dir: str | Path = "artifacts/cache/raw/complex_portal",
    source_url: str = COMPLEX_PORTAL_HUMAN_URL,
) -> dict[str, Any]:
    if input_path is None:
        input_file, manifest = cache_complex_portal_human(source_url, raw_cache_dir)
    else:
        input_file = Path(input_path)
        manifest = {"source": SOURCE, "url": source_url, "cached_path": str(input_file), "release": input_file.parent.name if input_file.parent.name else "manual"}
    release = str(manifest.get("release") or "")
    out_dir = Path(output_dir) if output_dir else Path("artifacts/staged") / f"complex-portal-protein-complexes-{date.today().isoformat()}"
    complextab = read_complextab(input_file)
    uniprot_to_protein, ambiguous_uniprot, mapping_stats = build_uniprot_to_protein_map(node_root=node_root, protein_nodes_path=protein_nodes_path)
    result = build_from_complextab(
        complextab,
        uniprot_to_protein=uniprot_to_protein,
        ambiguous_uniprot=ambiguous_uniprot,
        release=release,
        source_url=source_url,
    )

    _write_parquet(out_dir / "nodes" / "protein_complex.parquet", result.nodes)
    _write_parquet(out_dir / "edges" / f"{MEMBERSHIP_RELATION}.parquet", result.edges)
    _write_parquet(out_dir / "evidence" / f"{MEMBERSHIP_RELATION}.parquet", result.evidence)
    _write_parquet(out_dir / "edges" / f"{NESTED_RELATION}.parquet", result.nested_edges)
    _write_parquet(out_dir / "evidence" / f"{NESTED_RELATION}.parquet", result.nested_evidence)
    _write_parquet(out_dir / "mappings" / "complex_portal_participants_rejected.parquet", result.rejected)
    report = {
        **result.validation,
        "output_dir": str(out_dir),
        "inputs": {"complextab": str(input_file), "node_root": node_root or "", "protein_nodes": str(protein_nodes_path or ""), "manifest": manifest},
        "mapping_stats": mapping_stats,
        "artifacts": {
            "nodes": str(out_dir / "nodes" / "protein_complex.parquet"),
            "membership_edges": str(out_dir / "edges" / f"{MEMBERSHIP_RELATION}.parquet"),
            "membership_evidence": str(out_dir / "evidence" / f"{MEMBERSHIP_RELATION}.parquet"),
            "nested_edges": str(out_dir / "edges" / f"{NESTED_RELATION}.parquet"),
            "nested_evidence": str(out_dir / "evidence" / f"{NESTED_RELATION}.parquet"),
            "rejected_participants": str(out_dir / "mappings" / "complex_portal_participants_rejected.parquet"),
        },
    }
    _write_json(out_dir / "validation" / "complex_portal_build_report.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=None, help="Optional cached Complex Portal human complextab 9606.tsv; downloads current release if omitted")
    parser.add_argument("--raw-cache-dir", default="artifacts/cache/raw/complex_portal", help="Repo-local raw cache for release-pinned complextab")
    parser.add_argument("--node-root", default="", help="KG root containing nodes/protein.parquet for UniProt→ENSP mapping, e.g. gs://jouvencekb/main")
    parser.add_argument("--protein-nodes", default=None, help="Optional local protein.parquet for tests/offline builds")
    parser.add_argument("--output-dir", default=None, help="Defaults to artifacts/staged/complex-portal-protein-complexes-YYYY-MM-DD")
    parser.add_argument("--source-url", default=COMPLEX_PORTAL_HUMAN_URL)
    args = parser.parse_args(argv)
    report = build_staged_complex_portal(
        input_path=args.input,
        node_root=args.node_root or None,
        protein_nodes_path=args.protein_nodes,
        output_dir=args.output_dir,
        raw_cache_dir=args.raw_cache_dir,
        source_url=args.source_url,
    )
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0 if report.get("ok") else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
