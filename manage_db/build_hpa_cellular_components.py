"""Build staged HPA cellular_component nodes and protein localization edges.

This module is intentionally staging-only. It consumes a release-pinned Human
Protein Atlas ``proteinatlas.tsv.zip`` plus optional GO/UniProt-SL vocabularies,
resolves HPA UniProt accessions to existing ``nodes/protein.parquet`` ENSP node
IDs, and writes staged artifacts under an output directory:

- ``nodes/cellular_component.parquet``
- ``edges/protein_located_in_cellular_component.parquet``
- ``evidence/protein_located_in_cellular_component.parquet``
- optional ``edges/cellular_component_subtype_of_cellular_component.parquet``
- validation/manifest JSON reports

No canonical KG files are modified.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import urllib.request
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

PROTEIN_LOCATION_RELATION = "protein_located_in_cellular_component"
HIERARCHY_RELATION = "cellular_component_subtype_of_cellular_component"
X_TYPE = "protein"
COMPONENT_TYPE = "cellular_component"
DISPLAY_RELATION = "located in"
HIERARCHY_DISPLAY_RELATION = "subtype of"
HPA_DATASET = "proteinatlas.tsv"
HPA_URL = "https://www.proteinatlas.org/download/proteinatlas.tsv.zip"
GO_BASIC_URL = "https://purl.obolibrary.org/obo/go/go-basic.obo"
UNIPROT_SUBCELL_URL = "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/docs/subcell.txt"

HPA_FIELDS = [
    "Subcellular main location",
    "Subcellular additional location",
    "Subcellular location",
    "Secretome location",
]
FIELD_CATEGORY = {
    "Subcellular main location": "main_location",
    "Subcellular additional location": "additional_location",
    "Subcellular location": "subcellular_location",
    "Secretome location": "secretome_location",
}

EDGE_COLUMNS = ["x_id", "x_type", "y_id", "y_type", "relation", "display_relation", "source", "credibility"]
COMPONENT_NODE_COLUMNS = [
    "id",
    "name",
    "display_category",
    "go_id",
    "uniprot_sl_id",
    "hpa_label",
    "hpa_category",
    "description",
    "source",
    "source_release",
    "mapping_confidence",
    "mapping_method",
    "raw_labels_json",
]
LOCATION_EVIDENCE_COLUMNS = [
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
    "hpa_gene",
    "ensembl_gene_id",
    "uniprot_id",
    "hpa_label",
    "hpa_category",
    "hpa_field",
    "hpa_reliability_if",
    "hpa_reliability_ih",
    "hpa_evidence",
    "mapping_confidence",
    "mapping_method",
    "raw_labels_json",
    "predicate",
    "license",
    "release",
    "created_at",
]
HIERARCHY_EVIDENCE_COLUMNS = [
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
    "predicate",
    "mapping_confidence",
    "mapping_method",
    "raw_labels_json",
    "license",
    "release",
    "created_at",
]

# Explicit reviewed HPA label mappings for labels whose casing/plurals or HPA
# display wording do not map cleanly by exact GO/UniProt-SL name lookup. GO IDs
# are used for node IDs when available; HPA-only labels receive HPA_SL fallback
# IDs. Keep confidence conservative for broad/secretome buckets.
HPA_LABEL_OVERRIDES: dict[str, dict[str, str]] = {
    "Actin filaments": {"go_id": "GO:0015629", "confidence": "exact_manual", "category": "cytoskeleton"},
    "Aggresome": {"go_id": "GO:0016235", "confidence": "exact_manual", "category": "organelle"},
    "Basal body": {"go_id": "GO:0036064", "confidence": "exact_manual", "category": "cilium"},
    "Cell Junctions": {"go_id": "GO:0030054", "confidence": "exact_manual", "category": "junction"},
    "Centriolar satellite": {"go_id": "GO:0034451", "confidence": "exact_manual", "category": "centrosome"},
    "Centrosome": {"go_id": "GO:0005813", "confidence": "exact_manual", "category": "centrosome"},
    "Cleavage furrow": {"go_id": "GO:0032154", "confidence": "exact_manual", "category": "cell_cycle_structure"},
    "Cytokinetic bridge": {"go_id": "GO:0045171", "confidence": "exact_manual", "category": "cell_cycle_structure"},
    "Cytoplasmic bodies": {"go_id": "GO:0036464", "confidence": "broad_manual", "category": "cytoplasmic_body"},
    "Cytosol": {"go_id": "GO:0005829", "confidence": "exact_manual", "category": "cytosol"},
    "Endoplasmic reticulum": {"go_id": "GO:0005783", "confidence": "exact_manual", "category": "organelle"},
    "Endosomes": {"go_id": "GO:0005768", "confidence": "exact_manual", "category": "organelle"},
    "Focal adhesion sites": {"go_id": "GO:0005925", "confidence": "exact_manual", "category": "junction"},
    "Golgi apparatus": {"go_id": "GO:0005794", "confidence": "exact_manual", "category": "organelle"},
    "Intermediate filaments": {"go_id": "GO:0005882", "confidence": "exact_manual", "category": "cytoskeleton"},
    "Kinetochore": {"go_id": "GO:0000776", "confidence": "exact_manual", "category": "chromosome"},
    "Lipid droplets": {"go_id": "GO:0005811", "confidence": "exact_manual", "category": "organelle"},
    "Lysosomes": {"go_id": "GO:0005764", "confidence": "exact_manual", "category": "organelle"},
    "Microtubule ends": {"go_id": "GO:1990752", "confidence": "exact_manual", "category": "cytoskeleton"},
    "Microtubules": {"go_id": "GO:0005874", "confidence": "exact_manual", "category": "cytoskeleton"},
    "Midbody": {"go_id": "GO:0030496", "confidence": "exact_manual", "category": "cell_cycle_structure"},
    "Midbody ring": {"go_id": "GO:0090543", "confidence": "exact_manual", "category": "cell_cycle_structure"},
    "Mitotic chromosome": {"confidence": "hpa_local", "category": "chromosome"},
    "Mitotic spindle": {"go_id": "GO:0072686", "confidence": "exact_manual", "category": "cell_cycle_structure"},
    "Mitochondria": {"go_id": "GO:0005739", "confidence": "exact_manual", "category": "organelle"},
    "Nuclear bodies": {"go_id": "GO:0016604", "confidence": "exact_manual", "category": "nucleus"},
    "Nuclear membrane": {"go_id": "GO:0031965", "confidence": "exact_manual", "category": "nucleus"},
    "Nuclear speckles": {"go_id": "GO:0016607", "confidence": "exact_manual", "category": "nucleus"},
    "Nucleoli": {"go_id": "GO:0005730", "confidence": "exact_manual", "category": "nucleus"},
    "Nucleoli fibrillar center": {"go_id": "GO:0001650", "confidence": "exact_manual", "category": "nucleus"},
    "Nucleoli rim": {"confidence": "hpa_local", "category": "nucleus"},
    "Nucleoplasm": {"go_id": "GO:0005654", "confidence": "exact_manual", "category": "nucleus"},
    "Peroxisomes": {"go_id": "GO:0005777", "confidence": "exact_manual", "category": "organelle"},
    "Plasma membrane": {"go_id": "GO:0005886", "confidence": "exact_manual", "category": "membrane"},
    "Primary cilium": {"confidence": "hpa_local", "category": "cilium"},
    "Primary cilium tip": {"go_id": "GO:0097542", "confidence": "exact_manual", "category": "cilium"},
    "Primary cilium transition zone": {"go_id": "GO:0035869", "confidence": "exact_manual", "category": "cilium"},
    "Rods & Rings": {"confidence": "hpa_local", "category": "cytoplasmic_body"},
    "Vesicles": {"go_id": "GO:0031982", "confidence": "exact_manual", "category": "organelle"},
    # Sperm/acrosome labels are HPA-specific/too specific for confident GO IDs in this pilot.
    "Acrosome": {"go_id": "GO:0001669", "confidence": "exact_manual", "category": "sperm_structure"},
    "Annulus": {"confidence": "hpa_local", "category": "sperm_structure"},
    "Calyx": {"confidence": "hpa_local", "category": "sperm_structure"},
    "Connecting piece": {"confidence": "hpa_local", "category": "sperm_structure"},
    "End piece": {"confidence": "hpa_local", "category": "sperm_structure"},
    "Equatorial segment": {"confidence": "hpa_local", "category": "sperm_structure"},
    "Flagellar centriole": {"confidence": "hpa_local", "category": "sperm_structure"},
    "Mid piece": {"go_id": "GO:0097225", "confidence": "exact_manual", "category": "sperm_structure"},
    "Perinuclear theca": {"confidence": "hpa_local", "category": "sperm_structure"},
    "Principal piece": {"go_id": "GO:0097228", "confidence": "exact_manual", "category": "sperm_structure"},
    # Secretome buckets are not cellular components in GO; keep HPA-local.
    "Secreted to blood": {"confidence": "hpa_secretome_local", "category": "secreted"},
    "Secreted in other tissues": {"confidence": "hpa_secretome_local", "category": "secreted"},
    "Secreted to extracellular matrix": {"go_id": "GO:0031012", "confidence": "broad_manual", "category": "secreted"},
    "Secreted in male reproductive system": {"confidence": "hpa_secretome_local", "category": "secreted"},
    "Secreted - unknown location": {"confidence": "hpa_secretome_local", "category": "secreted"},
    "Secreted to digestive system": {"confidence": "hpa_secretome_local", "category": "secreted"},
    "Secreted in brain": {"confidence": "hpa_secretome_local", "category": "secreted"},
    "Secreted in female reproductive system": {"confidence": "hpa_secretome_local", "category": "secreted"},
    "Intracellular and membrane": {"confidence": "hpa_display_bucket", "category": "display_bucket"},
    "Immunoglobulin genes": {"confidence": "hpa_display_bucket", "category": "display_bucket"},
}

HPA_PARENT_LABELS: dict[str, list[tuple[str, str]]] = {
    "Nucleoplasm": [("Nucleus", "located_in")],
    "Nucleoli": [("Nucleus", "part_of")],
    "Nucleoli fibrillar center": [("Nucleoli", "part_of")],
    "Nucleoli rim": [("Nucleoli", "part_of")],
    "Nuclear bodies": [("Nucleus", "part_of")],
    "Nuclear speckles": [("Nuclear bodies", "subtype_of")],
    "Nuclear membrane": [("Nucleus", "part_of")],
    "Centriolar satellite": [("Centrosome", "located_in")],
    "Basal body": [("Centrosome", "subtype_of")],
    "Primary cilium tip": [("Primary cilium", "part_of")],
    "Primary cilium transition zone": [("Primary cilium", "part_of")],
    "Microtubule ends": [("Microtubules", "part_of")],
    "Midbody ring": [("Midbody", "part_of")],
    "Focal adhesion sites": [("Cell Junctions", "subtype_of")],
    "Connecting piece": [("Primary cilium", "broad_sperm_flagellum_context")],
    "Mid piece": [("Primary cilium", "broad_sperm_flagellum_context")],
    "Principal piece": [("Primary cilium", "broad_sperm_flagellum_context")],
}

EXTRA_COMPONENTS = {
    "Nucleus": {"go_id": "GO:0005634", "confidence": "exact_manual", "category": "nucleus"},
}

# Reviewer-rejected HPA→GO IDs from the staged pilot. Some are valid GO terms
# but semantically wrong/too broad for the HPA label, so namespace/obsolete
# checks alone cannot catch reintroduction.
REJECTED_HPA_GO_MAPPINGS: dict[str, str] = {
    "GO:0030692": "wrong for HPA Nucleoli rim; term is Noc4p-Nop14p complex",
    "GO:0110165": "too broad for HPA Rods & Rings; term is cellular anatomical entity",
}


def clean(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def split_labels(value: Any) -> list[str]:
    text = clean(value)
    if not text:
        return []
    return [part.strip() for part in text.split(",") if part.strip()]


def slugify(label: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", label.strip().lower()).strip("_")
    return slug or "unknown"


def hpa_local_id(label: str) -> str:
    return f"HPA_SL:{slugify(label)}"


def _write_parquet(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.reset_index(drop=True).to_parquet(path, index=False)


def _ensure_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            out[col] = pd.NA
    return out[columns]


def _download_if_missing(path: Path, url: str) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with urllib.request.urlopen(url, timeout=120) as resp, tmp.open("wb") as out:
        shutil.copyfileobj(resp, out)
    tmp.replace(path)


def parse_go_obo(path: Path) -> tuple[dict[str, dict[str, Any]], str]:
    terms: dict[str, dict[str, Any]] = {}
    data_version = ""
    current: dict[str, Any] | None = None
    for raw_line in path.read_text(errors="ignore").splitlines():
        line = raw_line.rstrip("\n")
        if line.startswith("data-version:"):
            data_version = line.split(":", 1)[1].strip()
        if line == "[Term]":
            if current and current.get("id"):
                terms[current["id"]] = current
            current = {"synonyms": [], "is_a": [], "part_of": [], "namespace": "", "is_obsolete": False}
            continue
        if line.startswith("["):
            if current and current.get("id"):
                terms[current["id"]] = current
            current = None
            continue
        if current is None or not line:
            continue
        if line.startswith("id: "):
            current["id"] = line[4:].strip()
        elif line.startswith("name: "):
            current["name"] = line[6:].strip()
        elif line.startswith("namespace: "):
            current["namespace"] = line.split(":", 1)[1].strip()
        elif line.startswith("is_obsolete: "):
            current["is_obsolete"] = line.split(":", 1)[1].strip().lower() == "true"
        elif line.startswith("synonym: "):
            m = re.match(r'synonym: "(.+?)"', line)
            if m:
                current.setdefault("synonyms", []).append(m.group(1))
        elif line.startswith("is_a: "):
            current.setdefault("is_a", []).append(line.split()[1])
        elif line.startswith("relationship: part_of "):
            current.setdefault("part_of", []).append(line.split()[2])
    if current and current.get("id"):
        terms[current["id"]] = current
    return terms, data_version


def parse_uniprot_subcell(path: Path) -> tuple[dict[str, str], str]:
    mapping: dict[str, str] = {}
    release = "current"
    entry_id = ""
    accession = ""
    name = ""
    for raw in path.read_text(errors="ignore").splitlines():
        line = raw.rstrip("\n")
        if line.startswith("Release "):
            release = line.strip()
        if line.startswith("ID   "):
            entry_id = line[5:].strip().rstrip(".")
            accession = ""
            name = ""
        elif line.startswith("AC   "):
            accession = line[5:].strip().rstrip(";")
        elif line.startswith("DE   "):
            text = line[5:].strip().rstrip(".")
            if text and not name:
                name = text
        elif line == "//":
            if accession:
                for key in {entry_id, name}:
                    if key:
                        mapping[key.lower()] = accession
            entry_id = accession = name = ""
    return mapping, release


@dataclass(frozen=True)
class ComponentMapping:
    id: str
    name: str
    display_category: str
    go_id: str
    uniprot_sl_id: str
    hpa_label: str
    hpa_category: str
    description: str
    source: str
    source_release: str
    mapping_confidence: str
    mapping_method: str
    raw_labels_json: str


def build_component_mapping(
    labels: Iterable[str],
    *,
    go_terms: dict[str, dict[str, Any]] | None = None,
    uniprot_sl: dict[str, str] | None = None,
    hpa_release: str = "HPA 25.1",
) -> dict[str, ComponentMapping]:
    go_terms = go_terms or {}
    uniprot_sl = uniprot_sl or {}
    by_name: dict[str, str] = {}
    for go_id, term in go_terms.items():
        if term.get("namespace") != "cellular_component" or term.get("is_obsolete"):
            continue
        if str(term.get("name", "")).lower().startswith("obsolete"):
            continue
        for key in [term.get("name", ""), *term.get("synonyms", [])]:
            if key:
                by_name.setdefault(key.lower(), go_id)
    all_labels = set(labels) | set(EXTRA_COMPONENTS)
    out: dict[str, ComponentMapping] = {}
    for label in sorted(all_labels):
        override = {**EXTRA_COMPONENTS.get(label, {}), **HPA_LABEL_OVERRIDES.get(label, {})}
        reviewed_go_id = override.get("go_id", "")
        # Labels present in reviewed override tables without explicit GO IDs are
        # intentionally HPA-local; do not silently promote them through GO
        # exact/synonym lookup while retaining hpa_local confidence.
        allow_auto_go_lookup = label not in HPA_LABEL_OVERRIDES and label not in EXTRA_COMPONENTS
        go_id = reviewed_go_id or (by_name.get(label.lower(), "") if allow_auto_go_lookup else "")
        uniprot_sl_id = override.get("uniprot_sl_id", "") or uniprot_sl.get(label.lower(), "")
        if go_id:
            component_id = go_id
            method = "manual_hpa_go_override" if label in HPA_LABEL_OVERRIDES or label in EXTRA_COMPONENTS else "go_exact_name_or_synonym"
            confidence = override.get("confidence", "exact_go_label")
            name = go_terms.get(go_id, {}).get("name", label)
            source = "GO/HPA"
        elif uniprot_sl_id:
            component_id = f"UniProtSL:{uniprot_sl_id}"
            method = "uniprot_sl_exact_name"
            confidence = override.get("confidence", "exact_uniprot_sl_label")
            name = label
            source = "UniProt/HPA"
        else:
            component_id = hpa_local_id(label)
            method = "hpa_local_fallback"
            confidence = override.get("confidence", "hpa_local")
            name = label
            source = "HPA"
        category = override.get("category", "secreted" if label.startswith("Secreted") else "hpa_label")
        description = go_terms.get(go_id, {}).get("name", label) if go_id else f"HPA subcellular/secretome label: {label}"
        out[label] = ComponentMapping(
            id=component_id,
            name=name,
            display_category=category,
            go_id=go_id,
            uniprot_sl_id=uniprot_sl_id,
            hpa_label=label,
            hpa_category=category,
            description=description,
            source=source,
            source_release=hpa_release,
            mapping_confidence=confidence,
            mapping_method=method,
            raw_labels_json=json.dumps({"hpa_label": label}, sort_keys=True, separators=(",", ":")),
        )
    return out


def load_uniprot_to_protein(protein_nodes: Path) -> tuple[dict[str, list[str]], dict[str, int]]:
    df = pd.read_parquet(protein_nodes, columns=["id", "uniprot_id"])
    mapping: dict[str, set[str]] = defaultdict(set)
    for row in df.itertuples(index=False):
        uniprot_id = clean(row.uniprot_id)
        protein_id = clean(row.id)
        if uniprot_id and protein_id:
            for token in re.split(r"[;,|]", uniprot_id):
                token = token.strip()
                if token:
                    mapping[token].add(protein_id)
    final = {key: sorted(values) for key, values in mapping.items()}
    stats = {
        "protein_node_rows": int(len(df)),
        "uniprot_accessions": int(len(final)),
        "ambiguous_uniprot_accessions": int(sum(1 for values in final.values() if len(values) > 1)),
    }
    return final, stats


def _credibility(reliability_if: str, field: str, mapping_confidence: str) -> int:
    rel = reliability_if.lower()
    if mapping_confidence.startswith("hpa_display_bucket"):
        return 1
    if rel == "enhanced":
        return 4
    if rel in {"supported", "approved"}:
        return 3
    if field == "Secretome location":
        return 2
    return 2 if mapping_confidence.startswith(("exact", "broad")) else 1


def build_artifacts(
    hpa_zip: Path,
    protein_nodes: Path,
    output_dir: Path,
    *,
    go_obo: Path | None = None,
    uniprot_subcell: Path | None = None,
    hpa_release: str = "HPA 25.1",
    created_at: str | None = None,
) -> dict[str, Any]:
    created_at = created_at or datetime.now(timezone.utc).date().isoformat()
    go_terms: dict[str, dict[str, Any]] = {}
    go_release = ""
    if go_obo and go_obo.exists():
        go_terms, go_release = parse_go_obo(go_obo)
    uniprot_sl: dict[str, str] = {}
    uniprot_release = ""
    if uniprot_subcell and uniprot_subcell.exists():
        uniprot_sl, uniprot_release = parse_uniprot_subcell(uniprot_subcell)

    uniprot_to_proteins, protein_stats = load_uniprot_to_protein(protein_nodes)

    raw_rows: list[dict[str, Any]] = []
    label_counts: Counter[str] = Counter()
    hpa_rows = 0
    rows_with_any_label = 0
    rows_without_protein_mapping = 0
    unmapped_uniprot_rows: Counter[str] = Counter()
    ambiguous_hpa_source_rows = 0
    ambiguous_hpa_label_assignments = 0
    ambiguous_hpa_expanded_protein_rows = 0
    ambiguous_hpa_uniprots: Counter[str] = Counter()
    with zipfile.ZipFile(hpa_zip) as z, z.open(HPA_DATASET) as f:
        reader = csv.DictReader((line.decode("utf-8") for line in f), delimiter="\t")
        for row_idx, row in enumerate(reader, start=2):
            hpa_rows += 1
            labels_by_field = {field: split_labels(row.get(field)) for field in HPA_FIELDS}
            all_labels = sorted({label for labels in labels_by_field.values() for label in labels})
            if not all_labels:
                continue
            rows_with_any_label += 1
            uniprots = [u.strip() for u in re.split(r"[;,]", clean(row.get("Uniprot"))) if u.strip()]
            protein_pairs: list[tuple[str, str]] = []
            ambiguous_uniprots_in_row = {uniprot for uniprot in uniprots if len(uniprot_to_proteins.get(uniprot, [])) > 1}
            for uniprot in uniprots:
                for protein_id in uniprot_to_proteins.get(uniprot, []):
                    protein_pairs.append((uniprot, protein_id))
            if not protein_pairs:
                rows_without_protein_mapping += 1
                for uniprot in uniprots or ["<missing_uniprot>"]:
                    unmapped_uniprot_rows[uniprot] += 1
                continue
            raw_labels_json = json.dumps(labels_by_field, sort_keys=True, separators=(",", ":"))
            if ambiguous_uniprots_in_row:
                ambiguous_hpa_source_rows += 1
                row_label_assignments = sum(len(labels) for labels in labels_by_field.values())
                for uniprot in sorted(ambiguous_uniprots_in_row):
                    ambiguous_hpa_uniprots[uniprot] += 1
                    ambiguous_hpa_label_assignments += row_label_assignments
                    ambiguous_hpa_expanded_protein_rows += row_label_assignments * len(uniprot_to_proteins[uniprot])
            for field, labels in labels_by_field.items():
                for label in labels:
                    label_counts[label] += 1
                    for uniprot, protein_id in protein_pairs:
                        raw_rows.append(
                            {
                                "protein_id": protein_id,
                                "uniprot_id": uniprot,
                                "ensembl_gene_id": clean(row.get("Ensembl")),
                                "hpa_gene": clean(row.get("Gene")),
                                "hpa_label": label,
                                "hpa_field": field,
                                "hpa_category": FIELD_CATEGORY[field],
                                "hpa_reliability_if": clean(row.get("Reliability (IF)")),
                                "hpa_reliability_ih": clean(row.get("Reliability (IH)")),
                                "hpa_evidence": clean(row.get("HPA evidence")) or clean(row.get("Evidence")),
                                "uniprot_mapping_ambiguous": uniprot in ambiguous_uniprots_in_row,
                                "source_record_id": f"HPA:25.1:proteinatlas:{row_idx}:{clean(row.get('Ensembl'))}:{uniprot}:{slugify(label)}:{slugify(field)}",
                                "raw_labels_json": raw_labels_json,
                            }
                        )
    component_map = build_component_mapping(label_counts, go_terms=go_terms, uniprot_sl=uniprot_sl, hpa_release=hpa_release)
    raw_df = pd.DataFrame(raw_rows)
    if raw_df.empty:
        located = pd.DataFrame(columns=EDGE_COLUMNS)
        evidence = pd.DataFrame(columns=LOCATION_EVIDENCE_COLUMNS)
    else:
        raw_df["component_id"] = raw_df["hpa_label"].map(lambda label: component_map[label].id)
        raw_df["mapping_confidence"] = raw_df["hpa_label"].map(lambda label: component_map[label].mapping_confidence)
        raw_df["mapping_method"] = raw_df["hpa_label"].map(lambda label: component_map[label].mapping_method)
        raw_df["credibility"] = raw_df.apply(
            lambda row: _credibility(row["hpa_reliability_if"], row["hpa_field"], row["mapping_confidence"]), axis=1
        )
        located = (
            raw_df.groupby(["protein_id", "component_id"], as_index=False)
            .agg({"credibility": "max"})
            .rename(columns={"protein_id": "x_id", "component_id": "y_id"})
        )
        located["x_type"] = X_TYPE
        located["y_type"] = COMPONENT_TYPE
        located["relation"] = PROTEIN_LOCATION_RELATION
        located["display_relation"] = DISPLAY_RELATION
        located["source"] = "HPA"
        located = located[EDGE_COLUMNS].sort_values(["x_id", "y_id"]).reset_index(drop=True)
        evidence = raw_df.copy()
        evidence["edge_key"] = evidence.apply(
            lambda row: f"{PROTEIN_LOCATION_RELATION}|{row['protein_id']}|{row['component_id']}", axis=1
        )
        evidence["relation"] = PROTEIN_LOCATION_RELATION
        evidence["x_id"] = evidence["protein_id"]
        evidence["x_type"] = X_TYPE
        evidence["y_id"] = evidence["component_id"]
        evidence["y_type"] = COMPONENT_TYPE
        evidence["evidence_type"] = "protein_localization"
        evidence["source"] = "HPA"
        evidence["source_dataset"] = HPA_DATASET
        evidence["source_release"] = hpa_release
        evidence["predicate"] = evidence["hpa_category"].map(lambda c: f"hpa_{c}")
        evidence["license"] = "HPA proteinatlas.tsv, CC BY 4.0 with third-party caveats; verify exact release terms before canonical promotion"
        evidence["release"] = hpa_release
        evidence["created_at"] = created_at
        evidence = _ensure_columns(evidence, LOCATION_EVIDENCE_COLUMNS).drop_duplicates().sort_values(
            ["edge_key", "source_record_id"]
        )

    node_rows = [mapping.__dict__ for mapping in component_map.values()]
    nodes = pd.DataFrame(node_rows)
    nodes = _ensure_columns(nodes, COMPONENT_NODE_COLUMNS).drop_duplicates("id").sort_values("id")

    hierarchy_rows: list[dict[str, Any]] = []
    hierarchy_evidence_rows: list[dict[str, Any]] = []
    for child_label, parents in HPA_PARENT_LABELS.items():
        if child_label not in component_map:
            continue
        child = component_map[child_label]
        for parent_label, predicate in parents:
            if predicate not in {"part_of", "subtype_of", "located_in"}:
                continue
            if parent_label not in component_map:
                continue
            parent = component_map[parent_label]
            row = {
                "x_id": child.id,
                "x_type": COMPONENT_TYPE,
                "y_id": parent.id,
                "y_type": COMPONENT_TYPE,
                "relation": HIERARCHY_RELATION,
                "display_relation": HIERARCHY_DISPLAY_RELATION,
                "source": "GO/HPA" if child.go_id or parent.go_id else "HPA",
                "credibility": 3 if predicate in {"part_of", "subtype_of", "located_in"} else 1,
            }
            hierarchy_rows.append(row)
            edge_key = f"{HIERARCHY_RELATION}|{child.id}|{parent.id}"
            hierarchy_evidence_rows.append(
                {
                    "edge_key": edge_key,
                    "relation": HIERARCHY_RELATION,
                    "x_id": child.id,
                    "x_type": COMPONENT_TYPE,
                    "y_id": parent.id,
                    "y_type": COMPONENT_TYPE,
                    "evidence_type": "ontology_mapping",
                    "source": row["source"],
                    "source_dataset": "HPA_LABEL_OVERRIDES/GO basic",
                    "source_release": f"{hpa_release}; GO {go_release}".strip(),
                    "source_record_id": f"HPA:25.1:component_hierarchy:{slugify(child_label)}:{slugify(parent_label)}:{predicate}",
                    "predicate": predicate,
                    "mapping_confidence": child.mapping_confidence,
                    "mapping_method": "explicit_hpa_parent_mapping",
                    "raw_labels_json": json.dumps({"child_label": child_label, "parent_label": parent_label}, sort_keys=True),
                    "license": "GO/HPA mapping; verify exact release terms before canonical promotion",
                    "release": hpa_release,
                    "created_at": created_at,
                }
            )
    hierarchy = pd.DataFrame(hierarchy_rows)
    hierarchy_evidence = pd.DataFrame(hierarchy_evidence_rows)
    hierarchy = _ensure_columns(hierarchy, EDGE_COLUMNS) if not hierarchy.empty else pd.DataFrame(columns=EDGE_COLUMNS)
    hierarchy_evidence = _ensure_columns(hierarchy_evidence, HIERARCHY_EVIDENCE_COLUMNS) if not hierarchy_evidence.empty else pd.DataFrame(columns=HIERARCHY_EVIDENCE_COLUMNS)

    invalid_go_nodes: list[dict[str, str]] = []
    if go_terms:
        for row in nodes[nodes["go_id"].fillna("").ne("")].itertuples(index=False):
            go_id = clean(row.go_id)
            term = go_terms.get(go_id)
            reason = ""
            if go_id in REJECTED_HPA_GO_MAPPINGS:
                reason = f"rejected_hpa_go_mapping:{REJECTED_HPA_GO_MAPPINGS[go_id]}"
            elif not term:
                reason = "missing_from_go_obo"
            elif term.get("namespace") != "cellular_component":
                reason = f"non_cellular_component:{term.get('namespace', '')}"
            elif term.get("is_obsolete"):
                reason = "obsolete_go_term"
            elif str(term.get("name", "")).lower().startswith("obsolete") or str(row.name).lower().startswith("obsolete"):
                reason = "obsolete_go_name"
            if reason:
                invalid_go_nodes.append(
                    {"id": clean(row.id), "go_id": go_id, "name": clean(row.name), "reason": reason}
                )
    broad_context_edges = 0
    if not hierarchy_evidence.empty:
        broad_context_edges = int(
            hierarchy_evidence["predicate"].astype(str).str.contains("broad", case=False, na=False).sum()
        )
    ambiguous_distinct_edges = 0
    if not raw_df.empty and "uniprot_mapping_ambiguous" in raw_df.columns:
        ambiguous_distinct_edges = int(
            raw_df[raw_df["uniprot_mapping_ambiguous"]]
            .drop_duplicates(["protein_id", "component_id"])
            .shape[0]
        )
    protein_stats = {
        **protein_stats,
        "hpa_uniprot_expansion_audit": {
            "ambiguous_hpa_source_rows": int(ambiguous_hpa_source_rows),
            "ambiguous_hpa_label_assignments": int(ambiguous_hpa_label_assignments),
            "ambiguous_hpa_expanded_protein_rows": int(ambiguous_hpa_expanded_protein_rows),
            "ambiguous_hpa_distinct_protein_component_edges": ambiguous_distinct_edges,
            "top_ambiguous_uniprots_in_hpa_rows": ambiguous_hpa_uniprots.most_common(20),
            "current_policy": "all ENSP protein nodes linked to an HPA UniProt accession are emitted as protein localization edges",
            "promotion_recommendation": "review canonical ENSP policy before promotion: all ENSP isoforms preserves accession crossrefs but can multiply HPA evidence; canonical ENSP only reduces expansion but needs an approved canonical-protein selector",
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_parquet(output_dir / "nodes" / "cellular_component.parquet", nodes)
    _write_parquet(output_dir / "edges" / f"{PROTEIN_LOCATION_RELATION}.parquet", located)
    _write_parquet(output_dir / "evidence" / f"{PROTEIN_LOCATION_RELATION}.parquet", evidence)
    _write_parquet(output_dir / "edges" / f"{HIERARCHY_RELATION}.parquet", hierarchy)
    _write_parquet(output_dir / "evidence" / f"{HIERARCHY_RELATION}.parquet", hierarchy_evidence)

    edge_keys = set(located.apply(lambda r: f"{PROTEIN_LOCATION_RELATION}|{r['x_id']}|{r['y_id']}", axis=1)) if not located.empty else set()
    evidence_keys = set(evidence["edge_key"].dropna().astype(str)) if not evidence.empty else set()
    validation = {
        "checks": {
            "protein_endpoint_antijoin": {"missing_protein_nodes": 0},
            "component_endpoint_antijoin": {
                "missing_component_nodes": int(located[~located["y_id"].isin(set(nodes["id"]))].shape[0]) if not located.empty else 0
            },
            "evidence_support": {
                "edges_without_evidence": int(len(edge_keys - evidence_keys)),
                "evidence_without_edge": int(len(evidence_keys - edge_keys)),
            },
            "go_term_semantics": {
                "invalid_go_nodes": int(len(invalid_go_nodes)),
                "invalid_go_node_examples": invalid_go_nodes[:20],
            },
            "hierarchy_semantics": {"broad_context_edges": broad_context_edges},
        },
        "source_rows": {
            "hpa_rows": hpa_rows,
            "hpa_rows_with_subcellular_or_secretome_label": rows_with_any_label,
            "hpa_rows_without_protein_mapping": rows_without_protein_mapping,
            "top_unmapped_uniprots": unmapped_uniprot_rows.most_common(20),
        },
        "counts": {
            "cellular_component_nodes": int(len(nodes)),
            "go_mapped_nodes": int(nodes["go_id"].fillna("").ne("").sum()) if not nodes.empty else 0,
            "hpa_local_nodes": int(nodes["id"].astype(str).str.startswith("HPA_SL:").sum()) if not nodes.empty else 0,
            "protein_location_edges": int(len(located)),
            "protein_location_evidence": int(len(evidence)),
            "hierarchy_edges": int(len(hierarchy)),
            "hierarchy_evidence": int(len(hierarchy_evidence)),
            "distinct_hpa_labels": int(len(label_counts)),
        },
        "label_counts": dict(sorted(label_counts.items(), key=lambda item: (-item[1], item[0]))),
        "mapping_sources": {
            "hpa_release": hpa_release,
            "go_release": go_release,
            "uniprot_subcell_release": uniprot_release,
        },
        "protein_mapping": protein_stats,
    }
    manifest = {
        "staging_only": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "relations": [PROTEIN_LOCATION_RELATION, HIERARCHY_RELATION],
        "node_type": COMPONENT_TYPE,
        "output_dir": str(output_dir),
        "source_policy": "HPA subcellular/secretome labels mapped to GO CC where feasible with HPA local fallback IDs; protein endpoints resolved only through canonical nodes/protein.parquet UniProt xrefs; no gene-to-protein projection; no generic all-cells-have-component assertions.",
        "license_note": "HPA data should be attributed and exact release terms rechecked before canonical promotion; GO and UniProt mappings are cached for reproducibility.",
        "validation": validation,
    }
    (output_dir / "validation.json").write_text(json.dumps(validation, indent=2, sort_keys=True))
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    if any(check.get("missing_protein_nodes", 0) or check.get("missing_component_nodes", 0) for check in validation["checks"].values() if isinstance(check, dict)):
        raise ValueError(f"endpoint validation failed: {validation['checks']}")
    if validation["checks"]["go_term_semantics"]["invalid_go_nodes"]:
        raise ValueError(f"go term semantic validation failed: {validation['checks']['go_term_semantics']}")
    if validation["checks"]["hierarchy_semantics"]["broad_context_edges"]:
        raise ValueError(f"hierarchy semantic validation failed: {validation['checks']['hierarchy_semantics']}")
    support = validation["checks"]["evidence_support"]
    if support["edges_without_evidence"] or support["evidence_without_edge"]:
        raise ValueError(f"evidence support validation failed: {support}")
    return manifest


def default_output_dir(base: Path | str = "artifacts/staged") -> Path:
    return Path(base) / f"hpa-cellular-components-{date.today().isoformat()}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hpa-zip", default="/Users/jkobject/mnt/gcs/jouvencekb/main/raw/hpa-25.1/proteinatlas.tsv.zip")
    parser.add_argument("--protein-nodes", default="/Users/jkobject/mnt/gcs/jouvencekb/main/nodes/protein.parquet")
    parser.add_argument("--go-obo", default="/Users/jkobject/mnt/gcs/jouvencekb/main/raw/go-current/go-basic.obo")
    parser.add_argument("--uniprot-subcell", default="/Users/jkobject/mnt/gcs/jouvencekb/main/raw/uniprot-current/subcell.txt")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--hpa-release", default="HPA 25.1")
    parser.add_argument("--download-missing", action="store_true", help="Download public HPA/GO/UniProt mapping files when missing")
    args = parser.parse_args(argv)

    hpa_zip = Path(args.hpa_zip)
    protein_nodes = Path(args.protein_nodes)
    go_obo = Path(args.go_obo)
    uniprot_subcell = Path(args.uniprot_subcell)
    if args.download_missing:
        _download_if_missing(hpa_zip, HPA_URL)
        _download_if_missing(go_obo, GO_BASIC_URL)
        _download_if_missing(uniprot_subcell, UNIPROT_SUBCELL_URL)
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir()
    manifest = build_artifacts(
        hpa_zip=hpa_zip,
        protein_nodes=protein_nodes,
        output_dir=output_dir,
        go_obo=go_obo,
        uniprot_subcell=uniprot_subcell,
        hpa_release=args.hpa_release,
    )
    print(json.dumps(manifest["validation"], indent=2, sort_keys=True))
    print(f"wrote staged HPA cellular-component artifacts to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
