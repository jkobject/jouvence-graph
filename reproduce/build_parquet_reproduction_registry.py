"""Build the exact per-Parquet reproduction lineage registry.

The catalog is the canonical identity/schema/count denominator.  This module adds
conservative, source-family lineage without treating migration receipts or the
older source-family notebooks as proof of biological replayability.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "docs/parquet-catalog/inventory.json"
SOURCE_INVENTORY_PATH = ROOT / "reproduce/source_family_inventory.json"
MIGRATION_MAP_PATH = ROOT / "docs/storage-migration-20260727/object-map.json"
REGISTRY_PATH = ROOT / "reproduce/parquet_reproduction_lineage.json"

STATUSES = {
    "fully-replayable",
    "documented-not-replayed",
    "historical-builder-only",
    "provenance-gap",
}

RAW_OBJECTS = {
    "biogrid": ["gs://jouvencekb/raw/biogrid_SYSTEM_5.0.258.tab3.zip"],
    "cellosaurus": [
        "gs://jouvencekb/raw/cellosaurus_20260623.txt",
        "gs://jouvencekb/raw/cellosaurus_20260623.obo",
    ],
    "depmap": ["gs://jouvencekb/raw/depmap_26Q1_CRISPRGeneDependency.csv"],
    "ensembl": ["gs://jouvencekb/raw/ensembl_Homo_sapiens_GRCh38_cdna_release_114.fa.gz"],
    "hpa": ["gs://jouvencekb/raw/hpa_pathology_proteinatlas_25.1.tsv.zip"],
    "hpo": ["gs://jouvencekb/raw/hpo_hp.obo", "gs://jouvencekb/raw/hpo_phenotype.hpoa"],
    "reactome": ["gs://jouvencekb/raw/reactome_UniProt2Reactome_All_Levels_20260323.txt"],
    "uniprot": ["gs://jouvencekb/raw/uniprot_entries_20260623.slim.json"],
    "uberon": ["gs://jouvencekb/raw/uberon_basic.obo"],
}

FAMILIES: dict[str, dict[str, Any]] = {
    "opentargets": {
        "label": "Open Targets Platform 26.03 and bundled ENCODE rE2G",
        "release": "Open Targets Platform 26.03; ENCODE rE2G snapshot bundled in that release where applicable",
        "native_inputs": ["https://ftp.ebi.ac.uk/pub/databases/opentargets/platform/26.03/output/etl/parquet/"],
        "builder": "manage_db/ingest_opentargets.py",
        "command": "uv run python -m manage_db.ingest_opentargets --data-dir artifacts/cache/<task-id>/opentargets --release 26.03",
        "mappings": "Normalize release-pinned ENSG/ENST/ENSP, ChEMBL, EFO, HP and variant identifiers; resolve endpoints against canonical typed node tables before accepting assertions.",
        "transformations": "Select source-native datasets, normalize endpoint IDs and predicates, retain scores/context in evidence, and emit broad graph assertions separately from source support rows.",
        "exclusions": "Quarantine unmapped or ambiguous endpoints and unsupported predicates; never project gene endpoints to proteins or convert association evidence into causal assertions.",
        "problems": "The current code and release endpoint remain, but the exact accepted raw Open Targets snapshot is not retained under raw/ and per-table historical commands/receipts are incomplete.",
        "links": ["reproduce/07_opentargets_edges_and_evidence.ipynb", "reproduce/26_source_reproduction_index.ipynb", "manage_db/ingest_opentargets.py"],
    },
    "txgnn_legacy": {
        "label": "TxData/TxGNN legacy source bundle",
        "release": "TxData Dataverse DOI 10.7910/DVN/CNQV69; constituent upstream releases vary and are not always recoverable",
        "native_inputs": ["https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/CNQV69"],
        "builder": "manage_db/export_kg.py",
        "command": None,
        "mappings": "Normalize legacy node and endpoint identifiers to the declared Jouvence namespaces while preserving source labels and typed endpoints.",
        "transformations": "Export legacy source tables into typed nodes, deduplicated edge assertions and evidence where recoverable.",
        "exclusions": "Do not infer missing evidence, source releases or endpoint projections from a legacy relation name.",
        "problems": "The immutable Dataverse identity is known, but the complete accepted acquisition/build argv and several constituent source releases are not tracked.",
        "links": ["reproduce/06_build_core_edges_and_evidence.ipynb", "reproduce/26_source_reproduction_index.ipynb", "manage_db/export_kg.py"],
    },
    "cellosaurus": {
        "label": "Cellosaurus identity and text",
        "release": "Cellosaurus release 55.0; retained 2026-06-23 text and OBO snapshots",
        "native_inputs": RAW_OBJECTS["cellosaurus"],
        "builder": "manage_db/build_textual_summary_features.py",
        "command": "uv run python -m manage_db.build_textual_summary_features --node-root <worker-local-main>/nodes --output-root artifacts/staged/<task-id>/textual-summary --release Cellosaurus-55.0 --cellosaurus-obo <worker-local-raw>/cellosaurus_20260623.obo",
        "mappings": "Map Cellosaurus CVCL identifiers and DepMap ACH cross-references exactly; use ontology comments as source-backed text.",
        "transformations": "Parse OBO records, normalize xrefs and text, join to canonical cell-line endpoints, and deduplicate source records.",
        "exclusions": "Quarantine absent or ambiguous CVCL/ACH mappings and empty comments; do not derive biological edges from free text.",
        "problems": "The text-feature builder is current, but the original cell-line node/origin-edge builder lineage predates the retained per-Parquet reproduction layer.",
        "links": ["reproduce/28_cell_line_pharmacology_clinical_reproduction.ipynb", "manage_db/build_textual_summary_features.py"],
    },
    "depmap": {
        "label": "DepMap/CCLE cell-line assays",
        "release": "Canonical lineage spans historical CCLE plus retained DepMap 26Q1 CRISPR dependency; exact per-table accepted release varies",
        "native_inputs": RAW_OBJECTS["depmap"],
        "builder": "manage_db/build_staged_cell_line_assays.py",
        "command": None,
        "mappings": "Join exact DepMap ModelID/ACH identifiers and Entrez-to-canonical-gene mappings; preserve assay score and release in evidence.",
        "transformations": "Parse wide assay matrices, normalize gene columns, threshold only where policy requires, and separate graph assertions from measurements.",
        "exclusions": "Reject missing model/gene mappings and malformed scores; never reinterpret gene dependency or RNA as protein expression.",
        "problems": "The CRISPR matrix is retained, but the exact accepted Model.csv/crosswalk set and production argv are not retained together.",
        "links": ["reproduce/28_cell_line_pharmacology_clinical_reproduction.ipynb", "manage_db/build_staged_cell_line_assays.py"],
    },
    "ensembl": {
        "label": "Ensembl release 114 GRCh38 identity and sequence",
        "release": "Ensembl release 114, GRCh38",
        "native_inputs": RAW_OBJECTS["ensembl"],
        "builder": "manage_db/build_sequence_features.py",
        "command": "uv run python -m manage_db.build_sequence_features --kg-root <worker-local-main> --output-root artifacts/staged/<task-id>/sequence-features --transcript-fasta <worker-local-raw>/ensembl_Homo_sapiens_GRCh38_cdna_release_114.fa.gz --source-release 'Ensembl release 114'",
        "mappings": "Strip Ensembl version suffixes and join exact ENSG/ENST/ENSP identifiers to canonical typed endpoints.",
        "transformations": "Parse release-pinned FASTA or identity tables, normalize sequence alphabets and emit deterministic source-feature keys.",
        "exclusions": "Reject non-matching IDs, invalid sequence symbols and over-policy lengths; do not invent protein sequences from transcript sequence.",
        "problems": "Transcript cDNA is retained and current code exists; protein FASTA and the complete node/central-dogma build inputs are not retained in the current raw surface.",
        "links": ["reproduce/29_official_features_exports_reproduction.ipynb", "manage_db/build_sequence_features.py"],
    },
    "hpo": {
        "label": "Human Phenotype Ontology",
        "release": "HPO release 2025-05-06",
        "native_inputs": RAW_OBJECTS["hpo"],
        "builder": "manage_db/build_textual_summary_features.py",
        "command": "uv run python -m manage_db.build_textual_summary_features --node-root <worker-local-main>/nodes --output-root artifacts/staged/<task-id>/textual-summary --release HPO-2025-05-06 --hpo-obo <worker-local-raw>/hpo_hp.obo",
        "mappings": "Normalize HP identifiers, obsolete/replacement terms and disease/gene annotation endpoints.",
        "transformations": "Parse ontology definitions and annotations, normalize identifiers, build hierarchy/association assertions and source-backed summaries.",
        "exclusions": "Quarantine obsolete-unresolved terms, missing endpoints and unsupported association direction; ontology hierarchy is not clinical causality.",
        "problems": "Raw ontology/annotation files are retained, but the exact accepted graph-builder argv is historical; only the textual-summary argv is current and explicit.",
        "links": ["reproduce/29_official_features_exports_reproduction.ipynb", "manage_db/build_textual_summary_features.py"],
    },
    "reactome": {
        "label": "Reactome and Gene Ontology pathway lineage",
        "release": "Reactome snapshot 2026-03-23 plus Open Targets 26.03 pathway inputs",
        "native_inputs": RAW_OBJECTS["reactome"],
        "builder": "manage_db/build_reactome_pathway_protein_membership.py",
        "command": None,
        "mappings": "Normalize R-HSA pathway IDs and exact gene/protein endpoints; preserve source database and hierarchy context.",
        "transformations": "Parse hierarchy and membership, deduplicate assertions and retain source-specific support separately.",
        "exclusions": "Reject non-human, unmapped or ambiguous endpoints; do not project protein-native memberships into gene relations without an explicit policy.",
        "problems": "A Reactome protein-membership source is retained, but canonical gene/pathway tables combine historical Reactome/GO lanes whose exact accepted argv is incomplete.",
        "links": ["reproduce/27_source_native_protein_context_reproduction.ipynb", "manage_db/build_reactome_pathway_protein_membership.py"],
    },
    "hpa": {
        "label": "Human Protein Atlas direct protein context",
        "release": "Human Protein Atlas 25.1",
        "native_inputs": RAW_OBJECTS["hpa"],
        "builder": "manage_db/backfill_protein_expression.py",
        "command": None,
        "mappings": "Map HPA gene/protein records to unambiguous canonical protein endpoints and tissue identifiers; retain direct assay/staining metadata.",
        "transformations": "Normalize tissue labels, protein measurements and pathology context while retaining modality in evidence.",
        "exclusions": "Reject unmapped endpoints and non-direct measurements; never populate protein expression by RNA-to-protein projection.",
        "problems": "A retained pathology snapshot and current builders exist, but the complete accepted canonical tissue-protein argv/input set is not preserved.",
        "links": ["reproduce/27_source_native_protein_context_reproduction.ipynb", "manage_db/backfill_protein_expression.py"],
    },
    "biogrid": {
        "label": "BioGRID physical protein interactions",
        "release": "BioGRID 5.0.258 TAB3",
        "native_inputs": RAW_OBJECTS["biogrid"],
        "builder": None,
        "command": None,
        "mappings": "Map source-native UniProt/RefSeq protein identifiers only when unambiguous and preserve experimental system/publication support.",
        "transformations": "Filter physical human interactions, deduplicate graph assertions, and retain multiple source evidence records.",
        "exclusions": "Reject genetic interactions, non-human rows, unresolved proteins and residue conflicts; PTM/PTMREL lanes remain separate/deferred.",
        "problems": "The accepted historical categorized builder was rejected as a current replay authority; the retained raw archive and promotion report preserve clues, not a current exact builder.",
        "links": ["reproduce/27_source_native_protein_context_reproduction.ipynb", "docs/part2_biogrid_ppi_canonical_promotion_report.md"],
    },
    "uniprot": {
        "label": "UniProtKB reviewed human protein features",
        "release": "Retained 2026-06-23 slim UniProt payload; accepted table release is recorded row-wise",
        "native_inputs": RAW_OBJECTS["uniprot"],
        "builder": "manage_db/build_textual_summary_features.py",
        "command": "uv run python -m manage_db.build_textual_summary_features --node-root <worker-local-main>/nodes --output-root artifacts/staged/<task-id>/textual-summary --release 2026-06-23 --uniprot-entries-json <worker-local-raw>/uniprot_entries_20260623.slim.json",
        "mappings": "Join UniProt accessions unambiguously to ENSP protein nodes and preserve source record/release fields.",
        "transformations": "Normalize reviewed protein summaries and source fields, enforce text limits and deduplicate feature keys.",
        "exclusions": "Reject ambiguous/unmapped accessions and empty payloads; sequence and disease assertions require separate source-native inputs.",
        "problems": "The slim text payload and current textual builder are retained; it is not a complete UniProt release for all protein-derived products.",
        "links": ["reproduce/27_source_native_protein_context_reproduction.ipynb", "manage_db/build_textual_summary_features.py"],
    },
    "uberon": {
        "label": "UBERON anatomy ontology",
        "release": "UBERON 2026-04-01 observed snapshot",
        "native_inputs": RAW_OBJECTS["uberon"],
        "builder": "manage_db/build_textual_summary_features.py",
        "command": "uv run python -m manage_db.build_textual_summary_features --node-root <worker-local-main>/nodes --output-root artifacts/staged/<task-id>/textual-summary --release UBERON-2026-04-01 --uberon-obo <worker-local-raw>/uberon_basic.obo",
        "mappings": "Join exact UBERON identifiers to canonical tissue nodes and normalize ontology parent/definition fields.",
        "transformations": "Parse anatomy ontology, normalize identifiers and emit hierarchy or source-backed text according to the target table.",
        "exclusions": "Reject obsolete-unresolved or absent endpoints; anatomy hierarchy is not experimental biological evidence.",
        "problems": "The ontology and current text builder are retained; the canonical tissue-node/hierarchy build argv is historical.",
        "links": ["reproduce/29_official_features_exports_reproduction.ipynb", "manage_db/build_textual_summary_features.py"],
    },
    "rdkit": {
        "label": "RDKit Morgan fingerprint derivation",
        "release": "Canonical molecule snapshot plus RDKit 2026.03.3; Morgan radius 2, 2048 bits",
        "native_inputs": ["gs://jouvencekb/main/nodes/molecule.parquet"],
        "builder": "manage_db/build_node_missing_features.py",
        "command": "uv run python -m manage_db.build_node_missing_features --kg-root <worker-local-main> --output-root artifacts/staged/<task-id>/node-features --source-release KG-molecule-snapshot+RDKit-2026.03.3",
        "mappings": "Use canonical molecule ID as the feature key and parse the retained canonical structure field.",
        "transformations": "Generate deterministic sparse Morgan binary fingerprints at radius 2 and 2048 bits with a content hash.",
        "exclusions": "Reject missing or invalid structures; fingerprints are computed features, not biological assertions.",
        "problems": "The builder is current and inputs are resolvable, but this per-Parquet layer did not replay the full canonical output.",
        "links": ["reproduce/29_official_features_exports_reproduction.ipynb", "manage_db/build_node_missing_features.py", "manage_db/kg_molecule_fingerprint_features.py"],
    },
    "clinical_trials": {
        "label": "ClinicalTrials.gov API v2 sidecars",
        "release": "Frozen API v2 response chunks used by the accepted staged/canonical candidate; fetch timestamps identify snapshots",
        "native_inputs": ["https://clinicaltrials.gov/data-api/api"],
        "builder": "manage_db/stage_clinicaltrials_gov_production_candidate.py",
        "command": None,
        "mappings": "Join NCT IDs to Open Targets treatment support and canonical molecule-disease keys where explicitly asserted.",
        "transformations": "Normalize trial metadata/text into sidecars; keep trial records separate from graph topology.",
        "exclusions": "Report fetch misses and unsupported treatment links; trial registration is not proof of efficacy.",
        "problems": "The accepted frozen response chunks are not retained under raw/, and the current builder may fetch missing studies; no complete offline exact argv is tracked.",
        "links": ["reproduce/28_cell_line_pharmacology_clinical_reproduction.ipynb", "manage_db/stage_clinicaltrials_gov_production_candidate.py"],
    },
    "remap": {
        "label": "ReMap CRM aggregate support",
        "release": "ReMap 2022, GRCh38/hg38",
        "native_inputs": [],
        "builder": None,
        "command": None,
        "mappings": "Join TF gene identifiers and enhancer intervals only as aggregate support keys; retain genome build and aggregation policy.",
        "transformations": "Aggregate chromosome-level CRM support shards and compact them byte-independently with row-count/SHA-256 validation.",
        "exclusions": "This is support/QA, not graph topology, observed TF binding or TF-to-gene regulation.",
        "problems": "The migration receipt preserves all historical shard names and compaction checks, but no selected native ReMap raw object or current source-native builder is retained.",
        "links": ["docs/remap_crm_canonical_readiness.md", "docs/storage-migration-20260727/README.md"],
    },
    "text_embeddings": {
        "label": "Source-backed biomedical text embedding pipeline",
        "release": "Accepted immutable S-BioBERT text embedding candidate; exact source revision is row-level and in preserved release evidence where available",
        "native_inputs": [],
        "builder": "manage_db/build_real_embeddings.py",
        "command": None,
        "mappings": "Join exact canonical node IDs to source-feature keys and hashes; preserve model, pooling, normalization and coverage metadata.",
        "transformations": "Encode source-backed textual summaries with the recorded S-BioBERT model, pooling and normalization policy; retain deterministic feature lineage.",
        "exclusions": "Do not fabricate vectors for missing source payloads or treat similarity as causality/equivalence; learned fallback stays model-side.",
        "problems": "The accepted text-vector objects and migration receipts survive, but a single verified exact argv with pinned model revision and all source-feature checksums is not tracked for every leaf.",
        "links": ["reproduce/29_official_features_exports_reproduction.ipynb", "manage_db/build_real_embeddings.py"],
    },
    "sequence_embeddings": {
        "label": "Nucleotide Transformer and ESM2 sequence embedding pipelines",
        "release": "Accepted immutable genomic/transcript/protein sequence embedding candidates; model identity is encoded in each canonical leaf",
        "native_inputs": [],
        "builder": "manage_db/build_real_embeddings.py",
        "command": None,
        "mappings": "Join exact ENSG, ENST or ENSP canonical IDs to source sequences and hashes; preserve model, pooling, normalization and coverage metadata.",
        "transformations": "Encode genomic, transcript or protein sequence with the leaf-specific Nucleotide Transformer or ESM2 model and retain deterministic feature lineage.",
        "exclusions": "Do not fabricate vectors for missing sequences, cross-project one sequence modality into another or treat similarity as biological evidence.",
        "problems": "Accepted sequence-vector objects and migration receipts survive, but no single verified argv with pinned model revisions and all accepted input checksums covers every sequence leaf.",
        "links": ["reproduce/29_official_features_exports_reproduction.ipynb", "manage_db/build_real_embeddings.py"],
    },
    "molecule_embeddings": {
        "label": "ChemBERTa molecule-structure embedding pipeline",
        "release": "Accepted immutable ChemBERTa molecule SMILES embedding candidate; exact source revision is preserved in release evidence where available",
        "native_inputs": ["gs://jouvencekb/main/nodes/molecule.parquet"],
        "builder": "manage_db/build_real_embeddings.py",
        "command": None,
        "mappings": "Join canonical molecule IDs to validated source structures and hashes; preserve model, pooling, normalization and coverage metadata.",
        "transformations": "Encode valid canonical molecule structures with the recorded ChemBERTa model and retain deterministic feature lineage.",
        "exclusions": "Reject missing or invalid structures; do not fabricate vectors or treat chemical similarity as biological evidence.",
        "problems": "The accepted molecule-vector object and migration receipt survive, but the exact pinned production argv and structure-input checksum set are incomplete.",
        "links": ["reproduce/29_official_features_exports_reproduction.ipynb", "manage_db/build_real_embeddings.py"],
    },
}

FAMILY_NOTEBOOKS = {
    "opentargets": "opentargets_associations.ipynb",
    "txgnn_legacy": "txgnn_legacy_bundle.ipynb",
    "cellosaurus": "cellosaurus_identity.ipynb",
    "depmap": "depmap_cell_context.ipynb",
    "ensembl": "ensembl_identity_and_sequence.ipynb",
    "hpo": "human_phenotype_ontology.ipynb",
    "reactome": "reactome_pathways.ipynb",
    "hpa": "hpa_protein_context.ipynb",
    "biogrid": "biogrid_protein_interactions.ipynb",
    "uniprot": "uniprot_protein_text.ipynb",
    "uberon": "uberon_anatomy.ipynb",
    "rdkit": "molecule_fingerprints.ipynb",
    "clinical_trials": "clinical_trials.ipynb",
    "remap": "remap_regulatory_support.ipynb",
    "text_embeddings": "text_embeddings.ipynb",
    "sequence_embeddings": "sequence_embeddings.ipynb",
    "molecule_embeddings": "molecule_embeddings.ipynb",
}

# Producer evidence is fail-closed and per canonical output. Related family
# code is still linked as context, but is not promoted to `producer` unless the
# tracked implementation actually emits that exact output.
EXACT_PRODUCERS = {
    "features__cell_line_textual_summary": "manage_db/build_textual_summary_features.py",
    "edges__cell_line_gene_essentiality": "manage_db/build_staged_cell_line_assays.py",
    "evidence__cell_line_gene_essentiality": "manage_db/build_staged_cell_line_assays.py",
    "features__protein_sequence": "manage_db/build_sequence_features.py",
    "features__transcript_sequence": "manage_db/build_sequence_features.py",
    "features__phenotype_textual_summary": "manage_db/build_textual_summary_features.py",
    "features__protein_textual_summary": "manage_db/build_textual_summary_features.py",
    "features__tissue_textual_summary": "manage_db/build_textual_summary_features.py",
    "features__molecule_fingerprint": "manage_db/build_node_missing_features.py",
    "embedding__cell_line_text_sbiobert_snli_multinli_stsb": "manage_db/build_real_embeddings.py",
    "embedding__cell_type_text_sbiobert_snli_multinli_stsb": "manage_db/build_real_embeddings.py",
    "embedding__disease_text_sbiobert_snli_multinli_stsb": "manage_db/build_real_embeddings.py",
    "embedding__gene_text_sbiobert_snli_multinli_stsb": "manage_db/build_real_embeddings.py",
    "embedding__molecule_text_sbiobert_snli_multinli_stsb": "manage_db/build_real_embeddings.py",
    "embedding__pathway_text_sbiobert_snli_multinli_stsb": "manage_db/build_real_embeddings.py",
    "embedding__phenotype_text_sbiobert_snli_multinli_stsb": "manage_db/build_real_embeddings.py",
    "embedding__protein_text_sbiobert_snli_multinli_stsb": "manage_db/build_real_embeddings.py",
    "embedding__tissue_text_sbiobert_snli_multinli_stsb": "manage_db/build_real_embeddings.py",
}

EXACT_COMMANDS = {
    "features__cell_line_textual_summary": FAMILIES["cellosaurus"]["command"],
    "features__transcript_sequence": FAMILIES["ensembl"]["command"],
    "features__phenotype_textual_summary": FAMILIES["hpo"]["command"],
    "features__protein_textual_summary": FAMILIES["uniprot"]["command"],
    "features__tissue_textual_summary": FAMILIES["uberon"]["command"],
    "features__molecule_fingerprint": FAMILIES["rdkit"]["command"],
}

OPEN_TARGETS_PREFIXES = (
    "disease_associated_", "disease_involves_", "disease_manifests_",
    "enhancer_regulates_", "gene_interacts_", "mutation_", "molecule_targets_",
)
PROVENANCE_GAP_IDS = {
    "nodes__dataset", "nodes__paper", "edges__molecule_associated_phenotype",
    "edges__molecule_contraindicates_disease", "edges__molecule_parent_of_molecule",
    "edges__molecule_synergizes_molecule", "edges__molecule_treats_disease",
    "features__clinical_trials_gov_trial_index", "features__clinical_trials_gov_trial_text_features",
    "features__molecule_treats_disease_clinical_trial_links",
}
DOCUMENTED_IDS = {
    "features__cell_line_textual_summary", "features__molecule_fingerprint",
    "features__phenotype_textual_summary", "features__protein_textual_summary",
    "features__tissue_textual_summary", "features__transcript_sequence",
}


def family_for(layer: str, name: str) -> str:
    if layer == "embedding":
        if name == "molecule_smiles_chemberta_77m_mlm":
            return "molecule_embeddings"
        if "sequence" in name:
            return "sequence_embeddings"
        return "text_embeddings"
    if layer == "nodes":
        return {
            "cell_line": "cellosaurus",
            "cell_type": "txgnn_legacy",
            "dataset": "txgnn_legacy",
            "disease": "opentargets",
            "enhancer": "opentargets",
            "gene": "ensembl",
            "molecule": "opentargets",
            "mutation": "opentargets",
            "organism": "ensembl",
            "paper": "opentargets",
            "pathway": "reactome",
            "phenotype": "hpo",
            "protein": "ensembl",
            "tissue": "uberon",
            "transcript": "ensembl",
        }[name]
    if name.startswith("clinical_trials_gov") or "clinical_trial" in name:
        return "clinical_trials"
    if name == "remap_crm_tf_enhancer_support" or name == "mutation_overlaps_enhancer_support":
        return "remap"
    if name == "molecule_fingerprint":
        return "rdkit"
    if name == "protein_interacts_protein":
        return "biogrid"
    if name == "tissue_expresses_protein":
        return "hpa"
    if name.startswith("cell_line") or name == "dataset_contains_cell_line":
        return "depmap" if ("expresses_gene" in name or "essentiality" in name or name == "dataset_contains_cell_line") else "cellosaurus"
    if name == "cell_type_expresses_gene":
        return "txgnn_legacy"
    if name in {"cell_line_textual_summary"}:
        return "cellosaurus"
    if name in {"protein_textual_summary"}:
        return "uniprot"
    if name in {"tissue_textual_summary", "tissue_subtype_of_tissue", "organism_has_tissue"}:
        return "uberon"
    if name in {"phenotype_textual_summary", "phenotype_subtype_of_phenotype", "disease_has_phenotype", "gene_associated_phenotype"}:
        return "hpo"
    if name in {"protein_sequence", "transcript_sequence", "gene_has_transcript", "transcript_encodes_protein", "organism_has_gene"}:
        return "ensembl"
    if "pathway" in name:
        return "reactome"
    if name.startswith(OPEN_TARGETS_PREFIXES):
        return "opentargets"
    return "txgnn_legacy"


def status_for(dataset_id: str, family: str) -> str:
    if dataset_id in PROVENANCE_GAP_IDS:
        return "provenance-gap"
    if dataset_id in DOCUMENTED_IDS:
        return "documented-not-replayed"
    return "historical-builder-only"


def producer_for(dataset_id: str, family_id: str) -> str | None:
    del family_id  # family membership is validated separately and cannot grant producer evidence
    return EXACT_PRODUCERS.get(dataset_id)


def command_for(dataset_id: str, family_id: str) -> str | None:
    del family_id  # commands are accepted only from the exact-output allowlist
    return EXACT_COMMANDS.get(dataset_id)


def _migration_receipts(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    receipts: dict[str, dict[str, Any]] = {}
    for row in payload["copy_objects"]:
        destination = row["destination"]
        if destination.startswith("main/"):
            receipts[destination] = {
                "action": row["action"],
                "source": row["source"],
                "source_generation": row["source_generation"],
                "destination_generation": row["destination_generation"],
                "destination_crc32c": row["destination_crc32c"],
                "verified": row["verified"],
            }
    for row in payload["compactions"]:
        receipts[row["destination"]] = {
            "action": row["action"],
            "sources": row["sources"],
            "destination_generation": row["destination_generation"],
            "destination_crc32c": row["destination_crc32c"],
            "destination_sha256": row["destination_sha256"],
            "source_rows": row["source_rows"],
            "destination_rows": row["destination_rows"],
            "verified": row["verified"],
        }
    return receipts


def build_registry() -> dict[str, Any]:
    catalog = json.loads(CATALOG_PATH.read_text())
    source_inventory = json.loads(SOURCE_INVENTORY_PATH.read_text())
    migration = json.loads(MIGRATION_MAP_PATH.read_text())
    receipts = _migration_receipts(migration)
    # Loading the historical inventory is an explicit denominator/evidence check;
    # per-record links below remain limited to the relevant family notebook(s).
    if not source_inventory["sources"]:
        raise ValueError("source-family inventory is empty")
    records = []
    for dataset in sorted(catalog["datasets"], key=lambda row: (row["layer"], row["name"])):
        layer, name = dataset["layer"], dataset["name"]
        dataset_id = f"{layer}__{name}"
        family_id = family_for(layer, name)
        family = FAMILIES[family_id]
        producer = producer_for(dataset_id, family_id)
        command = command_for(dataset_id, family_id)
        status = status_for(dataset_id, family_id)
        object_row = dataset["objects"][0]
        canonical_relative = dataset["uri"].removeprefix("gs://jouvencekb/")
        gap_fields = []
        if not family["native_inputs"]:
            gap_fields.append("retained native input")
        if not producer:
            gap_fields.append("current builder")
        if not command:
            gap_fields.append("verified exact rebuild command")
        if status == "provenance-gap":
            gap_fields.append("exact accepted source-to-object lineage")
        links = [
            f"docs/parquet-catalog/datasets/{dataset_id}.md",
            "docs/parquet-catalog/inventory.json",
            "docs/storage-migration-20260727/README.md",
            "docs/storage-migration-20260727/object-map.json",
            *family["links"],
        ]
        record = {
            "layer": layer,
            "name": name,
            "reproduce_notebook": f"notebooks/reproduce/{FAMILY_NOTEBOOKS[family_id]}",
            "pipeline_family": family_id,
            "catalog_page": f"docs/parquet-catalog/datasets/{dataset_id}.md",
            "canonical_uri": dataset["uri"],
            "meaning": dataset["semantics"]["meaning"],
            "non_meaning": dataset["semantics"].get("non_meaning", "No additional non-meaning is declared in the current catalog."),
            "source_family": family_id,
            "source_family_label": family["label"],
            "producer": producer,
            "native_source": family["native_inputs"],
            "native_inputs": family["native_inputs"],
            "release": family["release"],
            "acquisition_and_preconditions": "Read-only by default. Network, requester-pays GCS and production rebuilds require explicit opt-in; full rebuilds run only on an approved in-region worker with task-local staging.",
            "fields": dataset["fields"],
            "keys": dataset["keys"],
            "mappings_and_joins": family["mappings"],
            "transformations_and_filters": family["transformations"],
            "deduplication_and_evidence": "Deduplicate graph assertions by the declared relation/endpoint contract. Preserve source-specific multiplicity, scores, studies, assays and predicates in evidence or feature sidecars; do not treat migration copies as biological evidence.",
            "quarantines_exclusions_missing": family["exclusions"],
            "problems_and_decisions": family["problems"],
            "producer_builder": producer,
            "full_worker_rebuild_command": command,
            "rebuild_command_evidenced": command is not None,
            "safe_bounded_replay": "Validate this frozen registry record, schema contract and migration receipt offline. Optional live mode reads only canonical object metadata/Parquet footer with caller-owned ADC and JOUVENCE_BILLING_PROJECT; it does not scan rows or write.",
            "qc": {
                "rows": dataset["rows"],
                "bytes": dataset["bytes"],
                "row_groups": dataset["row_groups"],
                "schema_hash": dataset["schema_hash"],
                "schema_hash_version": dataset["schema_hash_version"],
                "generation": object_row["generation"],
                "crc32c_base64": object_row["crc32c_base64"],
                "md5_base64": object_row.get("md5_base64"),
                "as_of": dataset["as_of"],
            },
            "migration_receipt": receipts[canonical_relative],
            "reproducibility_status": status,
            "provenance_gaps": sorted(set(gap_fields)),
            "replay_level": status,
            "known_gaps": sorted(set(gap_fields)),
            "reproducibility_limits": "This notebook documents and checks current artifact identity. It does not prove that historical native inputs can recreate identical biological rows unless the status and evidence explicitly establish that stronger claim.",
            "links": list(dict.fromkeys(links)),
        }
        records.append(record)
    return {
        "schema_version": 2,
        "canonical_inventory": "docs/parquet-catalog/inventory.json",
        "canonical_root": catalog["canonical_root"],
        "record_count": len(records),
        "allowed_statuses": sorted(STATUSES),
        "records": records,
    }


def main() -> None:
    REGISTRY_PATH.write_text(json.dumps(build_registry(), indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
