from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
N4A_NOTEBOOK = REPO_ROOT / "reproduce" / "10_source_native_interactions_summary.ipynb"
N4C_NOTEBOOK = REPO_ROOT / "reproduce" / "12_pharmacology_context_metadata_summary.ipynb"
N4D_NOTEBOOK = REPO_ROOT / "reproduce" / "09_source_native_ingestion_index.ipynb"
REPRODUCE_ROOT_NOTEBOOKS = [
    REPO_ROOT / "reproduce" / name
    for name in (
        "10_source_native_interactions_summary.ipynb",
        "11_biological_nodes_context_summary.ipynb",
        "12_pharmacology_context_metadata_summary.ipynb",
        "13_non_remap_canonical_promotion_summary.ipynb",
        "19_node_sequence_text_features_summary.ipynb",
        "21_pyg_out_of_core_and_lamin_remaining.ipynb",
    )
]


def _notebook_text_for(path: Path) -> tuple[dict, str]:
    notebook = json.loads(path.read_text())
    text = "\n".join(
        "".join(cell.get("source", ""))
        for cell in notebook.get("cells", [])
    )
    return notebook, text


def _notebook_text() -> tuple[dict, str]:
    return _notebook_text_for(N4A_NOTEBOOK)


def test_n4a_source_native_interactions_notebook_is_read_only_by_default() -> None:
    notebook, text = _notebook_text()

    assert notebook["nbformat"] == 4
    assert len(notebook["cells"]) >= 10
    assert "ALLOW_CANONICAL_WRITES = False" in text
    assert "TXGNN_NOTEBOOK_FULL_VALIDATION" in text
    assert "TXGNN_NOTEBOOK_ALLOW_GCS_READ" in text
    assert "no downloads" in text
    assert "no canonical KG writes" in text
    assert "no flag in this notebook that performs canonical promotion" in text


def test_n4a_source_native_interactions_notebook_covers_required_tranches() -> None:
    _, text = _notebook_text()

    required_tokens = [
        "IntAct corrected/bounded protein_interacts_protein",
        "t_100231b1",
        "bounded/no node-root",
        "BioGRID physical PPI/PTM/category split",
        "t_28f83a7b",
        "t_d64b99c0",
        "complex outputs intentionally empty",
        "miRNA real-source alias/target path",
        "t_f1b51a59",
        "t_1734823c",
        "t_95bbd18c",
        "t_08770b04",
        "ReMap all-peak lineage and CRM support/QA pilot",
        "t_17f2b3d5",
        "t_8bc6dacf",
        "t_3b8a2c4d",
        "deferred / stopped / not_promoted",
    ]
    for token in required_tokens:
        assert token in text


def test_n4a_source_native_interactions_notebook_has_consolidated_status_and_recommendations() -> None:
    _, text = _notebook_text()

    assert "tranche_status = pd.DataFrame" in text
    assert "accepted / staged-only / not_promoted" in text
    assert "deferred / not_promoted" in text
    assert "promotion_recommendation" in text
    assert "reviewer_status" in text
    assert "source_release" in text
    assert "remote_prefix" in text
    assert "endpoint_anti_join_validation" in text
    assert "evidence_support_validation" in text


def test_n4c_pharmacology_context_metadata_notebook_is_read_only_by_default() -> None:
    notebook, text = _notebook_text_for(N4C_NOTEBOOK)

    assert notebook["nbformat"] == 4
    assert len(notebook["cells"]) >= 10
    assert "ALLOW_CANONICAL_WRITES = False" in text
    assert "TXGNN_NOTEBOOK_FULL_VALIDATION" in text
    assert "TXGNN_NOTEBOOK_ALLOW_GCS_READ" in text
    assert "no downloads" in text
    assert "no canonical KG writes" in text
    assert "no flag in this notebook that performs canonical promotion" in text


def test_n4c_pharmacology_context_metadata_notebook_covers_required_tranches() -> None:
    _, text = _notebook_text_for(N4C_NOTEBOOK)

    required_tokens = [
        "t_19516b59",
        "t_ee55140a",
        "OpenTargets clinical/safety evidence",
        "molecule_synergizes_molecule",
        "disease_associated_protein",
        "pathway_contains_protein",
        "molecule_targets_protein",
        "t_de04b319",
        "t_63ca49a0",
        "t_587ab15a",
        "candidate/context-only",
        "0 accepted edges/evidence",
        "paper/dataset provenance",
        "textual summary feature tables",
        "source decision matrix",
        "schema coverage sweep",
        "downstream-deferred",
        "not_promoted",
    ]
    for token in required_tokens:
        assert token in text


def test_n4c_pharmacology_context_metadata_notebook_has_status_and_gates() -> None:
    _, text = _notebook_text_for(N4C_NOTEBOOK)

    assert "tranche_status = pd.DataFrame" in text
    assert "pharmacology_native = pd.DataFrame" in text
    assert "biological_context = pd.DataFrame" in text
    assert "metadata_features = pd.DataFrame" in text
    assert "source_decision" in text
    assert "relation_or_feature_semantics" in text
    assert "script_or_command_paths" in text
    assert "staged_prefix" in text
    assert "counts_and_rejects" in text
    assert "validation_review_status" in text
    assert "canonical_promotion_gate" in text
    assert '"edge_rows": 0' in text
    assert '"evidence_rows": 0' in text

def test_n4d_source_native_l2_index_notebook_is_read_only_by_default() -> None:
    notebook, text = _notebook_text_for(N4D_NOTEBOOK)

    assert notebook["nbformat"] == 4
    assert len(notebook["cells"]) >= 12
    assert "READ_ONLY = True" in text
    assert "ALLOW_CANONICAL_WRITES = False" in text
    assert "ALLOW_GCS_WRITES = False" in text
    assert "ALLOW_DOWNLOADS = False" in text
    assert "no downloads" in text
    assert "no GCS writes" in text
    assert "no canonical KG writes" in text


def test_n4d_source_native_l2_index_resolves_repo_root_from_reproduce(
    monkeypatch,
) -> None:
    notebook, _ = _notebook_text_for(N4D_NOTEBOOK)
    setup_cell = next(
        cell for cell in notebook["cells"] if cell.get("cell_type") == "code"
    )
    namespace: dict[str, object] = {}

    monkeypatch.chdir(REPO_ROOT / "reproduce")
    exec(compile("".join(setup_cell["source"]), str(N4D_NOTEBOOK), "exec"), namespace)

    assert namespace["REPO_ROOT"] == REPO_ROOT


def test_moved_reproduction_notebooks_resolve_the_current_repo_root() -> None:
    for path in REPRODUCE_ROOT_NOTEBOOKS:
        _, text = _notebook_text_for(path)
        assert "pyproject.toml" in text, path
        assert ".parent" in text, path
        assert "/Users/jkobject/.openclaw/workspace/work/txgnn" not in text, path


def test_n4d_source_native_l2_index_links_split_notebooks_and_lanes() -> None:
    _, text = _notebook_text_for(N4D_NOTEBOOK)

    required_tokens = [
        "10_source_native_interactions_summary.ipynb",
        "11_biological_nodes_context_summary.ipynb",
        "12_pharmacology_context_metadata_summary.ipynb",
        "13_non_remap_canonical_promotion_summary.ipynb",
        "19_node_sequence_text_features_summary.ipynb",
        "top_level_matrix = pd.DataFrame",
        "tranche_status = pd.DataFrame",
        "edge_promotion_matrix = pd.DataFrame",
        "feature_layer_matrix = pd.DataFrame",
        "not_promoted_or_deferred = pd.DataFrame",
        "t_61fabcf3 -> t_ce6e158c -> t_17cfc462",
        "t_b5dd2399, t_76c42ace, t_f9ef6389",
        "t_e6227487",
        "t_c1feb247",
    ]
    for token in required_tokens:
        assert token in text


def test_n4d_source_native_l2_index_separates_edge_promotion_from_feature_layer() -> None:
    _, text = _notebook_text_for(N4D_NOTEBOOK)

    assert "Promoted only BioGRID physical PPI to canonical edge/evidence" in text
    assert "gs://jouvencekb/main/edges/protein_interacts_protein.parquet" in text
    assert "gs://jouvencekb/main/evidence/protein_interacts_protein.parquet" in text
    assert "Feature-layer promotion, not biological edge/evidence promotion" in text
    assert "gs://jouvencekb/main/features/" in text
    assert "molecule_fingerprint.parquet" in text
    assert "protein_textual_summary.parquet" in text
    assert "gene_sequence.parquet" in text
    assert "deferred / not_promoted" in text


def test_n4d_source_native_l2_index_has_explicit_deferred_statuses() -> None:
    _, text = _notebook_text_for(N4D_NOTEBOOK)

    required_tokens = [
        "all-peak ReMap tf_binds_enhancer",
        "Not promoted to canonical",
        "CRM is staged support/QA only",
        "BioGRID PTM / ptm_site",
        "miRNA target path",
        "Sci-Plex cell_type_responds_to_molecule",
        "0 accepted edges/evidence",
        "gene_sequence / gene_genomic_interval",
        "no reviewed coordinate/reference-build/mapping/strand/length/FASTA checksum policy",
    ]
    for token in required_tokens:
        assert token in text
