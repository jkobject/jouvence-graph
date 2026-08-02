from __future__ import annotations

import json
import re
import subprocess
from copy import deepcopy
from collections import Counter
from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient

from reproduce.build_parquet_reproduction_registry import STATUSES, build_registry
from reproduce.generate_parquet_reproduction_notebooks import (
    build_family_notebook,
    build_readme,
    family_records,
    validate_registry,
)

ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "docs/parquet-catalog/inventory.json"
REGISTRY_PATH = ROOT / "reproduce/parquet_reproduction_lineage.json"
NOTEBOOK_DIR = ROOT / "notebooks/reproduce"
REQUIRED_RECORD_FIELDS = {
    "layer",
    "name",
    "reproduce_notebook",
    "pipeline_family",
    "producer",
    "native_source",
    "replay_level",
    "known_gaps",
    "catalog_page",
    "canonical_uri",
    "meaning",
    "non_meaning",
    "source_family",
    "source_family_label",
    "native_inputs",
    "release",
    "acquisition_and_preconditions",
    "fields",
    "keys",
    "mappings_and_joins",
    "transformations_and_filters",
    "deduplication_and_evidence",
    "quarantines_exclusions_missing",
    "problems_and_decisions",
    "producer_builder",
    "full_worker_rebuild_command",
    "rebuild_command_evidenced",
    "safe_bounded_replay",
    "qc",
    "migration_receipt",
    "reproducibility_status",
    "provenance_gaps",
    "reproducibility_limits",
    "links",
}


def registry() -> dict:
    return json.loads(REGISTRY_PATH.read_text())


def identity(row: dict) -> tuple[str, str]:
    return row["layer"], row["name"]


def test_registry_is_exact_deterministic_catalog_denominator() -> None:
    catalog = json.loads(CATALOG_PATH.read_text())
    payload = registry()
    assert payload == build_registry()
    validate_registry(payload)
    expected = {(row["layer"], row["name"]) for row in catalog["datasets"]}
    actual = [identity(row) for row in payload["records"]]
    assert len(actual) == len(set(actual)) == 110
    assert set(actual) == expected
    assert payload["record_count"] == catalog["dataset_count"] == 110


def test_every_parquet_has_one_primary_family_notebook_with_a_defensible_group() -> None:
    payload = registry()
    grouped = family_records(payload)
    notebook_paths = [row["reproduce_notebook"] for row in payload["records"]]
    assert 1 <= len(grouped) <= 20
    assert len(grouped) == 17
    assert set(grouped) == set(notebook_paths)
    assert all(path.startswith("notebooks/reproduce/") and path.endswith(".ipynb") for path in grouped)
    assert all(rows and len({row["pipeline_family"] for row in rows}) == 1 for rows in grouped.values())
    assert all(len({row["reproduce_notebook"] for row in rows}) == 1 for rows in grouped.values())


def test_validator_rejects_missing_duplicate_stale_ambiguous_and_over_cap_coverage() -> None:
    payload = registry()

    missing = deepcopy(payload)
    missing["records"].pop()
    with pytest.raises(SystemExit, match="registry/catalog mismatch"):
        validate_registry(missing)

    duplicate = deepcopy(payload)
    duplicate["records"].append(deepcopy(duplicate["records"][0]))
    with pytest.raises(SystemExit, match="duplicate lineage registry identities"):
        validate_registry(duplicate)

    stale = deepcopy(payload)
    stale["records"][0]["name"] = "stale_legacy_identity"
    with pytest.raises(SystemExit, match="registry/catalog mismatch"):
        validate_registry(stale)

    ambiguous = deepcopy(payload)
    first = ambiguous["records"][0]
    second = next(row for row in ambiguous["records"] if row["pipeline_family"] != first["pipeline_family"])
    second["reproduce_notebook"] = first["reproduce_notebook"]
    with pytest.raises(SystemExit, match="multiple pipeline families share primary notebook"):
        validate_registry(ambiguous)

    over_cap = deepcopy(payload)
    for index, row in enumerate(over_cap["records"][:21]):
        row["reproduce_notebook"] = f"notebooks/reproduce/overflow_{index:02d}.ipynb"
    with pytest.raises(SystemExit, match="family notebook count outside 1..20"):
        validate_registry(over_cap)

    cross_family = deepcopy(payload)
    reassigned = next(row for row in cross_family["records"] if row["pipeline_family"] == "cellosaurus")
    reassigned["pipeline_family"] = "depmap"
    reassigned["source_family"] = "depmap"
    reassigned["reproduce_notebook"] = "notebooks/reproduce/depmap_cell_context.ipynb"
    with pytest.raises(SystemExit, match="authoritative pipeline family mismatch"):
        validate_registry(cross_family)

    wrong_producer = deepcopy(payload)
    wrong_producer["records"][0]["producer"] = "manage_db/export_kg.py"
    wrong_producer["records"][0]["producer_builder"] = "manage_db/export_kg.py"
    with pytest.raises(SystemExit, match="authoritative producer mismatch"):
        validate_registry(wrong_producer)

    wrong_command = deepcopy(payload)
    wrong_command["records"][0]["full_worker_rebuild_command"] = "uv run python -m manage_db.export_kg --task <task-id>"
    wrong_command["records"][0]["rebuild_command_evidenced"] = True
    with pytest.raises(SystemExit, match="authoritative rebuild command mismatch"):
        validate_registry(wrong_command)

    wrong_notebook = deepcopy(payload)
    wrong_notebook["records"][0]["reproduce_notebook"] = "notebooks/reproduce/wrong_same_family.ipynb"
    with pytest.raises(SystemExit, match="authoritative primary notebook mismatch"):
        validate_registry(wrong_notebook)

    contradictory_command_flag = deepcopy(payload)
    no_command = next(row for row in contradictory_command_flag["records"] if row["full_worker_rebuild_command"] is None)
    no_command["rebuild_command_evidenced"] = True
    with pytest.raises(SystemExit, match="rebuild command evidence flag mismatch"):
        validate_registry(contradictory_command_flag)


def test_producer_and_rebuild_command_are_exact_per_output_not_family_wide() -> None:
    records = {identity(row): row for row in registry()["records"]}
    assert records[("nodes", "cell_line")]["producer"] is None
    assert records[("features", "cell_line_textual_summary")]["producer"] == "manage_db/build_textual_summary_features.py"
    assert records[("features", "cell_line_textual_summary")]["full_worker_rebuild_command"]
    assert records[("nodes", "gene")]["producer"] is None
    assert records[("features", "protein_sequence")]["producer"] == "manage_db/build_sequence_features.py"
    assert records[("features", "protein_sequence")]["full_worker_rebuild_command"] is None
    assert records[("features", "transcript_sequence")]["full_worker_rebuild_command"]
    assert records[("edges", "pathway_contains_gene")]["producer"] is None
    assert records[("evidence", "gene_ortholog_gene")]["producer"] is None
    assert records[("edges", "tissue_expresses_protein")]["producer"] is None
    assert records[("embedding", "gene_genomic_sequence_nucleotide_transformer_v2_50m_multi_species")]["producer"] is None
    assert records[("embedding", "molecule_smiles_chemberta_77m_mlm")]["producer"] is None
    assert records[("embedding", "gene_text_sbiobert_snli_multinli_stsb")]["producer"] == "manage_db/build_real_embeddings.py"
    assert records[("nodes", "disease")]["full_worker_rebuild_command"] is None


def test_records_are_complete_conservative_and_reconcile_catalog_and_receipts() -> None:
    catalog = {identity(row): row for row in json.loads(CATALOG_PATH.read_text())["datasets"]}
    counts = Counter()
    gaps = []
    for row in registry()["records"]:
        assert REQUIRED_RECORD_FIELDS <= row.keys(), identity(row)
        assert row["reproducibility_status"] in STATUSES
        counts[row["reproducibility_status"]] += 1
        if row["reproducibility_status"] == "provenance-gap":
            gaps.append(identity(row))
            assert row["provenance_gaps"]
        source = catalog[identity(row)]
        assert row["canonical_uri"] == source["uri"]
        assert row["fields"] == source["fields"]
        assert row["keys"] == source["keys"]
        assert row["qc"]["rows"] == source["rows"]
        assert row["qc"]["schema_hash"] == source["schema_hash"]
        assert row["qc"]["generation"] == source["objects"][0]["generation"]
        assert row["migration_receipt"]["verified"] is True
        assert row["migration_receipt"]["destination_generation"] == row["qc"]["generation"]
        if row["reproducibility_status"] == "fully-replayable":
            assert row["native_inputs"] and row["producer_builder"]
            assert row["full_worker_rebuild_command"] and row["rebuild_command_evidenced"]
    assert counts == {
        "documented-not-replayed": 6,
        "historical-builder-only": 94,
        "provenance-gap": 10,
    }
    assert gaps


def test_tracked_links_and_evidenced_builders_exist() -> None:
    for row in registry()["records"]:
        assert (ROOT / row["catalog_page"]).is_file()
        for link in row["links"]:
            if link.startswith(("docs/", "reproduce/", "manage_db/", "tests/")):
                assert (ROOT / link).exists(), (identity(row), link)
        builder = row["producer_builder"]
        if builder is not None:
            assert (ROOT / builder).is_file(), (identity(row), builder)
        command = row["full_worker_rebuild_command"]
        assert row["rebuild_command_evidenced"] is (command is not None)
        if command:
            assert command.startswith("uv run python -m ")
            assert "<task-id>" in command


def test_generated_path_set_and_bytes_are_exact() -> None:
    payload = registry()
    grouped = family_records(payload)
    expected_paths = {ROOT / path for path in grouped}
    assert set(NOTEBOOK_DIR.glob("*.ipynb")) == expected_paths
    assert (NOTEBOOK_DIR / "README.md").read_text() == build_readme(payload)
    for relative_path, rows in grouped.items():
        path = ROOT / relative_path
        assert path.read_text() == nbformat.writes(build_family_notebook(rows)), path
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        assert all(cell.execution_count is None and cell.outputs == [] for cell in notebook.cells if cell.cell_type == "code")


def test_notebooks_have_required_sections_and_no_machine_specific_or_secret_material() -> None:
    required_sections = {
        "Objective and meaning",
        "Native inputs, release, acquisition and environment",
        "Expected schema and identifiers",
        "Keys, mappings and joins",
        "Cleaning, normalization, transformations and filters",
        "Deduplication and assertion/evidence semantics",
        "Quarantines, exclusions and missing data",
        "Problems encountered and decisions",
        "Producer / builder",
        "Full worker rebuild command",
        "Migration receipt",
        "QC, bounded replay and verification",
        "Reproducibility limits",
        "Linked code, tests, reports and historical notebooks",
    }
    forbidden = (
        "/Users/",
        "/home/",
        "/mnt/gcs",
        "jkobject@gmail.com",
        "JOUVENCE_BILLING_PROJECT=",
        "GOOGLE_APPLICATION_CREDENTIALS=",
        "BEGIN PRIVATE KEY",
    )
    for path in NOTEBOOK_DIR.glob("*.ipynb"):
        notebook = nbformat.read(path, as_version=4)
        markdown = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "markdown")
        code = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
        assert all(f"## {section}" in markdown for section in required_sections), path
        assert "## Canonical outputs owned by this notebook" in markdown, path
        assert markdown.count("### Output `") >= 1, path
        assert "shared evidenced producer" not in markdown.lower(), path
        assert not any(token in markdown or token in code for token in forbidden), path
        assert not re.search(r"(?i)(password|api[_-]?key|secret)\s*=\s*['\"][^'\"]+", code), path
        assert "subprocess" not in code and "os.system" not in code
        assert "JOUVENCE_LIVE_GCS" in code and "requester_pays" in code
        assert "version_aware=True" in code and "pinned_path" in code
        assert "assert " not in code


def test_representative_notebook_from_every_layer_executes_offline(tmp_path: Path) -> None:
    selected = {
        "txgnn_legacy_bundle.ipynb",  # nodes, edges, evidence, features + provenance gaps
        "opentargets_associations.ipynb",  # nodes, edges, evidence
        "ensembl_identity_and_sequence.ipynb",  # nodes, edges, features
        "text_embeddings.ipynb",  # embedding
        "molecule_fingerprints.ipynb",  # documented-not-replayed
    }
    for name in selected:
        notebook = nbformat.read(NOTEBOOK_DIR / name, as_version=4)
        executed = NotebookClient(
            notebook,
            timeout=120,
            kernel_name="python3",
            resources={"metadata": {"path": str(tmp_path)}},
        ).execute()
        code_cells = [cell for cell in executed.cells if cell.cell_type == "code"]
        assert all(cell.execution_count is not None for cell in code_cells), name
        assert any(output.get("data", {}).get("text/plain", "").find("SKIPPED") >= 0 for output in code_cells[-1].outputs), name


def test_generator_check_passes_in_subprocess() -> None:
    result = subprocess.run(
        ["uv", "run", "python", "reproduce/generate_parquet_reproduction_notebooks.py", "--check"],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
