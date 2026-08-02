#!/usr/bin/env python3
"""Validate and optionally execute the Jouvence data inventory explorer."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "07_data_inventory_explorer.ipynb"
REQUIRED = [
    "canonical-observed",
    "canonical-inferred",
    "non-canonical",
    "edges_inferred",
    "evidence_inferred",
    "JOUVENCE_BILLING_PROJECT",
    "live-gcs-only",
    "read_bounded_parquet",
    "list_parquet_uris",
    "server-side cap",
    "parquet_footer",
    "embedding_projection",
    "stable join",
    "schema_relations_absent_from_kg",
    "iter_relation_minibatches",
    "iter_kg_minibatches",
]
FORBIDDEN = [
    "/Users/jkobject/mnt/gcs",
    "jkobject-1549353370965",
    "GOOGLE_APPLICATION_CREDENTIALS=",
    "pd.read_parquet(SELECTED_URI)",
    "fs.glob(",
    "build_public_fixture",
    "kg-fixture",
    "fixture_rule_engine",
    "JOUVENCE_DATA_MODE",
    "build_pyg_export",
    "full_graph.pt",
    "pickle.load",
]


def check() -> dict[str, object]:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    text = "\n".join(str(cell.source) for cell in notebook.cells)
    failures: list[str] = []
    metadata = notebook.metadata.get("jouvence", {})
    for key in ("bounded", "read_only"):
        if metadata.get(key) is not True:
            failures.append(f"metadata.jouvence.{key} must be true")
    for phrase in REQUIRED:
        if phrase not in text:
            failures.append(f"missing required phrase: {phrase}")
    for token in FORBIDDEN:
        if token in text:
            failures.append(f"forbidden token: {token}")
    if len(notebook.cells) < 18:
        failures.append("explorer is missing meaningful workflow sections")
    selector_text = "\n".join(str(cell.source) for cell in notebook.cells if cell.cell_type == "code")
    if "SELECTED_LAYER" not in selector_text or "SELECTED_NAME" not in selector_text:
        failures.append("missing editable layer/name selector")
    for cell in notebook.cells:
        if cell.cell_type == "code" and (cell.get("outputs") or cell.get("execution_count") is not None):
            failures.append("committed notebook contains execution output")
            break
    return {"cells": len(notebook.cells), "failures": failures, "status": "pass" if not failures else "fail"}


def execute() -> str:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=180,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
    )
    client.execute()
    destination = Path(tempfile.mkdtemp(prefix="jouvence-data-explorer-")) / NOTEBOOK.name
    nbformat.write(notebook, destination)
    return str(destination)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    report = check()
    if args.execute and report["status"] == "pass":
        report["executed_copy"] = execute()
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
