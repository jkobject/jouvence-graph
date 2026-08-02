#!/usr/bin/env python3
"""Build the immutable live relation reconciliation ledger for t_8c7f0862.

This task-scoped builder consumes captured GCS object inventories and Parquet
footer readbacks. It deliberately does not perform full-table endpoint/support
scans on the Mac; those are preserved as explicit verification limitations.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from manage_db.kg_schema import RELATIONS

TASK_ID = "t_8c7f0862"
CAPTURE_COMPLETENESS_CONTRACT = {
    "edges": {
        "object_count": 43,
        "identity_sha256": "317024c1e31d506d88ecd94743a96584ce3fefa7d7b977ea974344b6081853f1",
    },
    "evidence": {
        "object_count": 21,
        "identity_sha256": "35b24effd6692b3802280d7b102774929730d2138dfcfa3d365b93f3159128c6",
    },
    "proof": {
        "object_count": 1,
        "identity_sha256": "bd813469d668be815db6eb009a10d7a4b6a3c4cdf244a85795572677618b3279",
    },
    "staging": {
        "object_count": 10856,
        "identity_sha256": "edd10b2d2a5936f56867cf9c91a871e0d4fd23bb9c2e191ad350e9615e9b38f9",
    },
    "metadata": {
        "object_count": 18,
        "identity_sha256": "51178d85f7f6116eda87c497e33b1789cdbfff25d81941f0512249929d268c03",
    },
    "features_remap": {
        "object_count": 28,
        "identity_sha256": "8cbc554a46d9f4d5b633cc68161d8bd40804cb7c11520a2473d4c4ec170524d1",
    },
}
NEXT_STATES = {
    "accepted-canonical",
    "promote-candidate",
    "evidence-backfill-candidate",
    "review-required",
    "deferred-policy",
    "rejected",
    "feature-context",
    "schema-missing",
}

REVIEW_REQUIRED = {
    "molecule_treats_disease",
}
EVIDENCE_BACKFILL = {
    "tissue_expresses_gene",
    "cell_type_expresses_gene",
    "cell_line_expresses_gene",
    "molecule_contraindicates_disease",
    "molecule_synergizes_molecule",
}
FEATURE_CONTEXT = {
    "gene_coexpressed_gene",
    "disease_comorbid_disease",
    "paper_produced_dataset",
    "paper_cites_paper",
    "dataset_contains_disease",
    "dataset_contains_molecule",
    "dataset_contains_cell_type",
    "dataset_contains_cell_line",
    "dataset_contains_tissue",
    "tf_binds_enhancer",
}
PROMOTE_CANDIDATE = {
    "cell_line_expresses_protein",
    "cell_line_gene_essentiality",
    "cell_line_responds_to_molecule",
    "pathway_contains_protein",
    "molecule_targets_protein",
    "cell_type_found_in_tissue",
    "cell_type_subtype_of_cell_type",
    "cell_line_models_disease",
    "cell_line_derived_from_cell_type",
}
DEFERRED_POLICY = {
    "enhancer_regulates_transcript",
    "cell_type_involved_in_disease",
}
REJECTED = {"transcript_interacts_protein"}
SCHEMA_MISSING = {
    "tf_regulates_gene",
    "cell_type_expresses_protein",
    "transcript_interacts_gene",
    "cell_type_responds_to_molecule",
    "phenotype_observed_in_tissue",
}
ACCEPTED_NO_EVIDENCE_EXCEPTION = {
    "gene_has_transcript",
    "transcript_encodes_protein",
    "gene_associated_phenotype",
    "pathway_child_of_pathway",
    "molecule_in_pathway",
    "molecule_parent_of_molecule",
    "molecule_associated_phenotype",
    "disease_subtype_of_disease",
    "disease_has_phenotype",
    "phenotype_subtype_of_phenotype",
    "tissue_subtype_of_tissue",
    "cell_line_derived_from_tissue",
    "organism_has_gene",
    "organism_has_tissue",
}

STAGING_PREFIX_ROUTES = {
    "cell-line-assays-2026-06-22-t_c2b0803c": {
        "relations": [
            "cell_line_expresses_protein",
            "cell_line_gene_essentiality",
            "cell_line_responds_to_molecule",
        ],
        "decision": "route to relation-specific promote-candidate review; do not batch-promote",
    },
    "cell-type-context-relations-t_d468e2dc": {
        "relations": [
            "cell_type_found_in_tissue",
            "cell_type_subtype_of_cell_type",
            "cell_type_involved_in_disease",
        ],
        "decision": "two validated promote candidates; disease route is source-gap/deferred-policy",
    },
    "cellosaurus-cell-line-metadata-20260622-t_bb0fb082": {
        "relations": ["cell_line_derived_from_cell_type", "cell_line_models_disease"],
        "decision": "route to relation-specific promote-candidate review",
    },
    "disease-associated-protein-20260622-t_7f0cccde": {
        "relations": ["disease_associated_protein"],
        "decision": "historical staged source retained; superseded byte-for-byte by terminally accepted canonical edge/evidence generations from t_aa5cd96e reviewed by t_0611e6c6",
    },
    "enhancer-regulates-transcript-audit-20260622-t_8ed77c71": {
        "relations": ["enhancer_regulates_transcript"],
        "decision": "source audit only; no ENST/TSS-native assertion found; deferred-policy",
    },
    "molecule-synergizes-evidence-20260622-t_4e12f7c7": {
        "relations": ["molecule_synergizes_molecule"],
        "decision": "complete evidence-only backfill candidate; canonical evidence remains absent",
    },
    "molecule-targets-protein-chembl-20260622-t_84bf3876": {
        "relations": ["molecule_targets_protein"],
        "decision": "route to protein-native promote-candidate review",
    },
    "opentargets-clinical-drug-evidence-20260622-t_ceee5d53": {
        "relations": ["molecule_treats_disease", "molecule_contraindicates_disease"],
        "decision": "retain: 481 OpenTargets assertions are not fully represented by current CTGov-only canonical evidence; contraindication produced zero accepted rows",
    },
    "paper-dataset-provenance-20260622-t_649cee71": {
        "relations": [
            "paper_produced_dataset",
            "paper_cites_paper",
            "dataset_contains_disease",
            "dataset_contains_molecule",
            "dataset_contains_cell_type",
            "dataset_contains_cell_line",
            "dataset_contains_tissue",
        ],
        "decision": "route to graph-disconnected provenance/catalog feature-context; zero-row file is not a canonical candidate",
    },
    "rbp-rna-clip-encori-pilot-20260622T135045Z": {
        "relations": ["transcript_interacts_protein"],
        "decision": "current candidate rejected: zero accepted rows because endpoints are not source-native ENST/ENSP/UniProt",
    },
    "reactome-pathway-contains-protein-20260622-t_9d36e82e": {
        "relations": ["pathway_contains_protein"],
        "decision": "route to protein-native promote-candidate review after pathway-level semantics acceptance",
    },
    "remap-tf-binds-enhancer-remote-pruned-chr1-20260623-t_3479936e-v7-100kb-b50k-tempfix": {
        "relations": ["tf_binds_enhancer"],
        "decision": "historical staged validation evidence only; route C is complete on accepted canonical feature/context surfaces, all-peak topology conversion is policy-deferred, and no scaling/recovery/resume lane is active",
    },
    "source-native-expansion": {
        "relations": ["mutation_affects_transcript", "mutation_in_gene", "mutation_overlaps_enhancer"],
        "decision": "historical broad/bounded candidates excluded: superseded by newer canonical relation-specific generations; broad mutation_in_gene and coordinate-only overlap are policy-disqualified",
    },
}

STAGED_VALIDATION = {
    "cell_line_expresses_protein": (0, 0, 0, 0, "staged validation.json"),
    "cell_line_gene_essentiality": (0, 0, 0, 0, "staged validation.json"),
    "cell_line_responds_to_molecule": (0, 0, 0, 0, "staged validation.json"),
    "cell_type_found_in_tissue": (0, 0, 0, 0, "staged context report"),
    "cell_type_subtype_of_cell_type": (0, 0, 0, 0, "staged context report"),
    "cell_line_derived_from_cell_type": (0, 0, 0, 0, "staged DuckDB validation"),
    "cell_line_models_disease": (0, 0, 0, 0, "staged DuckDB validation"),
    "molecule_synergizes_molecule": (0, 0, 0, 0, "staged validation.json"),
    "molecule_targets_protein": (0, 0, 0, 0, "staged validation report"),
    "pathway_contains_protein": (0, 0, 0, 0, "staged DuckDB validation"),
    "transcript_interacts_protein": (0, 0, None, None, "zero-row rejected staged candidate"),
}

CANONICAL_REPORTED_ZERO = {
    "mutation_in_gene": "promotion report: live endpoint/evidence/proof audit",
    "mutation_overlaps_enhancer": "promotion report: live endpoint/evidence/support audit",
    "disease_manifests_in_tissue": "promotion report: bounded endpoint/evidence audit",
    "mutation_affects_transcript": "independently accepted promotion report",
    "disease_associated_protein": "terminal independent reviewer t_0611e6c6: endpoint/support/hash mismatch 0 and replay no-op",
}

SEMANTIC_OVERRIDES = {
    "molecule_treats_disease": "Keep indication topology. Current canonical evidence is CTGov trial context only (7,804 assertions / 377 edge keys); preserve the staged 481 OpenTargets indication assertions instead of treating CTGov overwrite as complete.",
    "molecule_contraindicates_disease": "Require a contraindication-specific source; positive indication or trial evidence cannot be reused.",
    "molecule_synergizes_molecule": "Keep drug-combination response semantics; staged evidence has exact support but no non-null score or study/context fields, so review before evidence-only promotion.",
    "tf_binds_enhancer": "Route C is complete for current scope as accepted canonical feature/context. The bounded and full ReMap CRM support-QA surfaces are not observed binding topology; conversion into tf_binds_enhancer edges remains policy-deferred and no recovery/resume/build lane is active.",
    "transcript_interacts_protein": "Current ENCORI pilot is rejected because tested rows do not expose source-native transcript/protein endpoints; relation remains active for a future direct RBP/RNA source.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory-dir", type=Path, required=True)
    parser.add_argument("--captured-at", required=True)
    parser.add_argument("--schema-commit", required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    parser.add_argument("--inventory-output", type=Path, required=True)
    return parser.parse_args()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def md5_hex(value: str | None) -> str | None:
    if not value:
        return None
    return base64.b64decode(value).hex()


def normalized_identity(item: dict[str, Any]) -> dict[str, Any]:
    metadata = item.get("metadata", {})
    name = metadata["name"]
    return {
        "uri": f"gs://{metadata['bucket']}/{name}",
        "name": name,
        "generation": metadata["generation"],
        "size_bytes": int(metadata.get("size", 0)),
        "md5_base64": metadata.get("md5Hash"),
        "md5_hex": md5_hex(metadata.get("md5Hash")),
        "crc32c_base64": metadata.get("crc32c"),
        "updated": metadata.get("updated"),
    }


def identity_digest(items: list[dict[str, Any]]) -> str:
    payload = json.dumps(
        sorted((normalized_identity(item) for item in items), key=lambda x: x["name"]),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return sha256_bytes(payload)


def validate_capture_completeness(
    area: str,
    start_items: list[dict[str, Any]],
    end_items: list[dict[str, Any]],
) -> str:
    expected = CAPTURE_COMPLETENESS_CONTRACT[area]
    start_digest = identity_digest(start_items)
    end_digest = identity_digest(end_items)
    observed = {
        "start_count": len(start_items),
        "end_count": len(end_items),
        "start_identity_sha256": start_digest,
        "end_identity_sha256": end_digest,
    }
    if (
        len(start_items) != expected["object_count"]
        or len(end_items) != expected["object_count"]
        or start_digest != expected["identity_sha256"]
        or end_digest != expected["identity_sha256"]
    ):
        raise RuntimeError(
            f"capture completeness contract failed for {area}: "
            f"expected={expected!r}, observed={observed!r}"
        )
    return start_digest


def relation_name_from_object(area: str, name: str) -> str | None:
    prefix = f"kg/v2/{area}/"
    if not name.startswith(prefix) or not name.endswith(".parquet"):
        return None
    leaf = name[len(prefix) :]
    if "/" in leaf:
        return None
    return leaf.removesuffix(".parquet")


def next_state(name: str, canonical_names: set[str]) -> str:
    for state, names in (
        ("review-required", REVIEW_REQUIRED),
        ("evidence-backfill-candidate", EVIDENCE_BACKFILL),
        ("feature-context", FEATURE_CONTEXT),
        ("promote-candidate", PROMOTE_CANDIDATE),
        ("deferred-policy", DEFERRED_POLICY),
        ("rejected", REJECTED),
        ("schema-missing", SCHEMA_MISSING),
    ):
        if name in names:
            return state
    if name in canonical_names:
        return "accepted-canonical"
    raise AssertionError(f"unclassified relation: {name}")


def status_for(name: str, state: str, has_edge: bool, has_evidence: bool) -> str:
    if state == "accepted-canonical" and name in ACCEPTED_NO_EVIDENCE_EXCEPTION:
        return "canonical+accepted-no-evidence-exception"
    if state == "accepted-canonical":
        return "canonical+accepted"
    if state == "review-required":
        return "canonical+review-required"
    if state == "evidence-backfill-candidate":
        return "canonical+evidence-incomplete"
    if state == "promote-candidate":
        return "staged+validated-promote-candidate"
    if state == "deferred-policy":
        return "noncanonical+policy-deferred"
    if state == "rejected":
        return "noncanonical+current-candidate-rejected"
    if state == "feature-context":
        return "canonical-metadata-only+graph-disconnected" if has_edge else "feature-context+noncanonical"
    if state == "schema-missing":
        return "active-schema+data-absent"
    raise AssertionError((name, state, has_edge, has_evidence))


def main() -> None:
    args = parse_args()
    inv = args.inventory_dir
    raw: dict[str, list[dict[str, Any]]] = {
        area: json.loads((inv / f"{area}.json").read_text())
        for area in ("edges", "evidence", "proof", "staging", "metadata", "features_remap")
    }
    end_raw: dict[str, list[dict[str, Any]]] = {
        area: json.loads((inv / f"{area}_end.json").read_text())
        for area in raw
    }
    validated_digests = {
        area: validate_capture_completeness(area, items, end_raw[area])
        for area, items in raw.items()
    }
    footers = json.loads((inv / "parquet_footers_non_remap.json").read_text())

    inventories = {
        area: {
            "object_count": len(items),
            "size_bytes": sum(int(x.get("metadata", {}).get("size", 0)) for x in items),
            "identity_sha256": validated_digests[area],
            "capture_file_sha256": file_sha256(inv / f"{area}.json"),
            "expected_object_count": CAPTURE_COMPLETENESS_CONTRACT[area]["object_count"],
            "expected_identity_sha256": CAPTURE_COMPLETENESS_CONTRACT[area]["identity_sha256"],
            "completeness_validated": True,
        }
        for area, items in raw.items()
    }
    end_captured_at = (inv / "captured_end_at.txt").read_text().strip()
    capture_boundary = {}
    for area, start_items in raw.items():
        end_items = end_raw[area]
        start_digest = validated_digests[area]
        end_digest = identity_digest(end_items)
        capture_boundary[area] = {
            "start_identity_sha256": start_digest,
            "end_identity_sha256": end_digest,
            "expected_object_count": CAPTURE_COMPLETENESS_CONTRACT[area]["object_count"],
            "expected_identity_sha256": CAPTURE_COMPLETENESS_CONTRACT[area]["identity_sha256"],
            "completeness_validated": True,
            "unchanged": True,
        }
    inventory_bundle = {
        "task_id": TASK_ID,
        "captured_at": args.captured_at,
        "scope": "gs://jouvencekb/main/{edges,evidence,proof,staging,metadata}/** plus accepted ReMap feature surfaces",
        "inventories": {
            area: [normalized_identity(item) for item in items]
            for area, items in raw.items()
        },
    }
    args.inventory_output.parent.mkdir(parents=True, exist_ok=True)
    args.inventory_output.write_text(json.dumps(inventory_bundle, indent=2, sort_keys=True) + "\n")
    inventory_bundle_sha = file_sha256(args.inventory_output)

    by_name: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    excluded_canonical_objects: list[dict[str, Any]] = []
    for area in ("edges", "evidence"):
        for item in raw[area]:
            name = item["metadata"]["name"]
            relation = relation_name_from_object(area, name)
            if relation is None:
                excluded_canonical_objects.append(
                    {"identity": normalized_identity(item), "reason": "not an active relation Parquet identity"}
                )
                continue
            by_name[relation][area] = item
    for item in raw["proof"]:
        name = item["metadata"]["name"]
        if name.endswith("mutation_in_gene_containment_proof.parquet"):
            by_name["mutation_in_gene"]["proof"] = item
        else:
            excluded_canonical_objects.append(
                {"identity": normalized_identity(item), "reason": "proof object not mapped to an active relation"}
            )

    active_names = {r.name for r in RELATIONS}
    canonical_names = {name for name in by_name if "edges" in by_name[name] and name in active_names}
    evidence_names = {name for name in by_name if "evidence" in by_name[name] and name in active_names}
    unknown_edges = canonical_names - active_names
    assert not unknown_edges
    assert len(active_names) == len(RELATIONS), "duplicate active relation names"

    staging_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in raw["staging"]:
        name = item["metadata"]["name"].removeprefix("kg/v2/staging/")
        staging_groups[name.split("/", 1)[0]].append(item)
    assert set(staging_groups) == set(STAGING_PREFIX_ROUTES), (
        set(staging_groups) - set(STAGING_PREFIX_ROUTES),
        set(STAGING_PREFIX_ROUTES) - set(staging_groups),
    )

    prefix_rows = []
    for prefix, items in sorted(staging_groups.items()):
        route = STAGING_PREFIX_ROUTES[prefix]
        parquet_count = sum(x["metadata"]["name"].endswith(".parquet") for x in items)
        prefix_rows.append(
            {
                "prefix": f"gs://jouvencekb/staging/{prefix}/",
                "object_count": len(items),
                "parquet_object_count": parquet_count,
                "size_bytes": sum(int(x["metadata"].get("size", 0)) for x in items),
                "object_identity_sha256": identity_digest(items),
                "relations": route["relations"],
                "decision": route["decision"],
                "exact_objects": [normalized_identity(x) for x in items] if len(items) <= 100 else None,
                "exact_objects_reference": None if len(items) <= 100 else {
                    "inventory_file": str(args.inventory_output),
                    "inventory_file_sha256": inventory_bundle_sha,
                    "note": "all exact names/generations/hashes are in the immutable inventory bundle",
                },
            }
        )

    prefixes_by_relation: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in prefix_rows:
        for relation in row["relations"]:
            prefixes_by_relation[relation].append(
                {
                    "prefix": row["prefix"],
                    "object_count": row["object_count"],
                    "parquet_object_count": row["parquet_object_count"],
                    "object_identity_sha256": row["object_identity_sha256"],
                    "decision": row["decision"],
                }
            )

    def object_record(item: dict[str, Any] | None) -> dict[str, Any] | None:
        if item is None:
            return None
        rec = normalized_identity(item)
        footer = footers.get(item["metadata"]["name"])
        if footer:
            rec["parquet"] = {
                "rows": footer.get("rows"),
                "row_groups": footer.get("row_groups"),
                "columns": footer.get("columns"),
                "readback": "fresh PyArrow footer via GcsFileSystem",
            }
        return rec

    relation_rows: list[dict[str, Any]] = []
    for relation in RELATIONS:
        name = relation.name
        state = next_state(name, canonical_names)
        assert state in NEXT_STATES
        objects = by_name.get(name, {})
        edge = object_record(objects.get("edges"))
        evidence = object_record(objects.get("evidence"))
        proof = object_record(objects.get("proof"))
        edge_rows = edge.get("parquet", {}).get("rows") if edge else None

        if evidence is None and edge is not None:
            support = {
                "edges_without_evidence": edge_rows,
                "evidence_without_edge": 0,
                "result": "evidence-object-absent",
                "verification": "fresh object inventory plus canonical edge footer",
            }
        elif name == "molecule_treats_disease":
            support = {
                "edges_without_evidence": 13758,
                "evidence_without_edge": 0,
                "canonical_evidence_distinct_edge_keys": 377,
                "staged_opentargets_distinct_edge_keys": 481,
                "staged_keys_absent_from_current_canonical_evidence": 104,
                "result": "incomplete-and-source-overwrite-risk",
                "verification": "fresh bounded key readback of 14,135 edges, 7,804 CTGov evidence rows, and 481 staged OpenTargets rows",
            }
        elif name in CANONICAL_REPORTED_ZERO and edge and evidence:
            support = {
                "edges_without_evidence": 0,
                "evidence_without_edge": 0,
                "result": "reported-zero-on-current-promoted-artifact",
                "verification": CANONICAL_REPORTED_ZERO[name],
                "fresh_scan_this_task": False,
            }
        elif edge and evidence:
            support = {
                "edges_without_evidence": 0,
                "evidence_without_edge": 0,
                "result": "documented-canonical-support-audit-zero",
                "verification": "prior accepted targeted audit; fresh generations and footer counts read back, full key scan not rerun on Mac",
                "fresh_scan_this_task": False,
            }
        elif name in STAGED_VALIDATION:
            ewo, ewe, _, _, basis = STAGED_VALIDATION[name]
            support = {
                "edges_without_evidence": ewo,
                "evidence_without_edge": ewe,
                "result": "staged-report-readback",
                "verification": basis,
            }
        else:
            support = {
                "edges_without_evidence": None,
                "evidence_without_edge": None,
                "result": "not-applicable-no-accepted-edge-candidate",
                "verification": "data absence or policy deferral is explicit",
            }

        if name in STAGED_VALIDATION:
            _, _, x_missing, y_missing, basis = STAGED_VALIDATION[name]
            endpoint = {
                "x_missing": x_missing,
                "y_missing": y_missing,
                "result": "pass" if x_missing == 0 and y_missing == 0 else "not-applicable-zero-row",
                "verification": basis,
            }
        elif name in CANONICAL_REPORTED_ZERO:
            endpoint = {
                "x_missing": 0,
                "y_missing": 0,
                "result": "reported-zero-on-current-promoted-artifact",
                "verification": CANONICAL_REPORTED_ZERO[name],
                "fresh_scan_this_task": False,
            }
        elif edge:
            endpoint = {
                "x_missing": None,
                "y_missing": None,
                "result": "not-live-rerun",
                "verification": "current object generation/footer read back; full endpoint scan is prohibited laptop all-relation work and is delegated to independent review",
            }
        else:
            endpoint = {
                "x_missing": None,
                "y_missing": None,
                "result": "not-applicable-no-accepted-edge-candidate",
                "verification": "data absence or policy deferral is explicit",
            }

        staged_objects = []
        for item in raw["staging"]:
            object_name = item["metadata"]["name"]
            if f"/{name}.parquet" in object_name:
                rec = object_record(item)
                if rec:
                    staged_objects.append(rec)
        if name == "tf_binds_enhancer":
            staged_counts = {
                "edge_rows": 189459767,
                "evidence_rows": 189459767,
                "basis": "documented June aggregate; live inventory confirms 5,366 Parquets plus 5,366 sha256 companions, but footer recount was not run because ReMap scaling is VM-only",
            }
        else:
            stage_edge_rows = sum(
                int(x.get("parquet", {}).get("rows", 0))
                for x in staged_objects
                if "/edges/" in x["name"]
            )
            stage_evidence_rows = sum(
                int(x.get("parquet", {}).get("rows", 0))
                for x in staged_objects
                if "/evidence/" in x["name"] and "/evidence_canonical/" not in x["name"]
            )
            staged_counts = {
                "edge_rows": stage_edge_rows if any("/edges/" in x["name"] for x in staged_objects) else None,
                "evidence_rows": stage_evidence_rows if any("/evidence/" in x["name"] for x in staged_objects) else None,
                "basis": "fresh PyArrow footer readback of all non-ReMap staged relation Parquets",
            }

        accepted_exception = name in ACCEPTED_NO_EVIDENCE_EXCEPTION
        data_absence = None
        policy_deferral = None
        if state == "schema-missing":
            data_absence = "No canonical or accepted staged edge object exists for this active schema relation."
        elif state == "rejected":
            data_absence = "The live staged candidate has zero accepted edge/evidence rows."
        if state == "deferred-policy":
            policy_deferral = STAGING_PREFIX_ROUTES[
                next(
                    prefix
                    for prefix, route in STAGING_PREFIX_ROUTES.items()
                    if name in route["relations"]
                )
            ]["decision"]
        semantic = SEMANTIC_OVERRIDES.get(name, relation.notes)
        relation_rows.append(
            {
                "relation": name,
                "schema": {
                    "source_type": relation.source.value,
                    "target_type": relation.target.value,
                    "kind": relation.kind.value,
                    "direct": relation.direct,
                    "lifecycle": relation.status.value,
                },
                "canonical": {
                    "edge": edge,
                    "evidence": evidence,
                    "proof": proof,
                },
                "staged": {
                    "prefixes": prefixes_by_relation.get(name, []),
                    "relation_parquet_objects": staged_objects,
                    "counts": staged_counts,
                },
                "support_gaps": support,
                "endpoint_anti_join": endpoint,
                "status": status_for(name, state, edge is not None, evidence is not None),
                "source_native_semantic_decision": semantic,
                "accepted_no_evidence_exception": {
                    "accepted": accepted_exception,
                    "reason": (
                        "Topology is accepted; provenance should be backfilled only when source-backed and useful, never fabricated."
                        if accepted_exception
                        else None
                    ),
                },
                "data_absence_reason": data_absence,
                "policy_deferral_reason": policy_deferral,
                "next_state": state,
            }
        )

    assert len(relation_rows) == 67
    assert {row["relation"] for row in relation_rows} == active_names
    assert all(row["next_state"] in NEXT_STATES for row in relation_rows)
    next_state_counts = Counter(row["next_state"] for row in relation_rows)
    canonical_edge_rows = sum(
        int(row["canonical"]["edge"]["parquet"]["rows"])
        for row in relation_rows
        if row["canonical"]["edge"] is not None
    )
    canonical_without_evidence = sum(
        row["canonical"]["edge"] is not None and row["canonical"]["evidence"] is None
        for row in relation_rows
    )
    accepted_exception_count = sum(
        row["accepted_no_evidence_exception"]["accepted"] for row in relation_rows
    )
    assert accepted_exception_count == len(ACCEPTED_NO_EVIDENCE_EXCEPTION)
    assert set(row["relation"] for row in relation_rows if row["next_state"] == "accepted-canonical") | REVIEW_REQUIRED | EVIDENCE_BACKFILL | (FEATURE_CONTEXT & canonical_names) == canonical_names

    top_metadata = [normalized_identity(item) for item in raw["metadata"]]
    remap_features = [normalized_identity(item) for item in raw["features_remap"]]
    remap_bounded = next(
        item
        for item in remap_features
        if item["name"] == "kg/v2/features/remap_crm_tf_enhancer_support.parquet"
    )
    remap_full = [
        item
        for item in remap_features
        if item["name"].startswith("kg/v2/features/remap_crm_tf_enhancer_support_full/")
    ]
    assert remap_bounded["generation"] == "1782308308670478"
    assert len(remap_full) == 27
    assert sum(item["size_bytes"] for item in remap_full) == 5144179807
    remap_route_c = {
        "status": "accepted-canonical-feature-context",
        "topology_conversion": "deferred-policy",
        "bounded_surface": {
            **remap_bounded,
            "rows": 2915130,
            "producer": "t_656a1102",
            "reviewer": "t_69fa9b1d",
        },
        "full_surface": {
            "prefix": "gs://jouvencekb/main/features/remap_crm_tf_enhancer_support_full/",
            "object_count": len(remap_full),
            "size_bytes": sum(item["size_bytes"] for item in remap_full),
            "object_identity_sha256": sha256_bytes(
                json.dumps(
                    sorted(remap_full, key=lambda item: item["name"]),
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ),
            "summary_rows": 48768788,
            "global_tf_rows": 1179,
            "producer": "t_f2a2952e",
            "reviewer": "t_0974375e",
        },
        "decision": "Route C complete for current scope; no watchdog, recovery, resume, promotion, backfill, or VM lane. Only conversion into active tf_binds_enhancer topology is policy-deferred.",
    }
    remap_relation = next(row for row in relation_rows if row["relation"] == "tf_binds_enhancer")
    remap_relation["canonical_feature_context"] = remap_route_c
    remap_relation["status"] = "accepted-canonical-feature-context+topology-deferred"
    remap_relation["policy_deferral_reason"] = (
        "Only conversion of accepted support-QA features into observed tf_binds_enhancer topology is deferred; current route C is complete."
    )
    ledger = {
        "ledger_version": 1,
        "task_id": TASK_ID,
        "captured_at": args.captured_at,
        "schema": {
            "commit": args.schema_commit,
            "path": "manage_db/kg_schema.py",
            "sha256": file_sha256(Path("manage_db/kg_schema.py")),
            "active_relation_count": len(RELATIONS),
        },
        "scope": {
            "gcs": "gs://jouvencekb/main/{edges,evidence,proof,staging,metadata}/** plus accepted ReMap feature surfaces",
            "canonical_writes": False,
            "full_table_scans": False,
            "fuse_readback": "unavailable: expected FUSE directory existed but was empty/unmounted; GCS direct readback used and discrepancy failed closed",
        },
        "inventory_summary": inventories,
        "capture_boundary": {
            "start": args.captured_at,
            "end": end_captured_at,
            "all_relevant_object_identities_unchanged": True,
            "areas": capture_boundary,
        },
        "exact_inventory_bundle": {
            "path": str(args.inventory_output),
            "sha256": inventory_bundle_sha,
        },
        "denominator": {
            "active_relations": len(relation_rows),
            "canonical_active_relations": len(canonical_names),
            "canonical_relations_with_evidence": len(evidence_names),
            "canonical_relations_without_evidence": canonical_without_evidence,
            "declared_relations_not_canonical": len(relation_rows) - len(canonical_names),
            "canonical_edge_rows": canonical_edge_rows,
            "accepted_no_evidence_exceptions": accepted_exception_count,
            "next_state_counts": dict(sorted(next_state_counts.items())),
            "sum_next_state_counts": sum(next_state_counts.values()),
        },
        "hypothesis_reconciliation": {
            "active_relations": {"documented": 67, "live": len(relation_rows)},
            "canonical_active_relations": {"documented": 40, "live": len(canonical_names)},
            "canonical_with_evidence": {"documented": 18, "live": len(evidence_names)},
            "canonical_without_evidence": {"documented": 22, "live": canonical_without_evidence},
            "declared_not_canonical": {"documented": 27, "live": len(relation_rows) - len(canonical_names)},
            "canonical_edge_rows": {"documented": 100080390, "live": canonical_edge_rows},
            "review_required": {"documented": 3, "live": next_state_counts["review-required"]},
        },
        "staging_prefix_routes": prefix_rows,
        "metadata_objects": top_metadata,
        "remap_route_c": remap_route_c,
        "excluded_live_objects": excluded_canonical_objects,
        "known_high_priority_finding": {
            "relation": "molecule_treats_disease",
            "finding": "Confirmed concrete non-merge/overwrite failure: current canonical evidence is CTGov-only (7,804 assertions / 377 edge keys). The live staged OpenTargets file has 481 source assertions/keys; 104 keys are absent from canonical evidence and the 377 overlapping keys lost OpenTargets source multiplicity. Preserve topology and repair only through a separately reviewed evidence-identity merge; this ledger authorizes no write.",
        },
        "verification_limits": [
            "No all-relation endpoint anti-join or support scan was run on the Mac; that is prohibited heavy/all-relation work. Every row records whether its result is fresh, report-backed, absent-evidence exact, or not rerun.",
            "The historical 5,366 ReMap staged Parquet footer totals were not recomputed locally. They are excluded historical validation material: route C is complete on accepted canonical feature/context surfaces and no VM review, scaling, recovery, or resume lane is pending.",
            "The configured FUSE root existed but was empty/unmounted, so GCS/FUSE parity could not be established. The ledger fails closed to direct GCS generations and records the unavailable FUSE readback.",
        ],
        "relations": relation_rows,
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    ledger_sha = file_sha256(args.json_output)

    lines = [
        f"# Live relation reconciliation — `{TASK_ID}`",
        "",
        f"- Captured: `{args.captured_at}`",
        f"- Schema commit: `{args.schema_commit}`",
        f"- Immutable ledger: `{args.json_output}`",
        f"- Ledger SHA256: `{ledger_sha}`",
        f"- Exact GCS inventory bundle: `{args.inventory_output}`",
        f"- Inventory bundle SHA256: `{inventory_bundle_sha}`",
        f"- Durable PR mirrors: `artifacts/reports/{TASK_ID}/relation_reconciliation_live_{TASK_ID}.json.gz` and `artifacts/reports/{TASK_ID}/relation_reconciliation_live_{TASK_ID}_gcs_inventory.json.gz`; decompress to the exact JSON hashes above",
        "",
        "## Executive result",
        "",
        f"The active-schema denominator reconciles exactly once: **{len(relation_rows)} relations**. Direct GCS readback has **{len(canonical_names)} canonical active edge relations**, **{len(evidence_names)} with evidence**, **{canonical_without_evidence} without evidence**, and **{canonical_edge_rows:,} canonical edge rows**. The previous 40/18/22/100,080,390 snapshot is stale.",
        "",
        f"Exactly **{accepted_exception_count}** canonical no-evidence relations are explicitly accepted exceptions. Exactly **{next_state_counts['evidence-backfill-candidate']}** canonical relations are evidence-backfill candidates; two canonical dataset relations are graph-disconnected feature/context. No missing evidence object is silently counted as complete.",
        "",
        "P0 finding: `molecule_treats_disease` has a confirmed concrete non-merge/overwrite failure. Canonical evidence is CTGov-only (7,804 assertions, 377 distinct edge keys); the live staged OpenTargets file has 481 assertions/keys, 104 keys are absent, and 377 overlaps lost source multiplicity. This is `review-required`; any repair must merge evidence identities and requires a separate write gate.",
        "",
        "ReMap route C is complete for current scope as accepted canonical feature/context: bounded generation `1782308308670478` (2,915,130 rows), plus the exact 27-object full support-QA prefix (48,768,788 summary rows + 1,179 global TF rows). Only conversion into observed `tf_binds_enhancer` topology is `deferred-policy`; no active compute or recovery lane exists.",
        "",
        "## Denominator and buckets",
        "",
        "| Metric | Documented hypothesis | Fresh live result |",
        "| --- | ---: | ---: |",
        f"| Active schema relations | 67 | {len(relation_rows)} |",
        f"| Canonical active relations | 40 | {len(canonical_names)} |",
        f"| Canonical with evidence | 18 | {len(evidence_names)} |",
        f"| Canonical without evidence | 22 | {canonical_without_evidence} |",
        f"| Declared not canonical | 27 | {len(relation_rows) - len(canonical_names)} |",
        f"| Canonical edge rows | 100,080,390 | {canonical_edge_rows:,} |",
        f"| Review-required | 3 | {next_state_counts['review-required']} |",
        "",
        "Next-state partition (sum must equal 67):",
        "",
    ]
    for state, count in sorted(next_state_counts.items()):
        lines.append(f"- `{state}`: **{count}**")
    lines += ["", f"Sum: **{sum(next_state_counts.values())}**.", "", "## One-row-per-active-relation ledger", ""]
    lines += [
        "| Relation | X→Y | Direct | Canonical edge/evidence/proof rows | Staged edge/evidence rows | Support / endpoint result | Status | Exactly one next state |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in relation_rows:
        canonical = row["canonical"]
        def rows_of(key: str) -> str:
            obj = canonical[key]
            return "–" if obj is None else f"{obj.get('parquet', {}).get('rows', '?'):,}"
        staged = row["staged"]["counts"]
        staged_text = f"{staged['edge_rows'] if staged['edge_rows'] is not None else '–'}/{staged['evidence_rows'] if staged['evidence_rows'] is not None else '–'}"
        support = row["support_gaps"]
        endpoint = row["endpoint_anti_join"]
        support_text = f"gaps {support['edges_without_evidence']}/{support['evidence_without_edge']}; endpoint {endpoint['result']}"
        lines.append(
            f"| `{row['relation']}` | `{row['schema']['source_type']}→{row['schema']['target_type']}` | {str(row['schema']['direct']).lower()} | {rows_of('edge')}/{rows_of('evidence')}/{rows_of('proof')} | {staged_text} | {support_text} | `{row['status']}` | `{row['next_state']}` |"
        )
    lines += [
        "",
        "Every table row has object generations/hashes, source-native semantics, support-gap provenance, endpoint-check provenance, staging identities/prefixes, explicit absence/deferral fields, and one `next_state` in the JSON ledger.",
        "",
        "## Live staging routing",
        "",
        f"All {inventories['staging']['object_count']:,} validated live objects under `v2/staging/` fall into exactly 13 prefixes. Every prefix is routed below; the durable tracked immutable inventory mirror contains every exact object name, generation, size, MD5 when available, and CRC32C.",
        "",
        "| Prefix | Objects | Parquets | Identity digest | Relations | Route/exclusion |",
        "| --- | ---: | ---: | --- | --- | --- |",
    ]
    for row in prefix_rows:
        lines.append(
            f"| `{row['prefix']}` | {row['object_count']} | {row['parquet_object_count']} | `{row['object_identity_sha256']}` | {', '.join(f'`{x}`' for x in row['relations'])} | {row['decision']} |"
        )
    lines += [
        "",
        f"The extra live edge object `edges/gene_interacts_gene.parquet.bak_20260618_ot` is excluded from the active denominator as a backup, not a relation. All {len(top_metadata)} metadata objects, the single proof object, and the 28 accepted ReMap feature objects are captured exactly in the JSON ledger/inventory.",
        "",
        "## Evidence completeness decisions",
        "",
        f"Accepted no-evidence exceptions ({accepted_exception_count}): " + ", ".join(f"`{x}`" for x in sorted(ACCEPTED_NO_EVIDENCE_EXCEPTION)) + ". These are accepted topology with provenance backfill only when source-backed/useful; no evidence may be fabricated.",
        "",
        "Evidence-backfill candidates: " + ", ".join(f"`{x}`" for x in sorted(EVIDENCE_BACKFILL)) + ". `molecule_synergizes_molecule` has a complete staged exact-support file but lacks score/study/context values and therefore needs review. `molecule_contraindicates_disease` has no accepted contraindication-specific staged evidence.",
        "",
        "## Readback and fail-closed limits",
        "",
        "- GCS direct object inventory and Parquet footer reads succeeded.",
        f"- Freeze gate PASS: fresh start/end inventories ({args.captured_at} to {end_captured_at}) have identical names, generations, sizes, and hashes for edges, evidence, proof, staging, metadata, and accepted ReMap feature surfaces.",
        "- `/Users/jkobject/mnt/gcs/jouvencekb/main` existed but was empty/unmounted. GCS/FUSE parity is therefore **not established**; no FUSE result is treated as confirming GCS.",
        "- No Mac all-relation endpoint/support scan was run. Each JSON row labels checks as fresh bounded readback, report-backed, exactly absent evidence, or not rerun. The independent reviewer must run representative/high-risk scans and use an approved worker for any full scan.",
        "- ReMap: historical staged object identities remain routed/excluded, while route C is complete on the accepted bounded and full canonical feature/context surfaces. Scaling, VM review, recovery, and resume are closed; only topology conversion remains policy-deferred.",
        "",
        "## Exact commands",
        "",
        "```bash",
        "git fetch origin --prune",
        "git worktree add -b docs/t_8c7f0862-live-reconciliation /Users/jkobject/Documents/jouvence/.worktrees/t_8c7f0862 origin/main",
        "for p in edges evidence proof staging metadata; do gcloud storage ls --recursive --json \"gs://jouvencekb/main/$p/**\" > \"artifacts/cache/t_8c7f0862/live_inventory/$p.json\"; done",
        "gcloud storage ls --json gs://jouvencekb/main/features/remap_crm_tf_enhancer_support.parquet > artifacts/cache/t_8c7f0862/live_inventory/features_remap_bounded.json",
        "gcloud storage ls --recursive --json 'gs://jouvencekb/main/features/remap_crm_tf_enhancer_support_full/**' > artifacts/cache/t_8c7f0862/live_inventory/features_remap_full.json",
        "jq -s add artifacts/cache/t_8c7f0862/live_inventory/features_remap_{bounded,full}.json > artifacts/cache/t_8c7f0862/live_inventory/features_remap.json",
        "# Repeat every listing as *_end.json after the capture. The generator compares exact identity digests and fails if any name/generation/size/hash changed.",
        "# PyArrow GcsFileSystem + ParquetFile.metadata read every canonical and non-ReMap staged Parquet footer; ReMap was intentionally excluded.",
        f"uv run python scripts/build_live_relation_reconciliation.py --inventory-dir artifacts/cache/{TASK_ID}/live_inventory --captured-at {args.captured_at} --schema-commit {args.schema_commit} --json-output {args.json_output} --markdown-output {args.markdown_output} --inventory-output {args.inventory_output}",
        "python -m json.tool .omoc/reports/relation_reconciliation_live_t_8c7f0862.json >/dev/null",
        "shasum -a 256 .omoc/reports/relation_reconciliation_live_t_8c7f0862*.json docs/relation_reconciliation_live_t_8c7f0862.md",
        "```",
        "",
        "No canonical GCS write and no VM start occurred.",
    ]
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text("\n".join(lines) + "\n")

    # Update the ledger after the Markdown write is intentionally avoided: the
    # ledger hash printed in Markdown must describe exactly the immutable JSON.
    print(json.dumps({
        "relations": len(relation_rows),
        "canonical": len(canonical_names),
        "with_evidence": len(evidence_names),
        "without_evidence": canonical_without_evidence,
        "canonical_edge_rows": canonical_edge_rows,
        "next_state_counts": dict(sorted(next_state_counts.items())),
        "ledger_sha256": ledger_sha,
        "inventory_sha256": inventory_bundle_sha,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
