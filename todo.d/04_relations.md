# 04 — Relations

## Inferred edges / derived links

New user-requested analysis:

- `t_f0ad9dff` — `INFERRED-EDGES-POLICY`: analyze relation-chain templates for obvious inferred links from the KG.
- `t_ee00835b` — validate inferred-edge policy/prototype counts and leakage guardrails.
- `t_af066273` — review inferred-edge policy and first-family recommendation.

Definition: inferred links must remain labeled/separate from observed canonical edges, with explicit evidence/provenance and GNN leakage guardrails. Do not promote inferred candidates into observed canonical relation files by default.


## Current source of truth

- `t_0b1f53d9` — relation audit: `validated`/reviewed.
- `t_c07b8b57` — dataset/paper graph disconnection: `validated` policy/export guardrail.
  - Decision: `dataset` and `paper` nodes/relations are provenance/catalog metadata only and must be excluded from training/inference graph adjacency by default.
  - `t_9ad833bf` / `t_fa4e9f85` accepted the backup/checksum gate for `nodes/dataset.parquet`, `nodes/paper.parquet`, `edges/dataset_contains_cell_line.parquet`, and `edges/dataset_contains_tissue.parquet`.
  - `t_d97c4547` cleanup promotion: `review-required`; selected the reversible retain-with-labels option, adding canonical policy sidecars at `metadata/dataset_paper_graph_policy_t_d97c4547.{json,md}` and leaving the affected Parquets in place as metadata-only/non-training inventory.
- `t_9c86ca89` — literature paper/author/citation scope: `review-required` producer handoff.
  - Decision: keep publication identifiers as evidence metadata in the biomedical KG; if a Paper/Author/Citation layer is retained, store it as a separate `literature_index` namespace/export, not default TxGNN training adjacency.
  - Current canonical `nodes/paper.parquet` has 2,958,199 OpenTargets-sourced PMID rows; no canonical `paper_cites_paper` or `paper_produced_dataset` edge files exist. OpenAlex is the preferred source for any future metadata-only literature index; large ingest requires explicit approval.
- Approved docs:
  - `docs/kg_schema_overview.md`
  - `docs/relation_coverage_current.md`
  - `docs/dataset_paper_graph_disconnection_t_c07b8b57.md`
  - `docs/literature_metadata_policy_t_9c86ca89.md`
  - `reproduce/15_kg_schema_overview.ipynb`

Counts:

- 67 active declared relations
- 40 canonical active edge relations; the three previously review-required writes are review-accepted at unchanged object generations by `t_2d1f767d`
- 27 declared relations not canonical yet
- 18 staged-only/deferred
- 5 schema-only/missing

## Prioritized relation plan

- `t_cf77187d` — relation backlog/fanout: `design done`.
- `docs/relation_backlog_prioritized.md` exists.

## Mutation-specific path

- `t_60b3e504` — mutation policy: `design done`.
- `t_79f8684d` — 25k policy-aware staged tranche: `pilot accepted`/`staged-only` for QA only; not canonical; not full all-part rebuild.
- `t_f32f1f5b` — all-25-part `mutation_affects_transcript` staged candidate accepted after validation/review.
- `t_225ae18c` — `mutation_affects_transcript` canonical promotion completed and independently accepted.
- `t_1cfcd48f` — `mutation_in_gene` relation-specific canonical promotion completed from full all-25-part containment-gated candidate (`t_2bb8e7de`) after live endpoint revalidation; status `canonical promoted`/`review-accepted` by `t_18a346a4`, reconfirmed at unchanged object generations by `t_2d1f767d`. Canonical paths: `gs://jouvencekb/main/edges/mutation_in_gene.parquet` and `gs://jouvencekb/main/evidence/mutation_in_gene.parquet`; the containment proof is embedded row-by-row in the evidence payload and is no longer a separate canonical table. Report: `docs/mutation_in_gene_canonical_promotion_t_1cfcd48f.md`.
- `mutation_overlaps_enhancer` — coordinate-overlap alone remains context/support-only and not canonical observed regulatory evidence. User correction `t_0aa76f3b`: externally support-gated overlap may become an evidence-backed edge candidate with support source/score/context and leakage policy. `t_00551bc3` relation-specifically promoted the reviewed `t_73c67c1b` non-context-support-gated candidate to canonical `edges/` and `evidence/` with 1,664,278 rows each, 5,814 mutation endpoints, 114,191 enhancer endpoints, zero endpoint misses, zero duplicate/gap failures, canonical/staged SHA256 parity, and support-gate checks passing; status `canonical promoted`/`review-accepted` by `t_d11a3bb7`, reconfirmed by `t_2d1f767d`. Report: `docs/mutation_overlaps_enhancer_canonical_promotion_t_00551bc3.md`.
- `t_4b1227b3` — canonical promotion only after relation-specific explicit review acceptance; not a blanket genomic-direct promotion.

## Other active relation waves

- Wave B protein-native mechanisms: `t_15e780b9` (`review-required` producer accepted for downstream validation) → `t_145b3cb9` → `t_c3eb09e3`.
- Wave C pharmacology/cell-line context: `t_103021f3` (`review-required` producer accepted for downstream validation) → `t_d71683f5` → `t_a7c7a5d1`.
- Wave E evidence-only pharmacology: `t_cd7fec1f` (`review-required`; linked tester `t_18159206`) → `t_18159206` → `t_fcb5b69f`.

## Clinical trial support evidence/features

- `t_c4f67957` — ClinicalTrials.gov source-backed production candidate for the selected OpenTargets clinicalReportIds/NCT seed: accepted as `staged production candidate`, not global all-CTGov coverage.
- `t_f8841ff7` — promoted reviewed CTGov support artifacts to canonical KG as `canonical promoted`/`review-required` without changing `edges/molecule_treats_disease.parquet`: evidence rows at `gs://jouvencekb/main/evidence/molecule_treats_disease.parquet`, trial index/link metadata under `metadata/`, NCT text features under `features/`, and a 384-d HashingVectorizer scaffold under `features/embeddings/clinical_trials_gov_trial_text/hashing_vectorizer/scaffold_v1/`. Trial status/phase/outcomes remain evidence/features/weights only, not negative graph edges. See `docs/clinical_trials_gov_canonical_promotion_t_f8841ff7.md`.
- `t_957a3640` — resolved the ambiguous structured/text sidecar status: structured trial fields and text summaries/outcomes/eligibility/why_stopped are `canonical promoted` sidecar metadata/features, not clinical-trial graph nodes and not default PyG topology. The full 6,092-row HashingVectorizer trial-text table is now `canonical promoted fallback sidecar` under `features/embeddings/clinical_trials_gov_trial_text/hashing_vectorizer/full_staged_fallback_v1/`; foundation clinical text embeddings remain `blocked-with-resource`. See `docs/clinical_trials_canonical_features_resolution_t_957a3640.md`.

## Definition of done

A relation is not done until it is either:

- canonical promoted + reviewed, or
- explicitly accepted as staged/deferred with a reason and no misleading downstream claim.
