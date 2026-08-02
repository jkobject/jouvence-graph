# 05 — ReMap

ReMap route C is complete and unsupervised. There is no active ReMap watchdog, resume loop, SSH-liveness gate, or recovery lane. The historical VM-only guardrails and supervisor artifacts remain evidence of how prior runs were bounded; they are not permission to launch or auto-resume ReMap processing.

## Current state

Route C is complete as the selected support-only product outcome. The accepted sidecars remain preserved; all-peak/full expansion and conversion to active graph topology remain deferred and are not `production/full done`.

- `t_8bc6dacf` — stopped by user strategy decision; not canonical; do not auto-resume.
- No canonical `tf_binds_enhancer` edge/evidence exists yet, but the approved ontology direction is to use `tf_binds_enhancer` for ReMap CRM/peak/motif-supported TF-enhancer binding evidence rather than inventing a permanent support-only relation label.

Accepted support-only artifacts:

- `t_3b8a2c4d` — CRM support/QA first10k chr1 `pilot accepted`/`staged-only`.
- Prefix: `gs://jouvencekb/staging/source-native-expansion/remap-crm-tf-binds-enhancer-support-chr1-first10k-20260623-t_3b8a2c4d/`
- `t_b599d3bb` — CRM support/QA all-chromosome bounded 5k-per-chrom artifact accepted as staged-only/support-only.
- Prefix: `gs://jouvencekb/staging/source-native-expansion/remap-crm-tf-binds-enhancer-support-allchrom-5kperchrom-20260623-t_b599d3bb/all_chrom_5k_per_chrom/`
- `t_f2a2952e` — full/unbounded CRM support/QA sidecar `canonical promoted full support sidecar` / `review-required` after readiness gate `t_7e356c5c` and reviewer `t_0d77b4f0`.
- Prefix: `gs://jouvencekb/main/features/remap_crm_tf_enhancer_support_full/` with 24 chromosome summary shards, `tf_global_summary.parquet`, `manifest.json`, and `metadata.json`; see `docs/remap_crm_full_support_sidecar_canonical_promotion_t_f2a2952e.md`.
- Semantics: `crm_aggregated_support` / support-QA only.
- Not `observed_binding`; not `tf_regulates_gene`; not canonical `tf_binds_enhancer` edge/evidence.

## Canonical-readiness decision

- `t_9c0e6a68` — decision doc: `docs/remap_crm_canonical_readiness.md`.
- `t_f558cee3` — reassessment/prototype update: CRM support can be linked back to ReMap ChIP-seq peak rows by reconstructed same-TF coordinate overlap, but CRM itself still lacks source peak IDs, experiment accessions, antibody/protein metadata, and cell/biotype context.
- Bounded prototype: `artifacts/staged/t_f558cee3/reports/remap_crm_peak_decomposition_prototype_report.md` over first 80 chr1 CRM intervals found 6,876 same-TF ReMap `all` peak overlaps, 1,846 distinct source accessions, 1,421 distinct biotypes, and ReMap biotype XLSX metadata matches for 230 sample biotypes.
- The existing canonical feature sidecar remains useful as a bounded QA/support artifact and must not be overwritten by this reassessment.
- User policy correction: CRM is derived from ReMap ChIP-seq, so the strongest honest graph target is canonical `tf_binds_enhancer` with caveats encoded in `evidence/tf_binds_enhancer`, not a support-only replacement relation.
- A new bounded staged candidate is justified: active `tf_binds_enhancer` rows should use ReMap `all` peak evidence where available and may use CRM-derived reconstructed binding support when the evidence row records the reconstruction policy, missing CRM-native peak foreign key, metadata coverage, motif support, context fields, and leakage guard.
- `tf_regulates_gene` is blocked from the CRM support artifact. It requires a separate source-native TF→target regulation source or an explicitly reviewed inferred-relation policy.

## Completed route C / future approval boundary

- Keep the existing promoted bounded `features/remap_crm_tf_enhancer_support.parquet` as a `crm_aggregated_support` feature/QA sidecar; it was not overwritten by the full sidecar promotion.
- Historical fresh-UDC continuation policy and templates from `t_a2674d49` are retained for reproducibility only. They are retired from the active control plane and must not be invoked by preflight, supervisor, health, recovery, cron, or automatic dispatch logic.
- Treat `features/remap_crm_tf_enhancer_support_full/` as shard-aware support-only feature/QA material, not graph topology and not a replacement for `tf_binds_enhancer` edge/evidence promotion.
- `t_6c07d9c8` — shard-aware read-only helper added at `manage_db/remap_crm_support_reader.py` with fixture tests in `tests/test_remap_crm_support_reader.py`. Use it to list the 24 canonical shards, read bounded TF/enhancer samples from one chromosome, read `tf_global_summary.parquet`, and run bounded endpoint checks over loaded samples. Live FUSE readback report: `artifacts/reports/t_6c07d9c8_remap_crm_support_reader_live_readback.json`. Semantics remain support-only feature/QA material, not edge/evidence/observed binding/inferred topology.
- `t_a405fe3b` / reviewer `t_95856c15` — bounded first80 chr1 CRM/peak `tf_binds_enhancer` edge/evidence pilot is `pilot accepted` / `staged-only` with 1,224,536 edges and 6,356,561 evidence rows. It is not canonical/full production.
- `t_f8cc9e4b` — full/unbounded CRM/peak edge/evidence scaling is a validated feasibility/policy gate. Existing accepted full CRM lineage `t_5968ce32` reports 24,453,482,386 TF × CRM × enhancer candidate support rows intentionally not materialized; converting it into active `tf_binds_enhancer` edges/evidence would require either reviewed aggregate-reduction semantics or explicit external large-product materialization approval. See `docs/remap_crm_tf_binds_enhancer_full_feasibility_t_f8cc9e4b.md` and `artifacts/reports/t_f8cc9e4b_feasibility_gate.json`.
- `t_2e1b271a` — CTO decision: choose route C for now. Full ReMap CRM stays `support-only` / feature-QA material until a stricter reduction policy exists; do not create full edge/evidence build work, do not silently aggregate the sidecar into graph topology, and do not claim canonical `tf_binds_enhancer` edge/evidence promotion. Decision doc: `docs/remap_crm_tf_binds_enhancer_next_decision_t_2e1b271a.md`.
- `t_ea6e00ab` — bounded motif co-location layer for ReMap/CRM `tf_binds_enhancer` support is `review-required` / `staged-only`. It scans real JASPAR 2026 CORE vertebrate PFMs on bounded hg38 enhancer/CRM intersections from the accepted `t_a405fe3b`/`t_f558cee3` lineage and writes 549 motif rows: 440 `motif_support` rows linked to parent ReMap observed evidence plus 109 motif-only predicted/support rows with `edge_key=NULL`. Artifact/report: `artifacts/staged/t_ea6e00ab/` and `docs/remap_crm_motif_colocation_t_ea6e00ab.md`. No canonical writes.
- `t_ba65eb81` — compact-coded ReMap/CRM `tf_binds_enhancer` support prototype is `review-required` / `staged-only`. It supersedes treating the 24.45B TF×CRM×enhancer count as a final blocker: that count is the naive exploded row product, while the intended representation stores per-enhancer `support_codes: list<int64>` arrays plus a support-code dictionary mapping each code to TF/source/accession/biotype/antibody/protein/context/motif/evidence metadata. Prototype artifacts: `artifacts/staged/t_ba65eb81/features/tf_binds_enhancer_enhancer_support_codes.parquet`, `artifacts/staged/t_ba65eb81/features/tf_binds_enhancer_support_code_dictionary.parquet`, `artifacts/staged/t_ba65eb81/reports/query_examples.sql`, and `docs/remap_compact_coded_tf_binds_enhancer_t_ba65eb81.md`. Query examples now explicitly cover enhancer→TFs, TF→enhancers, support_class, motif, direct cell_line/tissue/cell_type/antibody/protein predicate shapes, and report zero direct-slot coverage with ReMap biotype/context_note fallback coverage. No canonical writes; active training edges require a later reviewed reducer/leakage policy.
- Do not create a separate support-only relation label unless a future reviewer explicitly needs a non-label namespace; the default relation name for this evidence family is `tf_binds_enhancer` when edge/evidence semantics are reviewed and accepted.
- No canonical writes should happen in ReMap readiness/scaling cards without a separate positive reviewer gate.

## Definition of done

Route C is done when its reviewed support-only sidecars and policy receipts remain preserved and no ReMap execution path participates in active preflight, health, recovery, or notification logic. A future `tf_binds_enhancer` topology product would be separate, explicitly approved work with its own staged artifact, motif/evidence policy, endpoint audits, leakage controls, and reviewer gate; route C completion does not claim that future topology product.
