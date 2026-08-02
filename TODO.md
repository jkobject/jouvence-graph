# Jouvence TODO — current human mirror

Kanban board `txgnn` remains the dispatch/source-of-truth. This file is a compact human overview; detailed phase mirrors live in `todo.d/`.

_Status snapshot: 2026-07-19 CEST._

## Operating rule

Do **not** use `.omoc` for new work. It is a legacy scratch/cache location from older runs. New outputs should go to:

- `artifacts/staged/<task-id>/` for local staged artifacts;
- `artifacts/cache/<task-id>/` for bounded local cache if unavoidable;
- `docs/` for human-readable reports;
- `gs://jouvencekb/staging/...` for remote staged artifacts;
- canonical writes only under `gs://jouvencekb/main/...` after validation + review.
- the single current neighbor-sampling build under `gs://jouvencekb/pyg/` only
  after staged build, review, direct-copy verification, and manifest-last publication.

Heavy Jouvence jobs are VM-only. Any card that may run LaminDB full/bulk syncs, production/full PyG/GNN exports or training, embeddings/full-KG scans, all-relation reads, or bulk canonical KG reads/writes must state `must_run_on=txgnn-worker` (the retained VM name) or another explicitly approved in-region worker, use `gs://jouvencekb/main` as source, and forbid `/Users/jkobject/mnt/gcs/...` / macOS GCS-FUSE for heavy work. Required preflight: verify `hostname`, launch/inspect with `gcloud compute ssh txgnn-worker`, check for an existing related writer/process, and fail if any heavy input/output path starts with `/Users/jkobject/mnt/gcs`. Copyable card template: `artifacts/reports/t_d682b7ad/heavy_job_vm_only_card_template.md`. ReMap route C is complete and unsupervised; these generic VM rules do not authorize or resume ReMap work.

ReMap route C is complete and has no active watchdog, resume loop, SSH-liveness gate, or recovery lane. Prior fresh-UDC continuation rules and supervisor templates are preserved as historical reproducibility evidence only; they are not current operating instructions and must not be auto-resumed.

Verified KG access:

- GCS canonical root: `gs://jouvencekb/main`
- macOS FUSE root: `/Users/jkobject/mnt/gcs/jouvencekb/main` for small bounded/local inspection only; forbidden for heavy LaminDB/PyG/ReMap/embedding/full-KG work.

## Status vocabulary

Avoid bare “done” except as a Kanban state. Use:

- `design done`
- `pilot accepted`
- `staged-only`
- `review-required`
- `validated`
- `canonical candidate`
- `canonical promoted`
- `stopped-by-user`
- `production/full done`

## Review snapshot — 2026-07-19

This snapshot supersedes the older June/July execution notes below. Detailed denominators and evidence are in `todo.d/01_lamindb.md`, `todo.d/02_pyg_gnn.md`, and `todo.d/03_embeddings.md`.

1. **The human Gene identity migration is staged-only and review-required.** `t_8b9cdabc` produced a validated staged candidate targeting 81,715 human ENSG nodes at commit `8714378`; its PR and independent review remain outstanding. The 27,610 NCBI IDs are aliases/endpoints that require authoritative remap or explicit quarantine; 158,505 non-human homologue nodes and `gene_ortholog_gene` are excluded from the human canonical candidate. No canonical promotion is claimed.
2. **LaminDB ingestion is partial, and accepted counters differ from physical counters.** The latest durable accepted ledger (2026-07-18 evidence) is 11,671,485 / 230,874,162 rows. The latest sealed physical readback from the same date is 12,011,512 rows, with +170,027 edges and +170,000 evidence still uncredited. No newer mismatch-0 readback is claimed; the denominator also awaits reviewed ENSG-only rebasing.
3. **The corrected immutable public embeddings v2 candidate is validated.** Producer `t_2d54477b` published 808,269 rows across 12 logical leaves; independent reviewer `t_2e6b355f` passed the exact 51-object candidate at generation `1784460889447648`. This is a validated immutable candidate, not a mutable latest-pointer or blanket source-backed vector for every node. Rejected v1 remains historical and unaccepted.
4. **The old Gene Nucleotide Transformer scratch run is superseded.** `t_d3b876b3` remains historical/non-canonical, but the later exact-ENSG lineage produced and independently accepted 78,644 / 81,715 genomic embeddings with 3,071 explicit missing. Do not resume the old scratch or superseded continuation `t_41b61edd`.
5. **DepMap revision 2 is code/test ready but not fully rebuilt.** PR #11 is pushed at `e40e2508b8f061f70fc7a4fcbf05b0f4a1accfaf`; `t_3c7766fa` waits behind the ENSG heavy-worker lane before the required dual full build and fresh immutable staged artifact. The prior candidate remains rejected.
6. **PyG/GNN has a real reviewed runtime smoke, not full-KG training.** Sequential edge streaming is retained for audit/build work, but the accepted training target is `LinkNeighborLoader`/`NeighborLoader` over a versioned disk-backed CSC `GraphStore` + `FeatureStore` snapshot under `gs://jouvencekb/pyg/`. Full multi-relation model-quality training has not run.

## Current phase mirrors

Use the live board plus the dated phase files below as the current-state anchor. `docs/current_state_20260623.md` is historical context, not live dispatch state.

- `todo.d/01_lamindb.md`
- `todo.d/02_pyg_gnn.md`
- `todo.d/03_embeddings.md`
- `todo.d/04_relations.md`
- `todo.d/05_remap.md`
- `todo.d/06_process.md`

## Current KG coverage source of truth

Use these, not old `.omoc` reports:

- `docs/kg_schema_overview.md`
- `docs/relation_coverage_current.md`
- `reproduce/15_kg_schema_overview.ipynb`
- `docs/relation_backlog_prioritized.md`

Accepted snapshot:

- active declared relations: `67`
- canonical active edge relations: `40`; the three previously review-required canonical writes are independently review-accepted at their unchanged object generations by `t_2d1f767d`
- canonical relations with evidence: `18`
- canonical relations without evidence: `22`
- declared relations not canonical yet: `27`
- staged-only/deferred: `18`
- source-audit-only/deferred: `2`
- feature-context-not-edge: `2`
- schema-only/missing: `5`
- canonical edge rows: `100,080,390`
- node rows: `55,523,691`

## Active priorities

### 1. LaminDB / `lnschema_txgnn`

`lnschema_txgnn` activation and bounded loaders are validated, but global ingestion remains partial. Current durable counter and Gene-identity boundaries are in `todo.d/01_lamindb.md`.

- `t_8b9cdabc` — human ENSG-only migration: `staged-only` / `review-required` handoff at `8714378`; canonical KG and LaminDB unchanged, PR/review outstanding.
- `t_ce839966` and `t_075f5353` — superseded historical +158,505 non-human Gene sync plans; remain inert and must not run.
- Accepted-versus-physical drift remains explicit; physical rows are not product credit without accepted readback evidence.

Production/full done requires reviewed ENSG-only denominators, exact-ID row parity, mismatch 0, and independent acceptance. Schema activation alone is not production/full done.

### 2. PyG / GNN

Existing PyG work is a bounded export/streaming pilot, not completion. The
2026-07-27 target is a versioned neighbor-index snapshot, worker-local verified
cache, memory-mapped `GraphStore`/`FeatureStore`, and split-safe
`LinkNeighborLoader` training. See `todo.d/02_pyg_gnn.md` and
`docs/guides/pyg-and-embedding-contracts.md`.

- `t_015bd9a4` — full KG / representative KG PyG export plus runnable GNN smoke/training.
- `t_1d1eb3a1` — validate actual HeteroData/GNN runtime.
- `t_468db80e` — review full PyG/GNN acceptance.

Done means an actual PyG/HeteroData object exists and a GNN run executes on it.

### 3. Embeddings

Four source-backed embedding families now have one independently validated immutable v2 candidate, while model-side fallback remains required for uncovered nodes/modalities.

- `t_2d54477b` — published immutable v2 candidate: 808,269 rows, 12 logical leaves, 51 objects; producer card is historical/triage after handoff.
- `t_2e6b355f` — independent v2 reviewer: `validated` PASS on 2026-07-19. No latest-pointer mutation or universal-coverage claim.
- Rejected v1 remains historical and unaccepted.
- `t_d3b876b3` — historical stopped scratch only; superseded by the accepted exact-ENSG lineage with 78,644 / 81,715 embeddings and 3,071 explicit missing. Do not auto-resume it or `t_41b61edd`.
- Learned fallback is still required where reviewed source vectors are absent; it is not biological evidence.
- Before more embedding compute, distinguish no-source-payload nodes from
  source-eligible rows whose embedding computation is genuinely missing. See the
  human registry `docs/guides/pyg-feature-registry.md`.

### 4. ReMap

ReMap route C is complete and unsupervised. The accepted support-only artifacts remain preserved, while all-peak expansion and graph-topology conversion remain deferred; do not auto-resume either line.

Accepted staged-only support artifacts:

- `t_3b8a2c4d` — CRM support/QA first10k chr1 pilot.
- Prefix: `gs://jouvencekb/staging/source-native-expansion/remap-crm-tf-binds-enhancer-support-chr1-first10k-20260623-t_3b8a2c4d/`
- `t_b599d3bb` — CRM support/QA all-chromosome bounded 5k-per-chrom artifact; staged-only/support-only.
- Prefix: `gs://jouvencekb/staging/source-native-expansion/remap-crm-tf-binds-enhancer-support-allchrom-5kperchrom-20260623-t_b599d3bb/all_chrom_5k_per_chrom/`

Canonical-readiness decision:

- `t_9c0e6a68` — decision doc: `docs/remap_crm_canonical_readiness.md`.
- Only appropriate canonical target from current CRM artifact: support/evidence sidecar `features/remap_crm_tf_enhancer_support.parquet`, via separately reviewed promotion card `t_656a1102`.
- CRM is `crm_aggregated_support` / support-QA only; not canonical `observed_binding`, not canonical `tf_binds_enhancer` edge/evidence, and not `tf_regulates_gene`.
- Observed binding requires peak-level/per-experiment source rows with assay/biosample/direct-binding evidence; TF→gene regulation requires a separate source-native TF→target regulation policy/source.

### 5. Mutation genomic direct relations

`mutation_affects_transcript`, `mutation_in_gene`, and the support-gated `mutation_overlaps_enhancer` candidate are canonical promoted/review-accepted. Consolidated reviewer `t_2d1f767d` reconfirmed the exact unchanged object generations and relation-specific gates. `mutation_in_gene` remains the containment-gated OpenTargets `target.genomicLocation` relation with 2,599,525 edge/evidence/proof rows. `mutation_overlaps_enhancer` remains limited to the reviewed non-context-support-gated `t_73c67c1b` candidate (1,664,278 edge/evidence rows); coordinate overlap alone remains context/support-only and not observed regulatory evidence.

Relations:

- `mutation_affects_transcript` — `canonical promoted` / reviewed.
- `mutation_in_gene` — `canonical promoted` / `review-accepted`; relation-specific canonical write by `t_1cfcd48f`, accepted by `t_18a346a4` and reconfirmed by `t_2d1f767d`.
- `mutation_overlaps_enhancer` — `canonical promoted` / `review-accepted` for the support-gated `t_73c67c1b` candidate promoted by `t_00551bc3`; coordinate-only overlap remains context/support-only and not observed regulatory evidence.

Cards:

- `t_60b3e504` — policy done.
- `t_79f8684d` — 25k staged tranche accepted for QA only.
- `t_f32f1f5b` / `t_225ae18c` — all-part `mutation_affects_transcript` candidate accepted and canonical promoted.
- `t_8de911c0` — remaining-relation next-state decision: `mutation_in_gene` bounded containment candidate only; `mutation_overlaps_enhancer` coordinate-only context/support feature; no broad or relation-specific promotion card created.
- `t_0aa76f3b` — support-gated `mutation_overlaps_enhancer` policy/pilot: evidence-backed staged edge candidate with external support context, no canonical write.
- `t_73c67c1b` — full non-context-support-gated `mutation_overlaps_enhancer` staged candidate: 1,664,278 edges/evidence rows with live endpoint anti-joins and edge/evidence validation passing. Report: `docs/mutation_overlaps_enhancer_support_gated_full_t_73c67c1b.md`.
- `t_00551bc3` — relation-specific canonical promotion of reviewed `t_73c67c1b` support-gated `mutation_overlaps_enhancer` to `gs://jouvencekb/main/{edges,evidence}/`; status `canonical promoted`/`review-accepted` by `t_d11a3bb7`, reconfirmed by `t_2d1f767d`. Report: `docs/mutation_overlaps_enhancer_canonical_promotion_t_00551bc3.md`.
- `t_2bb8e7de` — full all-25-part `mutation_in_gene` containment-gated staged candidate built/validated under `artifacts/staged/t_2bb8e7de/`; producer handoff was `review-required`, with no canonical write.
- `t_1cfcd48f` — `mutation_in_gene` live endpoint revalidation and relation-specific canonical write to `gs://jouvencekb/main/{edges,evidence}/`; status `canonical promoted`/`review-accepted` by `t_18a346a4`, reconfirmed by `t_2d1f767d`. Report: `docs/mutation_in_gene_canonical_promotion_t_1cfcd48f.md`.
- `t_4b1227b3` — do not use as blanket promotion; only relation-specific promotion after explicit acceptance.

### 6. Relation waves

Use `docs/relation_backlog_prioritized.md` and `todo.d/04_relations.md`. A relation is not complete until canonical promoted+reviewed or explicitly accepted as staged/deferred.

### 7. Process hygiene

- `t_caacd3d1` — keep `todo.d/` synced, enforce honest status labels, fix review routing/watchdog behavior, and prevent `.omoc` recreation.

## Git / reviewability

The migration from `t_4cab4a2f` is now executed: `/Users/jkobject/Documents/jouvence` is the canonical local checkout for `https://github.com/jkobject/jouvence-graph`. The legacy `txgnn` path remains a compatibility-only location for preserved historical worktrees. Project-level Git commands and human review run from the canonical checkout; new task worktrees live under `/Users/jkobject/Documents/jouvence/.worktrees/<task-id>` unless a card names another verified worktree.

The root still contains ignored local artifacts/caches. Reviewability therefore requires an explicit Git diff and generated-file guard; directory contents alone are not a commit surface. Do not initialize or maintain a second canonical checkout under `~/code`.

## Historical note

Older docs/reports may mention `.omoc` and old local caches. Treat those as historical evidence only, not current instructions. If an old worker actively targets the legacy path, let it finish, then preserve useful outputs under `artifacts/`, `docs/`, or GCS staging and retire the legacy path when no active command references it.
