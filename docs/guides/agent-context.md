
# Jouvence agent context

[← Documentation index](../README.md) · [Lessons learned](lessons-learned.md) · [LaminDB operations](lamindb-porting-operations.md)

`AGENTS.md` is the single boot file. This page holds task routing and domain rules for workers who need more than the boot gist.

## Current operating posture

`work/txgnn` is the canonical local worktree for `https://github.com/jkobject/jouvence-graph` and the project-level review surface. The local path is retained for compatibility. It also holds ignored local artifacts/caches, so reviewers must inspect the explicit Git diff rather than infer scope from directory contents. Parallel task worktrees belong under `/Users/jkobject/.openclaw/worktrees/txgnn/<branch-or-task-id>/`; do not create a second canonical checkout under `~/code`.

Default KG access:

- Canonical KG bucket root: `gs://jouvencekb/main`
- Verified macOS FUSE root: `/Users/jkobject/mnt/gcs/jouvencekb/main` for small bounded/local inspection only.
- Heavy Jouvence jobs are VM-only: LaminDB full/bulk syncs, production/full PyG/GNN exports or training, ReMap scaling, embeddings/full-KG scans, all-relation reads, and bulk canonical KG reads/writes must run on `txgnn-worker` (retained VM name) or another explicitly approved in-region worker using `gs://jouvencekb/main`; do not run them through `/Users/jkobject/mnt/gcs/...` / macOS GCS-FUSE.
- Heavy-card preflight must include `must_run_on=txgnn-worker`, `hostname`, `gcloud compute ssh txgnn-worker`, an existing-process check, and a hard failure if any heavy input/output path starts with `/Users/jkobject/mnt/gcs`.
- New outputs: `artifacts/staged/<task-id>/`, `artifacts/cache/<task-id>/`, `docs/`, or `gs://jouvencekb/staging/...`
- `.omoc/` is legacy-only.
- Jouvence automation is strictly project-scoped. Never stop, pause, resize, or otherwise manage `pert-gym` resources from a Jouvence task; another session may legitimately be using them.

Python is managed with `uv`; use `uv run ...`. Intended LaminDB instance is `jkobject/jouvencekb`; prove the connected instance before writes and stop if the VM resolves to `jkobject/repo` or anything else.

For LaminDB migration, worker sizing/storage, progress semantics, stall recovery, and cost controls, read [`lamindb-porting-operations.md`](lamindb-porting-operations.md).

## Role/task routing

### CTO / orchestrator

Read:

1. `AGENTS.md`
2. `TODO.md`
3. `docs/current_state_20260623.md` for broad status
4. The relevant `todo.d/*.md` phase file only when routing that phase

### KG schema / relation worker

Read:

1. `AGENTS.md`
2. Card context packet
3. `docs/kg_schema_overview.md`
4. `docs/relation_coverage_current.md` or `docs/relation_backlog_prioritized.md` only if the card names relation coverage

### Promotion / validation worker

Read:

1. `AGENTS.md`
2. Card context packet
3. Relevant producer report named in the card
4. Validation recipe in this file

Never promote canonical KG unless the card explicitly authorizes it and gives the target relation/artifact paths.

### PyG/GNN / embeddings worker

Read:

1. `AGENTS.md`
2. Card context packet
3. `TODO.md` section for PyG/GNN or embeddings
4. The exact design/report doc named in the card

### LaminDB sync / worker-operations worker

Read:

1. `AGENTS.md`
2. Card context packet
3. `docs/guides/lamindb-porting-operations.md`
4. Only the exact implementation/report named by the card

Do not infer progress from VM state or a live PID. Require subchunk telemetry and selected-live verification. Keep a strict single-writer gate and never advance a checkpoint from a started-but-uncommitted tranche.

## Modeling doctrine

- Relation names must match source-native assertion and endpoint type.
- Gene-level rows stay in gene relations; split to protein/TF/transcript only when the source is native to those endpoints/assertions.
- Directed disease associations use cause/source → disease endpoints.
- Protein relations require direct protein/isoform evidence or direct protein measurement; never project RNA/gene rows into protein edges.
- Molecule pair drug-effect rows use `molecule_synergizes_molecule`; physical molecular interactions need explicit source-native interaction relations.
- Entity→phenotype is canonical direction; evidence rows carry source-specific predicate/class.
- Edges are deduplicated graph assertions; evidence rows carry source predicates, scores, papers, studies, assays, and provenance.
- Do not create placeholder Parquets merely to satisfy schema coverage.

## Status vocabulary

Use: `design done`, `pilot accepted`, `staged-only`, `review-required`, `validated`, `canonical promoted`, `production/full done`. Avoid bare “done” outside Kanban state.

## Validation / promotion gates

Before promoting a KG tranche:

1. Build in `artifacts/staged/<task-id>/` or GCS staging, not `.omoc`.
2. Validate x/y endpoint anti-joins with DuckDB/PyArrow against canonical FUSE/GCS root.
3. Write evidence rows when source provenance exists.
4. Run `manage_db.audit_edge_evidence` when applicable.
5. Update coverage docs and relevant notebook/report.
6. Run targeted tests.
7. Require reviewer acceptance before canonical write.

Useful checks:

```bash
uv run python -m py_compile manage_db/kg_schema.py manage_db/kg_evidence.py manage_db/backfill_edge_evidence.py manage_db/ingest_opentargets.py
uv run --group dev pytest tests/test_kg_schema_cleanup.py tests/test_kg_evidence.py tests/test_backfill_edge_evidence.py -q
uv run python -m manage_db.audit_kg_coverage /Users/jkobject/mnt/gcs/jouvencekb/main --json > artifacts/reports/<task-id>-coverage.json
```
