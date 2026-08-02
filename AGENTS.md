
# AGENTS.md — Jouvence agent boot

This is the single required boot file for Jouvence agents. This file replaces the old dual-entrypoint pattern; do not ask workers to read a second root boot file. Claude Code and other agents must use this file directly: do not recreate a root `CLAUDE.md` or commit `.claude/` runtime/worktree state.

## Project gist

Jouvence is a biomedical knowledge-graph and zero-shot drug-repurposing project built from the upstream TxGNN method and Python library. This workspace expands that foundation with OpenTargets, HPA, TxData, source-backed evidence, LaminDB cataloging, PyG/GNN export, and learned/foundation embeddings. The legacy `txgnn` import package and `TxGNN` model class remain compatibility boundaries.

## Current source of truth

- Kanban board `txgnn` is the dispatch/source of truth.
- `README.md` is the public GitHub façade for installation, usage, architecture, contribution, and project navigation.
- `TODO.md` is the compact human/agent mirror of current phases and cards.
- `docs/README.md` is the durable documentation index. `docs/guides/` contains stable doctrine and runbooks; other `docs/` files contain dated designs, audits, validation, and promotion evidence.
- `docs/current_state_20260623.md` and `todo.d/` are broader dated phase mirrors; verify live state before relying on them.
- `docs/guides/agent-context.md` holds task routing, modeling doctrine, and validation recipes.
- `docs/guides/kg-architecture-and-evidence.md`, `source-native-modeling.md`, `pyg-and-embedding-contracts.md`, and `review-promotion-and-reviewability.md` hold durable topic doctrine; read only the page relevant to the card.
- `docs/guides/lamindb-porting-operations.md` holds the durable LaminDB/VM/GCS/storage/progress runbook. Read it only for LaminDB sync, worker operations, cost, or recovery tasks.
- `docs/guides/lessons-learned.md` compiles cross-cutting scientific, data, infrastructure, and review lessons; use it to avoid rediscovery, not as live status.

## Reviewability rule

`/Users/jkobject/Documents/jouvence` is the canonical local Jouvence checkout for `https://github.com/jkobject/jouvence-graph`. Task worktrees live under `/Users/jkobject/Documents/jouvence/.worktrees/` unless a card names another verified Git worktree. The checkout also contains ignored local artifacts/caches; those do not make the Git diff reviewable by themselves.

Run project-level Git commands from this root and verify `git rev-parse --show-toplevel` resolves to it. Never `git init` another Jouvence directory or commit ignored artifacts, caches, credentials, GCS/FUSE mirrors, or unrelated workspace state.

## Context discipline

Do not browse the repo/docs broadly by default.

1. Read this file.
2. Read the Kanban card context packet.
3. Read `TODO.md` for current phase/card gist.
4. Read only the specific phase/doc named in the card or in `docs/guides/agent-context.md` for your role.
5. If still unclear, block with the exact missing context instead of wandering through old reports.

When creating Kanban cards, include: target relation/artifact, source files, allowed writes, exact validation, canonical-write permission, relevant docs, and what not to read.

## Public data onboarding

For “set this up and show me Jouvence,” follow
[`docs/getting-started-data.md`](docs/getting-started-data.md). Start in fixture
mode unless `JOUVENCE_DATA_MODE=live` is explicit. Live GCS reads require the
consumer's own `JOUVENCE_BILLING_PROJECT` and ADC; never invent or reuse a
maintainer billing project. Default to read-only, bounded, named-table access.
Exception: `notebooks/07_data_inventory_explorer.ipynb` is intentionally
live-only and does not use `JOUVENCE_DATA_MODE`; its separate checker must be
treated as a real requester-pays GCS operation and run only after ADC and a
caller-owned `JOUVENCE_BILLING_PROJECT` are explicit. The fixture-suite checker
statically validates notebook 07 but skips its execution.
`JOUVENCE_LAMIN_LIVE=1` is a separate opt-in and does not make the currently
partial/external-blocked Lamin mirror canonical. All-relation scans, bulk
downloads, production PyG/GNN, bulk Lamin work, and embedding/full-KG scans are
heavy work and remain prohibited on laptops.

## Safety rules

- Do not treat old `.omoc` paths as current operating instructions; `.omoc/` is legacy-only.
- Use `artifacts/staged/<task-id>/`, `artifacts/cache/<task-id>/`, `docs/`, or `gs://jouvencekb/staging/<task-id>/` for new outputs.
- Bucket contract: canonical KG tables live in `gs://jouvencekb/main`, native inputs in `raw/`, the single current derived PyG build in `pyg/`, Lamin internals in private `.lamin/`, and candidates only in `staging/`. No lifecycle or object versioning is configured; clean staging explicitly. PyG jobs copy `pyg/*` to a user-selected local directory (default: `REPO/data/pyg/`) with `gcloud storage cp --recursive`, verify, then mmap; never mount GCS for sampling.
- Heavy Jouvence jobs (LaminDB full/bulk syncs, production/full PyG/GNN exports or training, ReMap scaling, embedding/full-KG scans, and any all-relation or bulk canonical KG read/write) must run on `txgnn-worker` or another explicitly approved in-region worker using `gs://jouvencekb/main`; do **not** run them from the Mac through `/Users/jkobject/mnt/gcs/...` / macOS GCS-FUSE.
- Heavy cards must include `must_run_on=txgnn-worker` (or the approved worker), preflight `hostname`, use `gcloud compute ssh` for worker launch/inspection, check for an existing related writer/process before starting, and fail immediately if any heavy input/output path starts with `/Users/jkobject/mnt/gcs`.
- Project boundary is strict: Jouvence agents and cost guards may manage only Jouvence resources. Never pause, stop, resize, reinterpret, or gate `pert-gym` VMs, crons, disks, processes, or cards; concurrent project activity may be intentional.
- Canonical writes require explicit card authorization, validation evidence, reviewer acceptance, and a destination-generation precondition. Never create `staged`, `metadata`, `proof`, archives, viewer bundles, or runtime state in the stable roots; temporary candidates may exist only under `gs://jouvencekb/staging/<task-id>/`.
- Python is managed with `uv`; use `uv run ...`.
- Intended LaminDB instance: `jkobject/jouvencekb`. Before every write-capable run, prove the connected instance explicitly; a VM config pointing to `jkobject/repo` is a hard blocker, not a harmless alias.
- LaminDB sync is single-writer. A live PID is not proof of progress: require a durable subchunk delta (`current_offset`, attempted/upserted/verified rows, throughput, and `last_progress_at`). Never advance an offset without selected-live edge/evidence equality and mismatch `0`.
- Do not put relation-wide ORM `count()` calls in the critical path of 10M+ row SQLite-backed Lamin syncs. Use selected-key verification per committed tranche; run full parity audits separately while no writer is active.
- As of the verified July 2026 migration, `txgnn-worker` boots from a 200 GB disk; the detached 500 GB disk and snapshot are rollback assets, not working space. Do not delete them without explicit destructive approval and independent recovery review.

## Domain rules

- Relation names must match source-native assertion and endpoint type.
- Edges are deduplicated graph assertions; evidence rows carry source-specific predicates, scores, studies, assays, and provenance.
- Do not let assay modality redefine an accepted biological relation: observed source assertions and inferred biological implications use the same stable endpoint relation where appropriate, while RNA/proteomics/clinical modality, derivation path, sample and context stay in `evidence/` or `evidence_inferred/`. Protein-product expression may support inferred expression of its encoding gene; it must not be mislabeled as an RNA measurement.
- Causal mechanism, effect direction, pharmacological action, pathogenicity, and response polarity are typed features of existing broad edge tables, with row-level assertions and conflicts in the corresponding evidence tables; do not create GoF/LoF/inhibitor/risk relation-name variants. See `docs/causal_edge_feature_model.md`.
- Do not create placeholder Parquets just to satisfy schema coverage.
- Use precise status vocabulary: `design done`, `pilot accepted`, `staged-only`, `review-required`, `validated`, `canonical promoted`, `production/full done`.

For detailed modeling doctrine and validation commands, read `docs/guides/agent-context.md`.
