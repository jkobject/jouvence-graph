# TxGNN / Jouvence KG — lightweight Kanban mirror

This directory is a human-readable mirror of the active Kanban phases. Kanban remains the source of truth for dispatch/status; these files summarize the big phases, honest status labels, and key card IDs.

## Required status vocabulary

Use these exact labels in docs/comments instead of bare "done" when scope is partial:

- `design done`: architecture/policy doc exists; no production artifact yet.
- `pilot accepted`: bounded pilot reviewed; not necessarily full scale.
- `staged-only`: artifact exists outside canonical KG; no canonical promotion.
- `review-required`: producer is blocked pending independent reviewer/tester/CTO route.
- `validated`: tester checked artifact behavior/counts/readability.
- `canonical promoted`: written to canonical `gs://jouvencekb/main/...` and reviewed.
- `production/full done`: full intended scope exists and runs, not just a pilot/tranche.

Process details live in `docs/kanban_status_hygiene.md`.

Current phase files:

- `docs/current_state_20260623.md` — short current-state anchor for workers/reviewers.
- `01_lamindb.md`
- `02_pyg_gnn.md` — PyG/HeteroData export and actual GNN runtime.
- `03_embeddings.md` — real node/edge embeddings, not surrogate-only pilots.
- `04_relations.md` — relation coverage and source-native relation waves.
- `05_remap.md` — ReMap all-peak stopped/deferred; CRM support/QA path.
- `06_process.md` — review routing, honest status labels, and todo.d hygiene.

## Mirror maintenance rule

When a major Kanban wave changes state, update the relevant phase file with:

1. card IDs;
2. honest status label (`design done`, `pilot accepted`, `staged-only`, `review-required`, `validated`, `canonical promoted`, or `production/full done`);
3. reviewer/tester route if a producer blocks `review-required`; and
4. the remaining condition for true completion.

Do not use this mirror to dispatch work. It is a human-readable status map only.
