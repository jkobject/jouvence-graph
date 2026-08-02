# Kanban status hygiene for TxGNN / Jouvence KG

This board has many design, staging, validation, and promotion cards. Kanban remains the source of truth for dispatch/status, but comments/docs must use precise status labels so a pilot or staged report is not mistaken for production completion.

## Required vocabulary

Use these labels in card summaries, handoff comments, reports, and `todo.d/` phase files:

- `design done`: architecture/policy/report exists; no production artifact is implied.
- `pilot accepted`: bounded pilot was independently reviewed and accepted; it is not full scale unless explicitly stated.
- `staged-only`: artifacts exist under `artifacts/`, `docs/`, local staging, or GCS staging; no canonical KG promotion is claimed.
- `review-required`: producer implementation is ready for independent review but must not be called validated/done yet.
- `validated`: tester/QA checked the observable behavior, counts, endpoints, and artifact readability.
- `canonical promoted`: reviewed artifact was written to canonical `gs://jouvencekb/kg/v2/...` or equivalent canonical store.
- `production/full done`: full intended scope exists and runs; not just a design, pilot, or staged tranche.

Avoid bare `done` for anything except a terminal Kanban state. Prefer `design done`, `pilot accepted`, `staged-only accepted`, `validated`, or `canonical promoted`.

## Producer handoff rule

Every producer that blocks with `review-required` must leave a durable `kanban_comment` containing:

1. changed files/artifacts;
2. workspace/branch/PR if applicable;
3. observable behavior implemented;
4. exact test/validation commands with PASS/FAIL output;
5. residual risks; and
6. the intended reviewer/tester route.

A producer is routed when one of these is true:

- it has a linked child assigned to `reviewer`, `tester`, or `cto`; or
- it has a comment of the form `Orchestration: routed the review-required handoff to ungated reviewer card t_xxxxxxxx ...`.

The ungated-comment route is allowed for already-blocked producers because adding a parent edge from a blocked producer can keep the reviewer child in `todo` and deadlock dispatch.

## Watchdog rule

`scripts/txgnn_kanban_watchdog.py` is the board hygiene watchdog.

Default safe checks:

```bash
python scripts/txgnn_kanban_watchdog.py --json
python scripts/txgnn_kanban_watchdog.py --silent
```

Behavior:

- reads `${HERMES_KANBAN_DB}` or `~/.hermes/kanban/boards/txgnn/kanban.db`;
- scans blocked TxGNN producer cards with `review-required` in title/body/result/comment/current run metadata;
- reports whether each has a linked reviewer/tester child or an explicit routed-comment reviewer card;
- recognizes completed reviewer cards whose latest run metadata/summary includes `approved: true` and `accept_close_<producer>` as accepted reviews pending producer close;
- exits `0` when all routes are healthy;
- exits `2` when a missing route is found in dry-run mode;
- prints nothing with `--silent` when all routes are healthy.

Mutation mode:

```bash
python scripts/txgnn_kanban_watchdog.py --apply
```

`--apply` should be used only from a trusted board cron/operator context. It creates missing reviewer cards via the Hermes CLI and comments the producer; it does not write raw SQLite rows.

## Scratch artifact location rule

Do not create new TxGNN scratch reports or staged tranches under `.omoc/`.

Use:

- `artifacts/staged/...` for local staged outputs;
- `docs/...` for human-readable reports;
- `gs://jouvencekb/staging/<task-id>/...` for large staged artifacts;
- `.omoc/` only for legacy artifacts that already exist or a currently running old process that cannot safely be interrupted.

If an old process is still writing `.omoc`, do not delete it mid-run. Let it finish, copy/promote the useful outputs into `artifacts/`, `docs/`, or bucket staging, then clean `.omoc` only when no active command references it.

## Definitions of promotion

- `staged-only` is not canonical.
- `validated` is not canonical unless the validator explicitly checked canonical writes.
- `canonical promoted` is not necessarily `production/full done` if only a bounded tranche or partial relation set was promoted.
- `production/full done` requires the intended full scope, validation, review, and canonical promotion or an explicit accepted decision that no canonical promotion is needed.
