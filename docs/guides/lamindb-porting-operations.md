# TxGNN → LaminDB porting operations

[← Documentation index](../README.md) · [Lessons learned](lessons-learned.md) · [Agent context](agent-context.md)

This page is the durable operating model for moving the Jouvence KG into LaminDB. It explains the infrastructure, the July 2026 stall, what counts as progress, and the gates for resuming safely. The Kanban board `txgnn` remains the live task/status source of truth; figures below are dated evidence, not a substitute for a live preflight.

## Target architecture

```text
gs://jouvencekb/main
  nodes/*.parquet
  edges/<relation>.parquet
  evidence/<relation>.parquet
             │
             │ in-region reads only for heavy work
             ▼
txgnn-worker (europe-west1-b)
  clean task-specific clone/worktree
  streaming relation sync
  durable subchunk telemetry
             │
             ▼
LaminDB instance: jkobject/jouvencekb
  KGEdge
  KGEdgeEvidence
  artifacts / schema / provenance
```

The macOS GCS-FUSE path `/Users/jkobject/mnt/gcs/jouvencekb/main` is restricted to small, bounded inspection. Bulk syncs from the Mac caused repeated remote object reads and avoidable GCS egress. Heavy scans and writes must use `gs://jouvencekb/main` from an approved in-region worker.

## Hard safety gates

Before any write-capable sync:

1. Confirm the card explicitly authorizes the exact relation, offset, and maximum row count.
2. Confirm `hostname` is `txgnn-worker` (or another explicitly approved in-region worker).
3. Confirm no related writer/supervisor already exists. One logical writer may have a shell/`uv`/Python PID tree; distinguish that from multiple independent writers.
4. Confirm every heavy input/output path is remote/in-region and none starts with `/Users/jkobject/mnt/gcs`.
5. Connect explicitly to `jkobject/jouvencekb` and prove the resulting owner/name before writing. The July 2026 worker migration observed a preserved config resolving to `jkobject/repo`; treat that as a hard blocker until corrected and independently verified.
6. Use a clean, task-specific Git clone/worktree. Never deploy from a dirty shared checkout.
7. Keep the old validated offset until the new tranche has selected-live proof. A launched or partially processed tranche does not move the checkpoint.

TxGNN tasks must not manage resources belonging to other projects. In particular, `pert-gym-worker-eu` and pert-gym crons/disks/processes are outside this runbook even when they share the same GCP project.

## What counts as real progress

A running VM, supervisor, shell, `uv`, or Python PID is only liveness. Real progress requires a durable change after a committed subchunk.

Persist JSONL or an equivalent atomic record after every subchunk with at least:

- relation and run identity;
- source window and current committed offset;
- source rows read;
- edge/evidence rows attempted, upserted, and verified;
- selected-live edge/evidence counts and mismatch count;
- elapsed time and throughput;
- RSS, free disk, I/O wait, and `last_progress_at`;
- classification such as `progressing`, `stalled_before_first_subchunk`, `stalled_after_partial_commit`, `verification_failed`, or `complete`.

An offset may advance only when:

```text
selected-live edges    == selected source edges
selected-live evidence == selected source evidence
mismatch count         == 0
summary/return code     == successful
```

If there is no durable delta for the configured threshold, terminate the child process group, preserve logs/telemetry, classify the failure, and stop after at most two automatic relaunches. Never create a second writer to “unstick” the first.

## July 2026 `enhancer_regulates_gene` stall

The last validated checkpoint was `10,315,000`. The attempted `10,315,000 → 14,315,000` continuation did not validate even its first subchunk; its retained `progress.json` showed `completed=[]`. Therefore `10,315,000`, not `14,315,000`, remains the safe resume point until newer reviewed evidence says otherwise.

The root cause was in the sync read/materialization design, not insufficient disk:

1. The implementation requested a 1M-row relation window at offset 10.315M.
2. `_read_limited_parquet` decoded from physical row zero and discarded the prefix to reach the offset.
3. Edge and evidence frames for the full million-row tranche were materialized before the first 5k subchunk could commit.
4. Per-subchunk code then repeated the expensive prefix reads.
5. Relation-wide ORM counts added further SQLite/index pressure.

This produced a process with roughly 3.6 GB RSS and historical I/O-wait state but no first committed subchunk. Empty stderr and a live PID did not mean useful work was happening.

Required remediation before a long retry:

- stream row groups once through the requested window;
- build and commit bounded subchunks without full-tranche Pandas materialization;
- remove global relation counts from the write-critical path;
- verify selected keys for each committed tranche;
- flush telemetry after each subchunk;
- pass a 10k idempotent smoke, then a 50k smoke, before any 1M+ continuation;
- obtain independent review before increasing cadence.

Detailed forensic evidence: [`../../artifacts/reports/t_d7f9c01a/root_cause.md`](../../artifacts/reports/t_d7f9c01a/root_cause.md).

## Worker sizing and why local disk exists

TxGNN's canonical Parquets live on GCS, but the worker still needs local storage for:

- the OS, repository/worktrees, Python environments, and logs;
- LaminDB's local SQLite/config state when applicable;
- SQLite WAL/temp/index working space during upserts and verification;
- bounded source buffers, checkpoints, and task artifacts;
- deliberately rebuildable caches.

It does **not** need a local copy of the whole canonical KG.

### Verified migration snapshot — 2026-07-11

- active boot disk: `txgnn-worker-200gb-t3cf62bd8`, 200 GB `pd-standard`;
- post-migration root: 194 GiB total, 26 GiB used, 169 GiB free;
- VM subsequently right-sized to `e2-standard-4` (4 vCPU, 16 GB RAM);
- old detached rollback disk: `txgnn-worker`, 500 GB `pd-standard`;
- recovery snapshot: `txgnn-worker-rollback-t3cf62bd8-20260711`;
- intentionally omitted rebuildable caches: about 46 GB under `~/.cache` and 96 GB under `repo/artifacts/cache`, mostly stale ReMap/UCSC UDC sparse caches.

The old 500 GB disk was oversized because caches and sparse intermediate files accumulated, not because the useful TxGNN KG requires 500 GB locally. A 200 GB disk leaves ample working margin while canonical data stays in-region on GCS.

Do not delete the rollback disk or snapshot based on this page. Cleanup is destructive and requires an independent recovery review plus explicit authorization. See [`../txgnn_worker_disk_migration_t_3cf62bd8.md`](../txgnn_worker_disk_migration_t_3cf62bd8.md) for checksums, rsync/fsck evidence, rollback instructions, and dated cost estimates.

## Cost and automation policy

- Compute is allowed only for a bounded TxGNN window with a named owner/task and expiry.
- A TxGNN cost guard must target only `txgnn-worker`; it must not stop or reinterpret VMs from pert-gym or other projects.
- A monitor should run hourly (or every two hours for a stable long run) and report only meaningful deltas, stalls, errors, relaunches, or gate changes.
- Monitoring must distinguish local observability failure (for example macOS DNS/OAuth/Compute API failure) from a remote workload failure. “Cannot observe” is not “remote process failed.”
- Stop compute after completion, a hard blocker, repeated no-progress, or expiry. Stopping the VM preserves persistent disks; deletion is a separate decision.

## Safe resume sequence

1. Read the current Kanban card and verify this page has not been superseded.
2. Verify live GCP state, connected Lamin instance, clean source commit, and absence of a writer.
3. Run targeted unit tests for the streaming reader, progress schema, selected-key verification, and stall timeout.
4. Run a 10k idempotent write smoke from the validated offset or another explicitly safe range.
5. Verify edge and evidence parity with mismatch `0`; confirm no residual writer.
6. Run the same gate at 50k.
7. Obtain tester/reviewer acceptance.
8. Increase tranche size gradually while the hourly monitor proves real throughput and stable memory/I/O.

Never skip directly from “process starts” to a 1M/4M production continuation.

## Evidence and live-state boundaries

Durable evidence:

- disk migration: [`../txgnn_worker_disk_migration_t_3cf62bd8.md`](../txgnn_worker_disk_migration_t_3cf62bd8.md);
- stall forensic: [`../../artifacts/reports/t_d7f9c01a/root_cause.md`](../../artifacts/reports/t_d7f9c01a/root_cause.md);
- source isolation report: [`../../artifacts/reports/t_ade56294/source_isolation.md`](../../artifacts/reports/t_ade56294/source_isolation.md);
- broader Lamin status mirror: [`../../todo.d/01_lamindb.md`](../../todo.d/01_lamindb.md).

Live values such as VM status, current offset, writer PID, remaining rows, and cron state must always be re-queried. Do not copy a dated snapshot from this wiki into a completion claim.
