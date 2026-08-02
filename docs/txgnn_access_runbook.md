# Jouvence KG access runbook

This runbook is the default access path for workers using the Jouvence KG at `gs://jouvencekb/main`. Small bounded inspection may happen from the macOS worker environment, but heavy Jouvence work is VM-only. The filename and `txgnn-worker` VM name are retained compatibility identifiers.

Emergency guardrail (`t_d682b7ad`): heavy LaminDB/PyG/ReMap/embedding/full-KG jobs must run on `txgnn-worker` or another explicitly approved in-region worker. Use `gs://jouvencekb/main` as the source for those jobs and `gs://jouvencekb/staging` for temporary outputs. Do **not** run heavy reads/writes through `/Users/jkobject/mnt/gcs/...` / macOS GCS-FUSE. Future heavy cards must state `must_run_on=txgnn-worker`, preflight `hostname`, use `gcloud compute ssh` for worker launch/inspection, check for an existing related writer/process, and fail immediately if any heavy input/output path starts with `/Users/jkobject/mnt/gcs`.

Status from verification on 2026-06-23:

- GCS CLI access: verified with both `gcloud storage` and `gsutil`.
- FUSE mount: verified on macOS after macFUSE approval at `~/mnt/gcs/jouvencekb-kg`, mounted with `gcsfuse --foreground --implicit-dirs --only-dir kg jouvencekb ~/mnt/gcs/jouvencekb-kg`. This mount exposes bucket prefix `gs://jouvencekb/kg/`, so canonical KG paths appear under `~/mnt/gcs/jouvencekb-kg/v2/...`. DuckDB read of `v2/features/protein_sequence.parquet` via the mount succeeded. This is for small bounded/local inspection only; it is forbidden for heavy LaminDB/PyG/ReMap/embedding/full-KG work.
- Repo-local `.omoc` caches are retired. For small bounded Mac inspection, use the FUSE mount or direct `gs://...` access; if a bounded local cache is unavoidable, use `artifacts/cache/<task-id>/` and preserve original GCS URIs in reports. For heavy work, use `txgnn-worker`/approved VM and bucket-local `gs://...` paths instead.
- LaminDB instance `jkobject/jouvencekb`: remote SQLite and managed artifacts now live under `gs://jouvencekb/.lamin`; its repaired catalog points to `.lamin/lamin` and `main`, contains 79 artifacts, and passes SQLite `quick_check`.

Do not print tokens, DB URLs, or raw Lamin/GCloud credential files in logs.

## 1. GCS access: primary verified path

Check the active Google account without printing the email in shared logs:

```bash
gcloud auth list --filter=status:ACTIVE --format='value(account)' | sed 's/.*/<configured-account>/'
```

Verify the bucket root and subdirectories:

```bash
gcloud storage ls gs://jouvencekb/kg/v2/
gcloud storage ls gs://jouvencekb/kg/v2/edges/ | sed -n '1,5p'
gsutil ls gs://jouvencekb/kg/v2/
```

Observed sanitized output on 2026-06-21:

```text
ACTIVE_GCLOUD_ACCOUNT=<configured-account>

GCS_ROOT_LISTING
gs://jouvencekb/kg/v2/_removed_relations_20260618/
gs://jouvencekb/kg/v2/archive/
gs://jouvencekb/kg/v2/edges/
gs://jouvencekb/kg/v2/evidence/
gs://jouvencekb/kg/v2/metadata/
gs://jouvencekb/kg/v2/nodes/

GCS_EDGES_SAMPLE
gs://jouvencekb/kg/v2/edges/cell_line_derived_from_tissue.parquet
gs://jouvencekb/kg/v2/edges/cell_line_expresses_gene.parquet
gs://jouvencekb/kg/v2/edges/cell_line_from_organism.parquet
gs://jouvencekb/kg/v2/edges/cell_type_expresses_gene.parquet
gs://jouvencekb/kg/v2/edges/dataset_contains_cell_line.parquet

GSUTIL_ROOT_LISTING
gs://jouvencekb/kg/v2/_removed_relations_20260618/
gs://jouvencekb/kg/v2/archive/
gs://jouvencekb/kg/v2/edges/
gs://jouvencekb/kg/v2/evidence/
gs://jouvencekb/kg/v2/metadata/
gs://jouvencekb/kg/v2/nodes/
```

Use `gcloud storage` for new scripts; keep `gsutil` examples for compatibility with older worker notes.

## 2. Local scratch/cache policy

Default for small bounded/local inspection: read the canonical KG through direct GCS paths or the verified FUSE mount. Heavy jobs must not use these Mac FUSE paths; run them on `txgnn-worker`/approved in-region worker with `gs://jouvencekb/kg/v2`.

```text
/Users/jkobject/mnt/gcs/jouvencekb-kg/v2/edges/
/Users/jkobject/mnt/gcs/jouvencekb-kg/v2/evidence/
/Users/jkobject/mnt/gcs/jouvencekb-kg/v2/nodes/
/Users/jkobject/mnt/gcs/jouvencekb-kg/v2/features/
```

Do **not** create new `.omoc/gcs-cache/...` directories. `.omoc` is a retired legacy scratch/cache location and should not appear in new card specs, scripts, or docs except as historical context.

If a bounded local cache is unavoidable for performance or offline inspection, use a task-scoped path under:

```text
artifacts/cache/<task-id>/
```

Example:

```bash
mkdir -p artifacts/cache/<task-id>/{edges,evidence,nodes,features,raw}

gcloud storage cp \
  gs://jouvencekb/kg/v2/edges/gene_interacts_gene.parquet \
  artifacts/cache/<task-id>/edges/gene_interacts_gene.parquet
```

For raw/source archives under `gs://jouvencekb/kg/...`, put copied files under `artifacts/cache/<task-id>/raw/<source-or-slice-name>/` and keep the original `gs://...` path in the report or script arguments.

### DuckDB verification on FUSE or task-scoped cache for small bounded inspection

FUSE is acceptable only for small bounded/local inspection:

```bash
uv run --with duckdb python - <<'PY'
import duckdb
p = '/Users/jkobject/mnt/gcs/jouvencekb-kg/v2/nodes/organism.parquet'
con = duckdb.connect()
print('count', con.sql(f"select count(*) from read_parquet('{p}')").fetchone()[0])
print(con.sql(f"describe select * from read_parquet('{p}')").df().to_string(index=False))
PY
```

Or task-scoped cache:

```bash
uv run --with duckdb python - <<'PY'
import duckdb
p = 'artifacts/cache/<task-id>/nodes/organism.parquet'
con = duckdb.connect()
print('count', con.sql(f"select count(*) from read_parquet('{p}')").fetchone()[0])
PY
```

For relation/evidence audits, use DuckDB summaries before making schema decisions:

```bash
uv run --with duckdb python - <<'PY'
import duckdb
relation = 'gene_interacts_gene'
p = f'/Users/jkobject/mnt/gcs/jouvencekb-kg/v2/evidence/{relation}.parquet'
con = duckdb.connect()
print(con.sql(f"""
    select source, source_dataset, evidence_type, predicate, direction, count(*) as n
    from read_parquet('{p}')
    group by 1,2,3,4,5
    order by n desc
""").df().to_string(index=False))
PY
```


## 3. FUSE mount policy

The verified macOS FUSE mount for this bucket is user-local and restricted to the `kg/` prefix:

```bash
mkdir -p "$HOME/mnt/gcs/jouvencekb-kg"
gcsfuse --foreground --implicit-dirs --only-dir kg jouvencekb "$HOME/mnt/gcs/jouvencekb-kg"
```

In Hermes, run this as a tracked background process because `--foreground` is a long-lived daemon. Then verify from a separate command:

```bash
MNT="$HOME/mnt/gcs/jouvencekb-kg"
mount | grep jouvencekb
python3 - <<'PY'
from pathlib import Path
m = Path.home() / 'mnt/gcs/jouvencekb-kg'
for rel in ['v2/edges', 'v2/evidence', 'v2/nodes', 'v2/features']:
    p = m / rel
    print(rel, p.exists(), p.is_dir(), [x.name for x in list(p.iterdir())[:5]] if p.is_dir() else None)
PY
uv run python - <<'PY'
import duckdb
p = '/Users/jkobject/mnt/gcs/jouvencekb-kg/v2/features/protein_sequence.parquet'
print(duckdb.sql(f"select count(*) from read_parquet('{p}')").fetchall())
PY
```

Observed on 2026-06-23 after approving macFUSE in macOS System Settings:

```text
jouvencekb on /Users/jkobject/mnt/gcs/jouvencekb-kg (macfuse, nodev, nosuid, synchronous, mounted by jkobject)
v2/edges True True [...]
v2/evidence True True [...]
v2/nodes True True [...]
DuckDB read of v2/features/protein_sequence.parquet: 112051 rows
```

If the mount is absent or empty, prefer direct `gs://...` reads where supported, or copy only needed files into `artifacts/cache/<task-id>/` and read with DuckDB/PyArrow locally. A live `gcsfuse` process is not enough; require a real `mount` entry plus visible files.

## 4. LaminDB connection

Active instance slug for this project:

```text
jkobject/jouvencekb
```

The repo currently pins `lamindb==2.2.1` in `pyproject.toml`; an older executed notebook notes that the remote DB may be ahead of that package version and suggests `pip install lamindb>=2.4`. In this repo, use `uv run` so the workspace schema package is available.

### CLI checks

Workers on this Mac should load the local Lamin API key before Lamin commands. Do not echo the key:

```bash
export LAMIN_API_KEY="$(cat ~/.laminkey)"
```

Then run:

```bash
uv run lamin info
uv run lamin connect jkobject/jouvencekb
uv run lamin info
```

If `uv run lamin info` reports `User: anonymous`, first check whether `~/.laminkey` exists and was exported as above. If credentials are absent or expired, run:

```bash
uv run lamin login
uv run lamin connect jkobject/jouvencekb
```

Do not print `~/.lamin/*.env`, `~/.laminkey`, or raw credential contents; they may contain secrets.

Observed sanitized output on 2026-06-21 after auth/cache repair:

```text
Instance: jkobject/jouvencekb
 - branch: main
 - space: all
Details:
 - storage: gs://jouvencekb (None)
 - db: sqlite:////Users/jkobject/Library/Caches/lamindb/jouvencekb/.lamindb/lamin.db
 - modules: bionty, pertdb
Cache & settings:
 - cache: /Users/jkobject/Library/Caches/lamindb
 - user settings: /Users/jkobject/.lamin
 - system settings: /Library/Application Support/lamindb
User: jkobject

uv run lamin connect jkobject/jouvencekb
! The original path gs://jouvencekb/.lamindb/lamin.db does not exist anymore.
However, the local path /Users/jkobject/Library/Caches/lamindb/jouvencekb/.lamindb/lamin.db still exists, you might want to reupload the object back.
! SQLite file does not exist in the cloud, but exists locally: /Users/jkobject/Library/Caches/lamindb/jouvencekb/.lamindb/lamin.db
To push the file to the cloud, call: lamin disconnect
→ connected lamindb: jkobject/jouvencekb
exit_code 0
```

The output above is retained as a historical 2026-06 transcript. The remote
storage-root mismatch was resolved on 2026-07-27 under
`gs://jouvencekb/.lamin`; do not use the old root paths in new commands.

### Python checks

Default repo environment:

```bash
export LAMIN_API_KEY="$(cat ~/.laminkey)"
uv run python - <<'PY'
import lamindb as ln
print('lamindb', getattr(ln, '__version__', 'unknown'))
db = ln.DB('jkobject/jouvencekb')
print('db_instantiated', type(db).__name__)
print('artifact_count', ln.Artifact.filter().count())
print('collection_count', ln.Collection.filter().count())
PY
```

One-off newer package probe, useful if the repo pin becomes too old for the remote DB:

```bash
export LAMIN_API_KEY="$(cat ~/.laminkey)"
uv run \
  --with 'lamindb>=2.4' \
  --with 'lnschema-txgnn @ ./manage_db/lnschema_txgnn' \
  python - <<'PY'
import lamindb as ln
print('lamindb', getattr(ln, '__version__', 'unknown'))
db = ln.DB('jkobject/jouvencekb')
print('db_instantiated', type(db).__name__)
PY
```

Observed on 2026-06-21 after auth/cache repair:

```text
lamindb 2.2.1
→ connected lamindb: jkobject/jouvencekb
db_instantiated DB
Artifact count 0
Collection count 0
```

Interpretation: this was the historical pre-migration probe. The missing-DB
warning is no longer expected: current instance configuration and remote DB use
`gs://jouvencekb/.lamin`.

### Local Lamin SQLite cache

A working local Lamin cache exists at Lamin's default cache path:

```text
/Users/jkobject/Library/Caches/lamindb/jouvencekb/.lamindb/lamin.db
```

It was repaired non-destructively by copying the real remote DB from:

```text
gs://jouvencekb/lamin/.lamindb/lamin.db
```

A historical repo-local copy was once used for direct SQLite inspection under `.omoc/gcs-cache/lamin/lamin.db`; this is retired and should not be recreated. If direct SQLite inspection is needed, use Lamin's default cache path above or copy into a task-scoped `artifacts/cache/<task-id>/lamin/` directory.

Treat direct SQLite access as a local inspection fallback. For normal KG relation audits, prefer the canonical FUSE/GCS Parquets unless the task explicitly needs Lamin registries.

## 5. Worker decision tree

1. Need canonical KG Parquets?
   - If the task is heavy (LaminDB full/bulk sync, production/full PyG/GNN, ReMap scaling, embedding/full-KG scan, all-relation read, or bulk canonical KG read/write), do not use the Mac FUSE root. Run on `txgnn-worker`/approved in-region worker with `gs://jouvencekb/kg/v2` and fail if any heavy path starts `/Users/jkobject/mnt/gcs`.
   - For small bounded/local inspection only, try the verified FUSE root `/Users/jkobject/mnt/gcs/jouvencekb-kg/v2/` or direct `gcloud storage ls gs://jouvencekb/kg/v2/`.
   - If a local copy is unavoidable, copy only needed Parquets into `artifacts/cache/<task-id>/{edges,evidence,nodes,features,raw}`.
   - Query FUSE, direct GCS-capable readers, or task-scoped cache files with DuckDB/PyArrow.
2. Need full tree filesystem semantics?
   - Use `~/mnt/gcs/jouvencekb-kg` if already mounted.
   - If absent, do not wait for macFUSE when direct GCS or targeted task-cache copies are enough.
3. Need LaminDB registries?
   - Run `uv run lamin info`.
   - If anonymous, export `LAMIN_API_KEY` from `~/.laminkey` using the CLI-check command above; if that file is missing/expired, run `uv run lamin login` or ask the operator for LaminHub auth/permissions.
   - Then run `uv run lamin connect jkobject/jouvencekb` and the Python `ln.DB(...)` probe.
4. Need to update canonical KG?
   - Build and validate locally first.
   - Use explicit GCS write/copy commands only after validation gates and human review where relevant.

## 6. Common pitfalls

- Do not use `/home/ubuntu/data` on this macOS worker.
- Do not assume `/mnt/gcs/jouvencekb` exists; this path is from older/Linux-oriented notes.
- Do not convert RNA/gene-level source rows into protein relations just because a gene-to-protein mapping exists.
- Do not copy whole bucket directories unless the task explicitly requires it; most audits need a handful of Parquets.
- Do not print Lamin/GCloud credential file contents.
- Do not treat historical repo-local Lamin DB copies as equivalent to a verified LaminDB connection; verify with `uv run lamin connect jkobject/jouvencekb` and a Python registry probe.
- Do not run the `lamin disconnect` cloud-push suggestion from Lamin's warning unless explicitly approved; it is a remote write to the bucket.
