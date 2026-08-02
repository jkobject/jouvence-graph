# Get started with Jouvence data

This is the public, agent-readable route for a small, read-only exploration of
Jouvence. Canonical Parquet is the working data plane for identities that have
bucket read access and supply their own billing project. **A new external friend
is currently blocked until the maintainer grants that identity read-only bucket
access.** The LaminDB mirror is partial and is also **not yet a working external
access path**. Fixture mode works immediately for anyone.

## 1. Install and prove fixture mode

Requirements: Git, Python 3.11+, and [`uv`](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/jkobject/jouvence-graph.git
cd jouvence-graph
uv sync --group dev
env -u JOUVENCE_DATA_MODE \
    -u JOUVENCE_BILLING_PROJECT \
    -u GOOGLE_APPLICATION_CREDENTIALS \
    uv run python scripts/verify_data_quickstart.py --mode fixture
```

Expected: one JSON object with `"mode": "fixture"` and `"status": "pass"`.
Fixture mode is deterministic, creates only a temporary local KG, needs no
cloud account, and is the right first check for a clean clone or coding agent.
The public notebooks also default to fixture mode.

For the interactive product, run the independent no-cloud viewer smoke with
`uv run jouvence-viewer --fixture-smoke`. A manifest-verified local bundle or
reviewed requester-pays viewer bundle is launched with `--data-root`; see the
[bilingual local viewer installation guide](viewer-install.html). The viewer
rejects the raw canonical `kg/v2` root rather than scanning full tables on a
laptop. The public/static viewer shows only a top-evidence summary. A compatible
localhost bundle exposes complete evidence through stable 10-row pages (hard
maximum 50 rows per request); dossier exports remain bounded to 50 evidence
rows and state their total/returned/truncated status.

Mode detection:

- unset or `JOUVENCE_DATA_MODE=fixture`: local fixture/notebook path;
- `JOUVENCE_DATA_MODE=live`: canonical GCS reads, which also require
  `JOUVENCE_BILLING_PROJECT`;
- `JOUVENCE_LAMIN_LIVE=1`: separate opt-in to the exact LaminDB instance; do
  not infer this from GCS mode.

All examples below are read-only. Nothing in this quickstart grants permission
to write canonical GCS or LaminDB data.

## 2. Authenticate for canonical Parquet

Canonical root: `gs://jouvencekb/main`. Its main layers are:

```text
nodes/<node_type>.parquet
edges/<relation>.parquet
evidence/<relation>.parquet
features/.../*.parquet
metadata/.../*.parquet
```

The bucket uses requester-pays. A collaborator supplies a Google Cloud project
that they are authorized to bill; Jouvence never supplies or embeds one.

Prerequisites in the consumer project:

1. billing is enabled;
2. the caller can use the Cloud Storage JSON API (`storage.googleapis.com`);
3. the caller has `serviceusage.services.use` on the consumer project (usually
   via `roles/serviceusage.serviceUsageConsumer`);
4. the caller has Google Application Default Credentials (ADC);
5. the maintainer has granted that exact identity `roles/storage.objectViewer`
   (or equivalent object get/list permissions) on `gs://jouvencekb`.

The bucket had no `allUsers` or `allAuthenticatedUsers` read binding in the July
2026 audit. It is not anonymous/public-download data. Request the minimum
read-only grant from the maintainer; this guide does not change bucket IAM.

One-time client setup:

```bash
gcloud auth application-default login
export JOUVENCE_DATA_MODE=live
export JOUVENCE_BILLING_PROJECT='<consumer-billing-project>'
```

Non-secret preflight:

```bash
gcloud auth application-default print-access-token >/dev/null
uv run python scripts/verify_data_quickstart.py --mode live
```

The live smoke is also the definitive permission/API/requester-pays check; an
ADC token alone does not prove bucket access. Do not commit ADC files,
service-account JSON, API keys, `.env`, or the billing
project value. ADC user login is preferred; if an organization supplies a
service account, keep its credential outside this repository and apply the
minimum read/list permissions.

## 3. Inventory, schema, and bounded samples

The committed catalog lists tables without touching GCS:

```bash
uv run python scripts/parquet_catalog.py check
uv run python - <<'PY'
import json
from pathlib import Path
inventory = json.loads(Path('docs/parquet-catalog/inventory.json').read_text())
for item in inventory['datasets'][:20]:
    print(item['layer'], item['name'], item['uri'])
PY
```

Use the generated [`Parquet catalog`](parquet-catalog/index.md) for per-table
schemas and dated counts. Counts are inventory facts, not proof of live state.
Do not run the catalog's all-table live refresh on a laptop.

This verified smoke reads three gene rows plus bounded prefixes of one edge and
its evidence table. `read_bounded_parquet` seeks by row group, selects columns,
and enforces a hard 10,000-row ceiling. The join is over 100 assertion rows and
1,000 evidence rows; it is illustrative, not complete.

```bash
uv run python scripts/verify_data_quickstart.py --mode live
```

Inspect specific node, edge, and evidence schemas/samples with the same helper:

```bash
uv run python - <<'PY'
import os
from manage_db.public_notebooks import read_bounded_parquet

root = 'gs://jouvencekb/main'
project = os.environ['JOUVENCE_BILLING_PROJECT']
for relative, columns in [
    ('nodes/gene.parquet', ['id', 'name', 'source']),
    ('edges/disease_associated_gene.parquet',
     ['relation', 'x_id', 'x_type', 'y_id', 'y_type', 'source']),
    ('evidence/disease_associated_gene.parquet',
     ['relation', 'x_id', 'x_type', 'y_id', 'y_type', 'source',
      'source_record_id']),
]:
    frame = read_bounded_parquet(
        f'{root}/{relative}', columns=columns, limit=5,
        billing_project=project,
    )
    print(relative, frame.dtypes.to_dict(), frame.to_dict('records'))
PY
```

Join bounded assertion and evidence prefixes without full materialization:

```bash
uv run python - <<'PY'
import os
from manage_db.public_notebooks import bounded_edge_evidence_join

rows = bounded_edge_evidence_join(
    'gs://jouvencekb/main',
    'disease_associated_gene',
    edge_limit=100,
    evidence_limit=1000,
    billing_project=os.environ['JOUVENCE_BILLING_PROJECT'],
)
print(rows.head(10).to_string(index=False))
PY
```

`edges/` contains deduplicated graph assertions. `evidence/` contains
source-specific support, scores, studies, assays, and provenance; join on
`relation, x_id, x_type, y_id, y_type`, not row position. A missing row in a
bounded prefix is not evidence of biological absence.

## 4. Cost and laptop safety

Requester-pays charges the consumer project for requests, bytes read, and any
egress. Stay in the bucket's region when scaling. On a laptop:

- read one named object and selected columns;
- keep helper limits at or below 10,000 rows;
- inspect Parquet footers/row groups instead of calling `pandas.read_parquet`
  followed by `.head()`;
- do not glob/read all relations, count the full KG, bulk-copy the bucket, run a
  LaminDB sync, build a production PyG export, train on the full graph, or scan
  all embeddings;
- do not use macOS GCS-FUSE for broad scans.

Heavy/all-relation work belongs on the explicitly approved in-region Jouvence
worker and requires its own reviewed task. This quickstart does not start one.

## 5. LaminDB: current BLOCK and truthful fallback

Target instance: `jkobject/jouvencekb`. LaminDB is a queryable registry/catalog
mirror; it is not the canonical data plane. Its edge/evidence coverage is
currently partial, and an empty Lamin result does not mean a canonical Parquet
assertion is absent.

External access is currently **BLOCKED**. A July 2026 clean-home probe showed:

- anonymous connection requires LaminHub authentication;
- a fresh authenticated connection did not obtain usable instance/storage
  access;
- the historical maintainer setup depended on a repaired local SQLite cache
  because the instance's configured remote database object path was missing.

Therefore, cloning the repository plus `lamin login` is not presently enough.
Use canonical Parquet. The maintainer still needs to repair/republish the remote
instance database path, configure collaborator read access, invite each
LaminHub account if the instance remains private, and reproduce a fresh-home
read before this section can become PASS.

After the maintainer explicitly confirms access, an external collaborator would
use their own LaminHub account/token (never a shared maintainer token):

```bash
uv sync --group dev
uv run lamin login
uv run lamin connect jkobject/jouvencekb
export JOUVENCE_LAMIN_LIVE=1
uv run python - <<'PY'
from manage_db.public_notebooks import query_lamindb_node
print(query_lamindb_node('gene', 'ENSG00000141510', limit=1).to_string(index=False))
PY
```

That is a future read-only smoke command, not a claim that access works today.
Do not use `lamin disconnect`, create/update/delete registries, or re-upload a
database while following this guide.

## 6. Useful first workflows and limits

1. **Relation inventory/schema:** start with the committed Parquet catalog, then
   inspect one live footer/sample.
2. **Source provenance:** pair one `edges/<relation>.parquet` object with the
   same-named `evidence/<relation>.parquet`; retain source/predicate/study fields.
3. **Bounded biological question:** choose one stable identifier and one
   relation, inspect a small neighborhood, and treat outputs as source-backed
   associations—not causality, efficacy, safety, or clinical advice.
4. **Small PyG sample:** install `uv sync --group dev --group gnn`, build the
   fixture-backed sample via `build_sampled_pyg`, and use
   [`notebooks/05_sampled_pyg_heterodata.ipynb`](../notebooks/05_sampled_pyg_heterodata.ipynb). Never
   present it as full-KG training or model-quality validation.
5. **Embeddings:** retrieve only an immutable, reviewed URI explicitly published
   by the project. `JOUVENCE_EMBEDDING_URI` has no default because accepted
   release/license/latest-pointer contracts are still incomplete.

Known limitations: KG coverage and the Lamin mirror are incomplete; some
relations have no evidence table; canonical status changes only through
reviewed promotion; fixture data is illustrative; bounded samples are not
counts or absence tests; current PyG/GNN evidence is a runtime smoke, not a
validated repurposing model; no output is a therapeutic recommendation.

For interactive examples, continue with the fixture-backed
[`public notebooks`](../notebooks/README.md). For durable semantics, read
[`KG architecture and evidence`](guides/kg-architecture-and-evidence.md).