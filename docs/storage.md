# Jouvence KG storage

Stable data, the single derived PyG artifact, LaminDB state, and temporary
candidates share `gs://jouvencekb` under disjoint prefixes. Only `raw/` and
`main/` are canonical public data. `pyg/` is a reproducible derived training
layer. The bucket currently has no lifecycle rule; staging cleanup is explicit.

## Namespaces

| Purpose | URI | Contract |
|---|---|---|
| Stable public data | `gs://jouvencekb/{raw,main}` | Durable, reviewed data |
| Current derived PyG build | `gs://jouvencekb/pyg` | One replaceable build, reproducible from exact `main/` generations; not canonical biology |
| Temporary candidates | `gs://jouvencekb/staging` | Non-canonical; delete explicitly after promotion/readback |
| LaminDB internals | `gs://jouvencekb/.lamin` | Hidden runtime/catalog state; not a public data layer |

## Stable layout

```text
gs://jouvencekb/
├── README.md
├── .lamin/
│   ├── .lamindb/lamin.db
│   └── lamin/...
├── raw/
│   └── <source>.<native-format>
├── main/
│   ├── nodes/<node-type>.parquet
│   ├── edges/<relation>.parquet
│   ├── edges_inferred/<relation>.parquet
│   ├── evidence/<relation>.parquet
│   ├── evidence_inferred/<relation>.parquet
│   ├── features/<feature>.parquet
│   └── embeddings/<entity>-<modality>-<model>.parquet
└── pyg/
    ├── manifest.json
    ├── node_maps/
    ├── adjacency/
    ├── feature_indices/
    │   ├── nodes/
    │   ├── edges/
    │   └── splits/
    └── validation/

gs://jouvencekb/
└── staging/
    └── <task-or-build-id>/
        └── ... temporary candidate outputs ...
```

GCS has no real empty directories. The inferred prefixes are absent while there
are no accepted inferred tables; do not publish placeholders or synthetic empty
Parquets.

## Layer semantics

- `raw/`: selected upstream snapshots in native formats. If a source cannot be
  mirrored for licensing or scale reasons, document the external release and
  checksum in the catalog; do not create a fake raw object.
- `main/nodes/`: canonical entity registries.
- `main/edges/`: observed canonical graph assertions.
- `main/edges_inferred/`: independently reviewed inferred assertions only.
- `main/evidence/`: source-level support rows for observed edges.
- `main/evidence_inferred/`: provenance/evidence for accepted inferred edges.
- `main/features/`: non-topological node/edge feature sidecars.
- `main/embeddings/`: accepted learned vector tables, one flat Parquet object per
  entity/modality/model release.
- `pyg/`: the current derived training build: node index maps, disk-backed CSC
  adjacency, aligned feature arrays/masks, split metadata, and validation
  reports. It is reproducible from exact `main/` generations and never replaces
  canonical Parquet.

The old `metadata/` layer was not a coherent data layer. Clinical-trial indexes,
trial links, and mutation support rows are feature sidecars. Promotion receipts,
manifests, summaries, and provenance reports belong in versioned `docs/`. The
old `proof/mutation_in_gene_containment_proof.parquet` duplicated the 2,599,525
row-level containment proofs already encoded in
`evidence/mutation_in_gene.parquet` and is not a separate canonical layer.

## Access

`manage_db.kg_storage.open_kg_root(uri)` accepts plain paths or `gs://` URIs.
Use:

- canonical read root: `gs://jouvencekb/main`
- raw source root: `gs://jouvencekb/raw`
- candidate write root: `gs://jouvencekb/staging/<task-or-build-id>`
- derived PyG root: `gs://jouvencekb/pyg`
- LaminDB storage root: `gs://jouvencekb/.lamin`

For PyG training, do not mount the bucket. Copy `gs://jouvencekb/pyg/*` once to
a user-selected local directory (default: `REPO/data/pyg/`) with
`gcloud storage cp --recursive`, verify the manifest, and open the local files
with memory mapping.

## Publication protocol

1. Build and validate in local scratch or `gs://jouvencekb/staging`.
2. Freeze an immutable candidate generation and review its row/schema/evidence
   contract.
3. Publish one flat Parquet to the appropriate `main/` layer using a
   destination-generation precondition (`ifGenerationMatch=0` for a new name,
   or the reviewed current generation for an intentional replacement).
4. Read back CRC32C, size, Parquet footer, row count, and schema.
5. Refresh and check `docs/parquet-catalog/` against the live bucket.
6. Delete the staging candidate explicitly only after canonical readback passes.

For PyG, build under `staging/<build-id>/pyg`, validate exact source generations,
adjacency, feature alignment and split leakage, replace the current `pyg/`
payload, then upload `manifest.json` last. There is only one PyG build for now.

Never create `staged/`, `staging/`, `metadata/`, `proof/`, or `archive/` in the
stable bucket or inside `main/`. Candidate bundles, checkpoints, and transient
reports belong only in `gs://jouvencekb/staging`; LaminDB-owned runtime state
belongs only below `.lamin/`.

## Atomic writes

Local/FUSE writes use temporary files and atomic rename through
`manage_db.kg_storage`. Direct canonical GCS writes require generation
preconditions; a rename alone is not a concurrency guard on object storage.
Append mode rewrites a complete validated Parquet and must not be used to mutate
an unreviewed canonical generation in place.

## Catalog and provenance

The live machine-readable inventory and generated dataset pages live under
`docs/parquet-catalog/`. Source/release/license fields remain in each Parquet
where applicable. Promotion receipts and detailed migration manifests are Git
artifacts, not a bucket namespace.

Live bucket readback on 2026-07-28 reported no lifecycle, soft-delete, or object
versioning configuration. Clean `staging/` explicitly; it is not an archive.
