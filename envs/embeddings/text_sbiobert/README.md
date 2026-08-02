# Text / ontology summary embeddings

Task scaffold: `t_8892763b`.

Runtime requirements: CPU okay for smoke; GPU recommended for full run.

Install/run with the repo `uv` dependency group rather than mixing model families in one environment.

```bash
uv run --group embeddings-text python -m manage_db.build_real_embeddings --kg-root <kg_or_prepared_local_cache_root> --output-dir artifacts/staged/<task-id>/text_sbiobert_full --text-limit-per-table 100000000 --skip-edge-embeddings --clean
```

If the macOS FUSE root is stale/unavailable, use canonical GCS plus a local cache instead of `/Users/jkobject/mnt/gcs/...`:

```bash
uv run --with sentence-transformers python -m manage_db.build_real_embeddings --gcs-kg-root gs://jouvencekb/main --local-cache-dir artifacts/cache/<task-id>_kg_text --output-dir artifacts/staged/<task-id>/text_sbiobert_smoke --text-limit-per-table 1 --skip-edge-embeddings --clean
```

Outputs must remain under `artifacts/staged/<task-id>/...` until validation and independent review. Do not write `.omoc`; do not promote to canonical `features/` from a production worker without a reviewer gate.
