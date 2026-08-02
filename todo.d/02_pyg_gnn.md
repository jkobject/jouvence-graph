# 02 — PyG / GNN

_Last verified: 2026-07-15. Kanban board `txgnn` remains the live source of truth._

Heavy-job guardrail: production/full PyG exports and training must run on `txgnn-worker` or another approved in-region worker with `gs://jouvencekb/main`. Do not run full-KG work through macOS GCS-FUSE.

## Current verdict

**PyG is training-ready in architecture, but full-KG model training is not done.**

### 2026-07-27 neighbor-loading correction

Sequential Parquet edge minibatching is retained for exhaustive audits, derived
artifact builds, and simple edge scorers. It is **not** the final multi-hop GNN
training loader. The accepted production direction is:

- one current derived build under `gs://jouvencekb/pyg/`;
- destination-sorted CSC adjacency plus technical reverse relations;
- user-selected local cache (default: `REPO/data/pyg/`) and memory-mapped
  `GraphStore`/`FeatureStore`;
- `NeighborLoader` for node-seed tasks and `LinkNeighborLoader` for Jouvence link
  prediction;
- split-aware adjacency that excludes validation/test labels and their reverse
  edges;
- small public helpers for resolve/cache/open/loader construction/batch
  inspection, demonstrated in notebook 07.

Detailed contract: `docs/guides/pyg-and-embedding-contracts.md`. Human feature
policy: `docs/guides/pyg-feature-registry.md`.

Validated runtime evidence:

- real `torch_geometric.data.HeteroData` artifact exists;
- representative export contains 5 node types and 7 forward relations plus reverse edges;
- a real heterogeneous GraphSAGE link-prediction smoke ran successfully and was independently rerun by reviewer;
- real node embeddings can populate `x` where present;
- learned fallback tensors replace structural one-column placeholders for missing node/edge rows;
- edge embeddings are exposed as `edge_attr` and consumed by the smoke predictor;
- sidecar/memmap export supports relation-scoped loading without requiring a 100M-edge monolithic pickle.

## Executed PyG/GNN proof

Representative accepted artifact:

- node types: `molecule`, `gene`, `disease`, `pathway`, `phenotype`;
- forward relations: 7;
- cap: 20,000 edges per selected relation;
- real `HeteroData` size: 5,388,959 bytes;
- tester GraphSAGE smoke: 2 epochs, 512 positive + 512 negative training edges, validation status `pass`, final train loss 0.5626, validation accuracy 0.8203;
- independent reviewer rerun: status `pass`.

This proves executable integration, **not biological model quality** and not whole-KG training.

## 16 GB RAM assessment

The current architecture is designed to fit a 16 GB worker by keeping graph arrays and embeddings in sidecars/memmaps and loading only selected relations/batches.

Measured bounded readiness smoke:

- build max RSS: 487,063,552 bytes;
- smoke max RSS: 488,390,656 bytes;
- process-tree peak: approximately 502 MB.

Therefore a bounded relation-scoped smoke fits comfortably in 16 GB. However:

- the complete 55,523,691-node graph with one dense float32 embedding per node would require about **53 GiB at 256 dimensions**, **106 GiB at 512**, or **159 GiB at 768**, before optimizer state, activations or graph tensors;
- a 100,080,390-edge `int64` forward `edge_index` is about **1.49 GiB**, or **2.98 GiB** with fully duplicated reverse indices;
- full dense all-node/all-relation materialization and end-to-end training have **not** been proven inside 16 GB.

So the correct answer is: **yes for the sidecar/memmap sampled architecture and bounded training; no evidence that a monolithic full-KG training job fits in 16 GB.** Full-scale training must remain relation/batch sampled, mmap-backed and independently measured on the 16 GB worker.

## Remaining work

1. Build and independently review the CSC neighbor index on an approved in-region
   worker; publish its payload under `pyg/` and `manifest.json` last.
2. The store, direct-copy verification, and ergonomic loader helpers are now
   implemented and shown in notebook 07; exercise them against the first build.
3. Add exact train/validation/test and reverse-edge anti-leakage validation.
4. Demonstrate a real heterogeneous two-hop sampled batch in notebook 07 without
   downloading or rebuilding the full artifact on the Mac.
5. Train/evaluate using bounded neighbor sampling and record peak RSS under the
   real 16 GB limit.
6. Run production model-quality and biological validation; current smoke accuracy
   is not such evidence.
7. Keep `paper` and `dataset` outside default message-passing topology.

## Definition of done

- all intended training node/relation types have a reviewed sidecar export path;
- loader never requires full graph or all embeddings in RAM;
- full intended training/evaluation executes under an explicit memory budget with measured peak RSS <16 GB;
- real/fallback node features and accepted edge features are consumed;
- biological/model-quality evaluation passes independently reviewed criteria.
