# PyG and embedding contracts

[← Documentation index](../README.md) · [KG architecture](kg-architecture-and-evidence.md) · [Feature registry](pyg-feature-registry.md) · [Lessons learned](lessons-learned.md)

The production representation is store-backed and neighbor-sampled. A monolithic
`HeteroData` pickle is useful only for bounded pilots or explicitly materialized
subsets; it is not the full-scale storage contract. Sequential edge minibatching
is useful for audits, builds, and non-message-passing models, but it is not the
final GNN training loader.

## Durable derived namespace

Canonical biological tables remain under `gs://jouvencekb/main`. The single current PyG
training build lives directly under:

```text
gs://jouvencekb/pyg/
```

This prefix is durable, derived, and replaceable. It is not subject to the
`staging/` lifecycle and is not a canonical biological layer. Every snapshot must
identify the exact canonical object generations/hashes from which it was built.
LaminDB may catalog these artifacts and their lineage, but LaminDB registration is
secondary to the simple GCS artifact contract.

Build candidates first under `gs://jouvencekb/staging/<build-id>/pyg/`, validate
them, replace the current payload under `pyg/`, then upload `manifest.json` last.

Canonical build layout:

```text
gs://jouvencekb/pyg/
├── manifest.json
├── node_maps/
├── adjacency/
├── feature_indices/
└── validation/
```

`feature_indices/` may contain node-aligned matrices, edge-aligned matrices,
masks, vocabularies and split indices. Raw text, sequences and one-to-many
evidence remain in canonical Parquet unless an explicit encoder/aggregation in
the feature registry materializes a fixed-shape training representation.

## Training loader decision

Use the loader that matches the learning task:

| Task | Runtime loader |
|---|---|
| Node classification or node-seed representation learning | `NeighborLoader` |
| Link prediction/classification, including Jouvence/TxGNN drug repurposing | `LinkNeighborLoader` |
| Exhaustive relation audit, feature build, or simple edge scorer | sequential edge minibatching |

For GNN training, the loader consumes a tuple of PyG stores:

```python
data = (feature_store, graph_store)
```

The `GraphStore` exposes relation-specific CSC adjacency arrays and stable edge
IDs. The `FeatureStore` gathers only the sampled node/edge rows and modalities.
Both stores open worker-local memory-mapped files from a cached immutable GCS
snapshot; random neighbor lookups must not issue repeated scans of canonical
Parquet or many tiny GCS reads.

### Disk-backed adjacency

For each heterogeneous edge type `(source_type, relation, target_type)`, publish:

- deterministic `node_id ↔ int64 index` maps;
- destination-sorted CSC `colptr` and `row` arrays;
- stable edge IDs/permutation maps aligning edge features and provenance;
- a technical reverse edge type for opposite-direction message passing;
- exact source generations, counts, hashes, split policy, and validation results.

The arrays are contiguous and memory-mappable. Looking up one destination's
neighbors becomes a slice `row[colptr[i]:colptr[i + 1]]`, rather than a scan of a
100M-edge Parquet corpus. The operating system pages in only the requested
regions. PyG's neighbor sampler still receives the full CSC index interface, so
the store must return pre-sorted memory-mapped tensors and must not rematerialize
or re-sort the graph at every job start.

### Job startup

1. Resolve `pyg/manifest.json` and verify that it matches
   the requested canonical snapshot.
2. Copy/cache the artifact once from GCS to a user-selected local directory
   (default: `REPO/data/pyg/`) and verify hashes.
3. Open `JouvenceGraphStore` and `JouvenceFeatureStore` over memory-mapped arrays.
4. Construct `LinkNeighborLoader` or `NeighborLoader` with a reviewed,
   relation-specific fanout policy.
5. Move only each sampled minibatch to the accelerator.

Do not neighbor-sample through random GCS or FUSE reads. Copy the complete build
with `gcloud storage cp --recursive`, then train from that local directory.

### Split and leakage gate

The message-passing adjacency for a training run may contain only training-visible
edges. Validation/test labels and their technical reverse edges must be absent.
For link prediction, `edge_label_index` is supervision; it is not permission to
leave the same held-out assertion in the sampling graph. Every snapshot/split must
ship an exact anti-leakage report.

## PyG representation

Store each relation independently with:

- relation-wise CSC adjacency arrays (`colptr`, `row`, edge ID/permutation);
- node maps `node_id ↔ node_index`;
- edge row maps `edge_key ↔ edge_pos`;
- reverse-edge mappings via `forward_edge_pos`;
- feature/evidence descriptors in the manifest;
- source identities, hashes, counts, and leakage policy.

Memory-map selected arrays and bound the runtime relation/sample scope. Stage
remote sidecars to worker-local storage before using local-path mmap loaders. Do
not require or publish a no-cap `heterodata/full_graph.pt` to claim architecture
readiness.

A bounded smoke proves executability only. It does not prove full-scale materialization, training stability, biological utility, or model quality.

## Manifest feature states

Every selected node type and relation must distinguish:

| State | Meaning | Allowed fallback |
| --- | --- | --- |
| `source-backed available` | A real sidecar exists for some or all rows | Use the sidecar where joined; report row coverage |
| `absent` | No source-backed sidecar exists | Model-side learned embedding, explicitly declared |
| `deferred` | Modality intentionally postponed | No claim of current coverage |
| `fallback` | Deterministic scaffold or learned representation supports execution | Must not be described as source-derived biological signal |

`available` never means complete coverage unless row-level parity proves it. Do not hide absence with zero vectors or pseudo-source embeddings.

The human-readable owner/join/tensor/leakage registry is
[`pyg-feature-registry.md`](pyg-feature-registry.md). Generated manifests may
mirror that policy for runtime validation, but do not replace it with an opaque
JSON-only registry.

## Node embeddings

Keep modalities separate and fuse downstream:

- protein sequence versus protein text;
- transcript cDNA/UTR versus gene text;
- molecule structure versus molecule text;
- ontology text versus numeric/categorical attributes.

Derived embeddings are immutable versioned features tied to exact source hashes, model revision, tokenizer/preprocessing, pooling, dimension/dtype, code version, and license. Regenerate into a new versioned path rather than silently replacing prior vectors.

## Edge/evidence embeddings

An edge embedding represents an assertion and its accepted evidence, not an arbitrary serialization of every payload.

- numeric evidence uses explicit normalization/encoding;
- categorical/text evidence uses versioned encoders;
- multiple evidence rows aggregate deterministically;
- output identity remains one vector per consumed edge/group;
- rich provenance payload stays in sidecars rather than the dense hot path;
- reverse edges reuse forward identity rather than inventing separate biological evidence.

## Leakage contract

For every feature or embedding, record:

- prediction tasks it may leak;
- temporal/source release cutoff;
- whether it is input, supervision, evaluation-only, or provenance;
- split/masking policy;
- exclusion of labels, split assignments, model predictions, or downstream target evidence from the embedding payload.

Clinical-trial outcome/status/intervention text is canonical metadata/feature material but may reveal `molecule_treats_disease`; it must be masked or partitioned for held-out treatment prediction.

## `clinical_trial` policy resolution

Some older PyG examples mention a `clinical_trial` node type or trial-related graph relations. They are not the current canonical doctrine.

The accepted policy is:

- ClinicalTrials.gov trial records are metadata keyed by NCT;
- edge-to-NCT links are evidence/link metadata for existing treatment assertions;
- trial text and deterministic vectors are feature sidecars;
- no `clinical_trial` node type or default graph adjacency is implied;
- any future trial-node topology requires a new explicit schema/science decision and leakage review.

Therefore, do not copy older commands containing `clinical_trial` into new guides or runbooks unless the conflict has first been resolved by a newer reviewed policy.

## Readiness claims

Use precise boundaries:

- sequential edge-stream contract implemented;
- bounded exporter/smoke passed;
- versioned neighbor-index snapshot built and independently validated;
- worker-local cache + mmap `GraphStore`/`FeatureStore` loading validated;
- `NeighborLoader`/`LinkNeighborLoader` multi-hop minibatch validated;
- split/reverse-edge anti-leakage gate passed;
- bounded training smoke passed;
- full intended neighbor-index snapshot complete;
- full training/evaluation complete.

Architecture readiness is not the same as full production training.

## Authoritative detailed sources

- [`../pyg_gnn_readiness_gate_t_825cfdcf.md`](../pyg_gnn_readiness_gate_t_825cfdcf.md)
- [`../pyg_manifest_metadata_contract_t_a3b15bc8.md`](../pyg_manifest_metadata_contract_t_a3b15bc8.md)
- [`../pyg_mapping_design.md`](../pyg_mapping_design.md)
- [`../edge_evidence_embedding_policy.md`](../edge_evidence_embedding_policy.md)
- [`../foundation_embedding_policy.md`](../foundation_embedding_policy.md)
- [`../clinical_trials_canonical_features_resolution_t_957a3640.md`](../clinical_trials_canonical_features_resolution_t_957a3640.md)
- PyG manifest QA proof is tracked by Kanban task `t_3df2bfc3`.
