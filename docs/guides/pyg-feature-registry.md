# PyG feature registry

[← Documentation index](../README.md) · [PyG and embedding contracts](pyg-and-embedding-contracts.md) · [Storage](../storage.md)

This Markdown page is the human-readable policy registry for features consumed by Jouvence training. It records ownership, join keys, tensorization, missingness, and leakage boundaries. Generated manifests may mirror these fields for machines, but a JSON manifest does not replace this reviewable policy.

Status date: **2026-07-27**. Counts below come from the migrated live Parquet catalog and accepted Kanban handoffs. Re-check the live catalog and immutable artifact manifest before a new training release.

## Governing rules

1. `main/` remains the canonical biological source. `pyg/` contains only reproducible derived training artifacts.
2. A feature is owned by a node, an edge assertion, evidence, or external context. These owners are not interchangeable.
3. Every training tensor needs a fixed representation, dimension, dtype, missing-value policy, and leakage decision.
4. Raw text, sequences, evidence rows, source payloads, and provenance remain inspectable sidecars. They are not blindly concatenated into `x` or `edge_attr`.
5. Missing source payload and unfinished embedding computation are different states and must be counted separately.
6. Splits are part of the feature contract. A feature safe for one prediction task may leak another.

## Owner and join registry

| Family | Owner | Join key | Runtime representation | Missing policy | Default training role |
|---|---|---|---|---|---|
| Canonical node attributes | node | `(node_type, id)` | selected numeric/categorical encodings | mask + learned fallback where justified | input |
| Text embeddings | node | `(node_type, node_id)` | dense vector, modality-specific projection | explicit mask; never fake a source vector | input, subject to leakage review |
| Sequence embeddings | node | `(node_type, node_id)` | dense vector, modality-specific projection | explicit mask | input |
| Molecule fingerprint | node | `(molecule, node_id)` | sparse bits or reviewed dense projection | explicit mask | input |
| Numeric columns physically on an edge table | edge assertion | `(relation, x_id, x_type, y_id, y_type)` | relation-specific `edge_attr` tensor | mask/imputation fitted on train only | input |
| Evidence rows | evidence | `edge_key` or full assertion key | raw sidecar plus reviewed aggregates | zero-to-many rows per edge | provenance by default; input only after leakage review |
| Clinical-trial linkage/text | evidence/context | `edge_key`, `nct_id` | separate encoder or approved aggregate | absent if no linked trial | evaluation/provenance by default for treatment prediction |
| ReMap CRM support | contextual support | `enhancer_id`, `tf_gene_id`, `support_entity_id` | separate relation-specific support encoder | absent unless matching support exists | not generic `edge_attr`; review required |
| Split labels/masks | supervision | edge or node integer index | boolean/index tensors | no fallback | supervision/evaluation only |

## Current node-embedding coverage

A source embedding is only expected where a reviewed source payload exists. The denominator therefore needs two views:

- **node coverage**: how many canonical target nodes have this modality;
- **source-job parity**: whether every eligible source payload was embedded.

| Node type / modality | Embedded rows | Relevant node denominator | Node coverage | Current interpretation |
|---|---:|---:|---:|---|
| cell line text | 1,140 | 1,183 | 96.37% | Embedding count matches textual-summary input; 43 nodes lack that reviewed text payload. |
| cell type text | 3,135 | 3,513 | 89.24% | Embedding count matches textual-summary input; upstream ontology text is incomplete. |
| disease text | 26,395 | 41,859 | 63.06% | Embedding count matches OpenTargets textual-summary input; missing nodes lack the selected reviewed description. |
| gene text | 212,029 | 267,830 physical gene rows | 79.17% | Embedding count matches textual-summary input. This physical denominator includes non-human/quarantined identities and is not the exact human-ENSG training denominator. |
| human ENSG gene genomic sequence | 78,644 | 81,715 eligible ENSG genes | 96.24% | Canonical promoted and independently accepted; 3,071 missing are explicit, not an unfinished hidden checkpoint. |
| molecule SMILES | 18,614 | 31,007 | 60.03% | Embedding count matches the reviewed valid-SMILES/fingerprint input set; do not invent structures for the other molecules. |
| molecule text | 22,230 | 31,007 | 71.69% | Embedding count matches textual-summary input. |
| pathway text | 37,492 | 48,575 | 77.18% | Mostly GO-backed descriptions; Reactome/other text coverage remains incomplete. |
| phenotype text | 13,810 | 16,449 | 83.96% | Embedding count matches HPO textual-summary input. |
| protein sequence | 112,051 | 233,995 | 47.89% | Embedding count matches the promoted protein-sequence table; missingness is primarily sequence/source mapping coverage. |
| protein text | 162,163 | 233,995 | 69.30% | Embedding count matches the promoted UniProt textual-summary table. |
| tissue text | 11,942 | 16,061 | 74.35% | Embedding count matches UBERON textual-summary input. |
| transcript sequence | 187,268 | 507,365 | 36.91% | Embedding count matches the promoted transcript-sequence table; source sequence coverage is the limiting layer. |

### Node types without a current accepted source embedding

- `enhancer`: coordinates exist, but a reviewed reference-sequence extraction and embedding policy is still required. This is the dominant physical node family, so a dense vector for every enhancer would be expensive.
- `mutation`: no accepted local reference/alternate-context sequence feature and embedding release yet.
- `organism`: one current human node; metadata or a learned type-level representation is sufficient unless non-human expansion changes the use case.
- `paper` and `dataset`: graph-disconnected metadata by default, not message-passing nodes.

### Required coverage audit before more compute

For every `(node type, modality)` produce these disjoint counts:

```text
canonical target nodes
eligible source payloads
embedded rows
skipped source rows by reason
source-eligible but missing embedding rows
nodes without any eligible source payload
quarantined/out-of-training-scope nodes
```

Only `source-eligible but missing embedding rows` indicates a potentially unfinished embedding job. A node without a valid SMILES, sequence, description, or approved source mapping cannot be fixed by merely rerunning the model.

## Multimodal tensor policy

Keep each modality separate in the feature store:

```text
gene.x_text + gene.x_text_mask
gene.x_genomic + gene.x_genomic_mask
protein.x_text + protein.x_text_mask
protein.x_sequence + protein.x_sequence_mask
molecule.x_text + molecule.x_text_mask
molecule.x_smiles + molecule.x_smiles_mask
molecule.x_fingerprint + molecule.x_fingerprint_mask
```

The model applies modality-specific projectors to a common hidden dimension and then fuses them. Example:

```python
h_gene = fuse(
    text_projector(x_text, x_text_mask),
    genomic_projector(x_genomic, x_genomic_mask),
)
```

Do not concatenate raw vectors with incompatible dimensions or treat an all-zero vector as proof of missingness. The mask is authoritative.

## Edge-feature policy

### Safe mechanical inclusion

A numeric column physically attached to every row of one relation can become relation-specific `edge_attr`, for example:

- `disease_associated_gene.score`;
- `enhancer_regulates_gene.e2g_score`;
- `cell_line_expresses_gene.gene_effect` and `expression`;
- `cell_type_expresses_gene.tpm`;
- `tissue_expresses_gene.tpm`;
- numeric orthology identity/confidence fields.

The exact schema differs by relation. Therefore, there is no universal edge-feature matrix shared by all 43 relations. Each edge type gets its own schema and projector.

Categorical fields such as `action_type`, `expression_level`, `go_evidence`, `homology_type`, mechanism, or direction require a frozen vocabulary/encoding with unknown and missing states. Normalization and vocabularies must be fit on the training partition where the feature can vary by split.

### Why specialized `main/features/` tables are not generic `edge_attr`

1. **Different owners:** text and sequence tables describe nodes; clinical-trial links and evidence describe assertions; ReMap support describes contextual combinations.
2. **Different keys:** `node_id`, `edge_key`, `nct_id`, and multi-column support keys cannot be joined through one universal key.
3. **Different cardinalities:** some are one row per node, some zero-to-many per edge, and some many-to-many contextual records.
4. **Different shapes:** scalar numbers, categories, text, variable-length sequences, sparse fingerprints, and dense embeddings require different encoders.
5. **Different semantics:** evidence is provenance for an assertion, not automatically a biological property suitable for message passing.
6. **Different leakage risk:** trial outcomes or evidence used to establish a held-out edge can disclose the target label.

## Leakage registry

| Feature family | Main leakage risk | Default policy |
|---|---|---|
| Node names and ontology definitions | Usually low, but may include target relation wording or post-cutoff updates | pin source release and inspect payload construction |
| Gene/protein/pathway descriptions | Explicit statements of disease association, target status, or treatment mechanism | benchmark with text ablation; temporal/source cutoff where task requires it |
| Molecule descriptions | Indication or mechanism may directly name a held-out disease/target | mask indication/outcome text for drug-repurposing evaluation or keep evaluation-only |
| Clinical-trial text/status/outcomes | Directly discloses molecule–disease testing and possibly success/failure | provenance/evaluation-only by default for `molecule_treats_disease`; use only with explicit temporal protocol |
| Evidence aggregates | Edge existence and evidence count may disclose the label | derive only from training-visible evidence and remove held-out edge evidence |
| Graph-derived embeddings | May have been trained on validation/test topology | rebuild on training graph only or declare transductive setting explicitly |
| Text embeddings | Inherit every leakage property of their source text | record source hash, release, mask policy, and task allowlist |

## Neighbor-sampling runtime contract

The intended GNN training path is:

```text
canonical main/ snapshot
        ↓ reproducible build on an in-region worker
single current gs://jouvencekb/pyg/ artifact
        ↓ copy/cache once per job
user-selected local directory (default: REPO/data/pyg/)
        ↓ memory-map
JouvenceGraphStore + JouvenceFeatureStore
        ↓
LinkNeighborLoader for link prediction
        ↓
bounded heterogeneous multi-hop minibatches
```

`NeighborLoader` is for node-seed tasks. `LinkNeighborLoader` is the default for Jouvence/TxGNN link prediction. The message-passing graph must contain only training-visible edges; validation/test labels and their technical reverse edges must not be present in sampled adjacency.

## Implemented human-facing helper API

The API in `manage_db/pyg_artifact.py` makes the safe path short and explicit:

```python
build = resolve_pyg_build("gs://jouvencekb/pyg")

local_build = materialize_pyg_build(
    build,
    cache_dir="/path/chosen/by/the/user",  # default: REPO/data/pyg/
    verify=True,
)

graph_store, feature_store = open_pyg_stores(local_build, mmap=True)

node_loader = make_neighbor_loader(
    graph_store=graph_store,
    feature_store=feature_store,
    input_nodes=(node_type, seed_ids),
    num_neighbors=fanouts,
    batch_size=...,
)

link_loader = make_link_neighbor_loader(
    graph_store=graph_store,
    feature_store=feature_store,
    edge_label_index=(edge_type, train_edges),
    num_neighbors=fanouts,
    batch_size=...,
)

batch = next(iter(link_loader))
inspect_sampled_batch(batch, include_canonical_ids=True)
```

Required helpers:

- `build_pyg_index(...)`: heavy, VM-only, deterministic CSC/reverse-index builder;
- `resolve_pyg_build(...)`: validate build identity and canonical source generations;
- `materialize_pyg_build(...)`: copy once to a user-selected local directory, verify hashes, reuse safely;
- `open_pyg_stores(...)`: return memory-mapped PyG `GraphStore` and `FeatureStore` implementations;
- `make_neighbor_loader(...)`: node-task wrapper;
- `make_link_neighbor_loader(...)`: link-task wrapper with split/leakage gates;
- `inspect_sampled_batch(...)`: recover canonical node/edge IDs and report modalities/masks;
- `audit_embedding_coverage(...)`: produce the disjoint coverage counts above;
- `validate_no_split_leakage(...)`: prove held-out labels and reverse edges are absent from message-passing adjacency.

Notebook 07 imports and demonstrates the implemented resolve/materialize/open/loader helpers. Production materialization remains opt-in and worker-only until the first reviewed build is published.

## Review checklist for a registry change

- [ ] Owner and join key are explicit.
- [ ] Tensor representation, dimension, dtype, and mask are explicit.
- [ ] Source rows and embedded rows are counted separately.
- [ ] Model/source revision and immutable hash are recorded.
- [ ] Training/evaluation/provenance role is explicit.
- [ ] Prediction-task leakage is assessed.
- [ ] Split-time fitting and temporal cutoff are specified where relevant.
- [ ] Missingness is not hidden by zeros or fabricated source vectors.
- [ ] Notebook and loader behavior agree with this registry.
