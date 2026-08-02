# Jouvence-Graph: a source-aware biomedical knowledge graph for drug repurposing

This repository is **Jouvence-Graph**, a source-aware biomedical knowledge graph and drug-repurposing project built on the upstream TxGNN method and library. Jouvence-Graph expands the data, provenance, feature, export, catalog, and reproducible scientific-notebook layers around TxGNN's zero-shot modeling foundation.

Compatibility boundary: Python imports remain `txgnn`, and public model classes such as `TxGNN`, `TxData`, and `TxEval` retain their upstream names. The upstream implementation and citation remain at [`mims-harvard/TxGNN`](https://github.com/mims-harvard/TxGNN); the active Jouvence repository is [`jkobject/jouvence-graph`](https://github.com/jkobject/jouvence-graph).

**Start here:** [project website](https://www.jkobject.com/jouvence-graph/) · [why this architecture](https://www.jkobject.com/jouvence-graph/why-not-open-targets.html) · [data quickstart](docs/getting-started-data.md) · [installation](#installation) · [API usage](#core-api-interface) · [paper reproduction](reproduce/README.md) · [documentation](docs/README.md) · [current work](TODO.md) · [agent instructions](AGENTS.md)

## Jouvence KG

Jouvence builds a larger, source-aware biomedical knowledge graph around the TxGNN model with:

- canonical Parquet nodes, edges, evidence, features, and embeddings under `gs://jouvencekb/main`;
- source-native relation semantics and explicit provenance rather than projected or placeholder edges;
- LaminDB cataloging through the intended instance `jkobject/jouvencekb`;
- memory-safe PyG/GNN export and learned/foundation feature pipelines;
- staged build → independent review → canonical promotion gates.

Large KG scans, LaminDB bulk syncs, ReMap processing, and production exports run only on the in-region `txgnn-worker`; that VM name, the Kanban board `txgnn`, and existing local paths are retained technical identifiers. The macOS GCS-FUSE mount is for small bounded inspection, not production work. Kanban is the live execution source of truth; `TODO.md` is only a compact mirror.

Documentation and contribution routes:

- [`docs/getting-started-data.md`](docs/getting-started-data.md) — clone-to-first-query guide for agents and external collaborators, including current access blocks;
- [`docs/README.md`](docs/README.md) — documentation index: architecture, source/evidence policy, runbooks, and durable knowledge;
- [`docs/guides/kg-architecture-and-evidence.md`](docs/guides/kg-architecture-and-evidence.md) — topology, evidence, metadata, features, and proof;
- [`docs/guides/lamindb-porting-operations.md`](docs/guides/lamindb-porting-operations.md) — LaminDB migration, progress semantics, VM/GCS/storage rationale, cost controls, and recovery gates;
- [`docs/pyg_export_runbook.md`](docs/pyg_export_runbook.md) — PyG/GNN export operations;
- [`TODO.md`](TODO.md) — dated current-work mirror (Kanban is live truth);
- [`AGENTS.md`](AGENTS.md) — mandatory boot and safety rules for automated contributors;
- [`docs/coding_standards.md`](docs/coding_standards.md) — contribution conventions;
- [`docs/txgnn_worker_disk_migration_t_3cf62bd8.md`](docs/txgnn_worker_disk_migration_t_3cf62bd8.md) — evidence for the 500 GB → 200 GB worker-disk migration.

### Upstream TxGNN model

This repository retains the upstream TxGNN implementation, a model for identifying therapeutic opportunities for diseases with limited treatment options and minimal molecular understanding that leverages recent advances in geometric deep learning and human-centered design.

TxGNN is a graph neural network pre-trained on a comprehensive knowledge graph of 17,080 clinically-recognized diseases and 7,957 therapeutic candidates. The model can process various therapeutic tasks, such as indication and contraindication prediction, in a unified formulation. Once trained, we show that TxGNN can perform zero-shot inference on new diseases without additional parameters or fine-tuning on ground truth labels.

### MedRxiv preprint is at [https://www.medrxiv.org/content/10.1101/2023.03.19.23287458v2](https://www.medrxiv.org/content/10.1101/2023.03.19.23287458v2)

### TxGNN Explorer of model predictions and explanations is at [http://txgnn.org](http://txgnn.org/)

![TxGNN](fig/txgnn_fig1.png)

## Jouvence scientific notebooks

The numbered suite in [`notebooks/`](notebooks/) contains only user-facing
exploration and usage notebooks. It introduces the
Jouvence node/assertion/evidence/feature model and then demonstrates bounded
entity exploration, provenance-aware biological questions, exact-instance
LaminDB queries, a sampled PyG `HeteroData`, embedding retrieval, neighborhood
analysis, and a deterministic link-prediction smoke. Each notebook states what
its output means biologically and what it does not prove.

Database construction, ingestion, schema-audit, and paper-reproduction
notebooks live separately in [`reproduce/`](reproduce/), also in a numbered
sequence. They are not part of the user quickstart.

Run notebooks 01–06 in fixture mode without reading the live KG. This command
also checks notebook 07 statically, but reports it as `skipped-live-gcs` instead
of executing it:

```bash
uv sync --group dev --group notebooks --group gnn
uv run python scripts/build_public_notebooks.py
uv run python scripts/check_public_notebooks.py --execute
```

Fixture mode is the default for notebooks 01–06 so they execute in a clean
environment. Their bounded live mode is explicitly enabled with
`JOUVENCE_DATA_MODE=live`.

Notebook 07, the data inventory explorer, is a separate live-only entry point:
it does not use `JOUVENCE_DATA_MODE`. Its execution performs real, bounded,
read-only requester-pays requests against fixed canonical and staging GCS roots.
Authenticate with Google application-default credentials and supply your own
billing project before running it:

```bash
gcloud auth application-default login
export JOUVENCE_BILLING_PROJECT='<consumer-billing-project>'
uv run python scripts/check_data_explorer_notebook.py --execute
```

The repository does not embed or require a project-specific default billing
project. The bucket is not anonymous: a new collaborator currently needs the
maintainer to grant their Google identity read-only object access first. See the
[`data quickstart`](docs/getting-started-data.md) for the current PASS/BLOCK
boundary and exact prerequisites. For live mode in notebooks 01–06, use:

```bash
export JOUVENCE_DATA_MODE=live
export JOUVENCE_BILLING_PROJECT='<consumer-billing-project>'
```

The helpers in `manage_db.public_notebooks` cap interactive reads at 10,000
rows, seek by Parquet row group, and never default to macOS GCS-FUSE or a full-KG
materialization.  Live LaminDB access is opt-in with
`JOUVENCE_LAMIN_LIVE=1` and refuses any instance other than
`jkobject/jouvencekb`; the mirror is currently incomplete, so canonical Parquet
remains the source of truth.  Public embedding URIs are not assumed while the
immutable release/license contract remains pending; set
`JOUVENCE_EMBEDDING_URI` only to an accepted published shard.

## Jouvence-Graph local viewer

The entity viewer supports the deterministic fixture and immutable,
manifest-verified local or requester-pays query bundles. It binds only to
`127.0.0.1`, keeps ADC in the host process, and exposes no arbitrary file,
bucket, or SQL surface. Run the no-cloud smoke from a clean clone with:

```bash
uv sync --group dev
uv run jouvence-viewer --fixture-smoke
```

The public preview at
[`docs/viewer.html`](docs/viewer.html) first loads a reviewed relative static
bundle (`manifest.json`, search shard, and per-entity shards) over ordinary
HTTPS. Opening the HTML directly from disk uses an embedded copy generated from
that same fixture. Both zero-backend paths have the same dossier/export
semantics as the fixture API and are explicitly labeled as non-live data.

For a complete compatible bundle, follow the bilingual
[local full-data viewer installation guide](docs/viewer-install.html). The
viewer refuses raw canonical `main/` roots rather than scanning 100M+ row tables
on a laptop.

The committed static assets are reproducible from the backend fixture:

```bash
uv run jouvence-viewer-bundle
```

### Installation

```bash
git clone https://github.com/jkobject/jouvence-graph.git
cd jouvence-graph
uv sync --group dev
```

The import namespace remains `txgnn` for compatibility. The historical `pip install TxGNN` command installs the upstream distribution rather than this active Jouvence workspace.

Note that if you want to use disease-area split, you should also install PyG following [this instruction](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) since some archived data processing code uses PyG utility functions.

### Core API Interface
Using the API, you can (1) reproduce the results in our paper and (2) train TxGNN on your own drug repurposing dataset using a few lines of code, and also generate graph explanations. 

```python
from txgnn import TxData, TxGNN, TxEval

# Download/load knowledge graph dataset
TxData = TxData(data_folder_path = './data')
TxData.prepare_split(split = 'complex_disease', seed = 42)
TxGNN = TxGNN(data = TxData, 
              weight_bias_track = False,
              proj_name = 'TxGNN', # wandb project name
              exp_name = 'TxGNN', # wandb experiment name
              device = 'cuda:0' # define your cuda device
              )

# Initialize a new model
TxGNN.model_initialize(n_hid = 100, # number of hidden dimensions
                      n_inp = 100, # number of input dimensions
                      n_out = 100, # number of output dimensions
                      proto = True, # whether to use metric learning module
                      proto_num = 3, # number of similar diseases to retrieve for augmentation
                      attention = False, # use attention layer (if use graph XAI, we turn this to false)
                      sim_measure = 'all_nodes_profile', # disease signature, choose from ['all_nodes_profile', 'protein_profile', 'protein_random_walk']
                      agg_measure = 'rarity', # how to aggregate sim disease emb with target disease emb, choose from ['rarity', 'avg']
                      num_walks = 200, # for protein_random_walk sim_measure, define number of sampled walks
                      path_length = 2 # for protein_random_walk sim_measure, define path length
                      )

```

Instead of initializing a new model, you can also load a saved model:

```python
TxGNN.load_pretrained('./model_ckpt')
```

We provide an example pre-trained model weight at [here](https://drive.google.com/file/d/1fxTFkjo2jvmz9k6vesDbCeucQjGRojLj/view).

To do pre-training using link prediction for all edge types, you can type:

```python
TxGNN.pretrain(n_epoch = 2, 
               learning_rate = 1e-3,
               batch_size = 1024, 
               train_print_per_n = 20)
```

Lastly, to do finetuning on drug-disease relation with metric learning, you can type:

```python
TxGNN.finetune(n_epoch = 500, 
               learning_rate = 5e-4,
               train_print_per_n = 5,
               valid_per_n = 20,
               save_name = finetune_result_path)
```

To save the trained model, you can type:

```python
TxGNN.save_model('./model_ckpt')
```

To evaluate the model on the entire test set using disease-centric evaluation, you can type:

```python
from txgnn import TxEval
TxEval = TxEval(model = TxGNN)
result = TxEval.eval_disease_centric(disease_idxs = 'test_set', 
                                     show_plot = False, 
                                     verbose = True, 
                                     save_result = True,
                                     return_raw = False,
                                     save_name = 'SAVE_PATH')

```

If you want to look at specific disease, you can also do:

```python
result = TxEval.eval_disease_centric(disease_idxs = [9907.0, 12787.0], 
                                     relation = 'indication', 
                                     save_result = False)
```


After training a satisfying link prediction model, we can also train graph XAI model by:

```python
TxGNN.train_graphmask(relation = 'indication',
                      learning_rate = 3e-4,
                      allowance = 0.005,
                      epochs_per_layer = 3,
                      penalty_scaling = 1,
                      valid_per_n = 20)
```

You can retrieve and save the graph XAI gates (whether or not an edge is important) into a pkl file located as `SAVED_PATH/'graphmask_output_RELATION.pkl'`:

```python
gates = TxGNN.retrieve_save_gates('SAVED_PATH')
```

Of course, you can save and load graphmask model as well via:

```python
TxGNN.save_graphmask_model('./graphmask_model_ckpt')
TxGNN.load_pretrained_graphmask('./graphmask_model_ckpt')

```

### Splits

There are numerous splits prepared in TxGNN. You can switch among them in the `TxData.prepare_split(split = 'XXX', seed = 42)` function.

- `complex_disease` is the systematic split in the paper, where we first sample a set of diseases and then move all of their treatments to test set such that these diseases have zero treatments in training.
- Disease area split first obtains a set of diseases in a disease area using disease ontology and move all of their treatments to the test set and then further removes a fraction of local neighborhood around these diseases to simulate the lack of molecular mechanism characterization of these diseases. There are nine disease areas: `cell_proliferation`, `mental_health`, `cardiovascular`, `anemia`, `adrenal_gland`, `autoimmune`, `metabolic_disorder`, `diabetes`, `neurodigenerative`
- `random` is namely random splits which it randomly shuffles across drug-disease pairs. In the end, most of diseases have seen some treatments in the training set.

During deployment, when evaluate a specific disease, you may want to just mask this disease and use all of the other diseases. In this case, you can use `TxData.prepare_split(split = 'disease_eval', disease_eval_idx = 'XX')` where `disease_eval_idx` is the index of the disease of interest. 

Another setting is to train the entire network without any disease masking. You can do that via `split = 'full_graph'`. This will automatically use 95% of data for training and 5% for validation set calculation to do early stopping. No test set is used. 


### Cite Us

[MedRxiv preprint](https://www.medrxiv.org/content/10.1101/2023.03.19.23287458)

```
@article{huang2023zeroshot,
  title={Zero-shot Prediction of Therapeutic Use with Geometric Deep Learning and Clinician Centered Design},
  author={Huang, Kexin and Chandak, Payal and Wang, Qianwen and Havaldar, Shreyas and Vaid, Akhil and Leskovec, Jure and Nadkarni, Girish and Glicksberg, Benjamin and Gehlenborg, Nils and Zitnik, Marinka},
  journal = {medRxiv},
  doi = {10.1101/2023.03.19.23287458},
  volume={},
  number={},
  pages={},
  year={2023},
  publisher={}
}
```
