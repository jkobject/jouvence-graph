# PRISM Repurposing 20Q2 staged response candidate — `t_1cdcf211`

Status: **staged-only / review-required**. This work did not write the canonical KG or LaminDB and is independent of DepMap CRISPR essentiality.

## Source and scope

The producer uses only the Broad PRISM Repurposing 20Q2 **secondary** screen from Figshare article `20564034`:

- title: `PRISM Repurposing 20Q2 Dataset`;
- release: `PRISM Repurposing 20Q2 Secondary`;
- DOI: `10.6084/m9.figshare.20564034.v1`;
- license: `CC BY 4.0`;
- source-native measurements: four-parameter dose-response curve fields (`upper_limit`, `lower_limit`, `slope`, `r2`, `auc`, `ec50`, `ic50`) plus replicate-collapsed log2 fold change and derived viability (`2 ** logfold_change`);
- 24Q2 mono-dose data is excluded and is not mixed into this candidate.

The immutable source manifest records Figshare file IDs, URLs, byte sizes, supplied MD5 values, and locally computed SHA-256 values. The three quantitative inputs are:

| Input | Bytes | SHA-256 |
| --- | ---: | --- |
| secondary dose-response curve parameters | 290,170,269 | `2ac69a21f1d681fe7447689262b82ca6e3dc90bfef0bd96eb5479b96f424e43d` |
| secondary replicate-collapsed treatment info | 4,284,816 | `f0ec000acda7c1a01ebae2aa0a2b5dba2821abb674f1fec7a363cdc60d8b379e` |
| secondary replicate-collapsed logfold change | 114,666,971 | `5e5d87a272c3bb81498a8645d2901a66aaec6b65d322d15aed39a501da998cc9` |

Pooling metadata and the source README are retained beside the staged candidate. The exact schema audit found that the README documents a `convergence` field but the published curve-parameter CSV does not contain it. The producer records that discrepancy and does not invent a convergence value. Exposure time is also absent and remains null.

## Endpoint mapping

Cell lines map only by exact DepMap `ACH-*` identity to existing canonical `cell_line.id` values. Compound identity remains the source Broad formulation/batch ID in every feature/evidence row. Canonical molecule mapping uses RDKit-derived exact InChIKey equality against existing `molecule.inchikey`; names are evidence only and are prohibited as mapping keys.

The source Broad-ID crosswalk contains 1,552 identities:

| Mapping status | Broad IDs |
| --- | ---: |
| `mapped_unique_inchikey` | 1,054 |
| `unmatched_inchikey` | 447 |
| `conflicting_source_structures` | 39 |
| `missing_structure` | 12 |

All 498 non-unique/non-authoritative mappings are quarantined. No ambiguous or name-only compound mapping enters the feature or edge candidate.

## Staged artifact

Accepted corrected prefix:

`gs://jouvencekb/kg/staging/source-native-expansion/prism-20q2-secondary-20260719-t_1cdcf211-v2/`

The earlier unsuffixed prefix is **rejected**: independent readback found 2,481 duplicate feature `record_id` values because two native matrix row labels can normalize to the same ACH identifier. It remains historical staged output and must not be reviewed or promoted. Version 2 preserves `source_row_name` and includes it in deterministic feature identity.

Version 2 surfaces (329,554,530 staged bytes including reports and retained source metadata):

| Surface | Rows | Meaning |
| --- | ---: | --- |
| `features/cell_line_molecule_viability_response.parquet` | 3,910,010 | 435,256 mapped curve fits plus 3,474,754 source dose observations |
| `edges/cell_line_responds_to_molecule.parquet` | 31,349 | separately thresholded, deduplicated response candidate |
| `evidence/cell_line_responds_to_molecule.parquet` | 31,349 | one source curve/QC evidence row per candidate edge |
| `mapping/broad_id_to_molecule.parquet` | 1,552 | complete Broad-ID crosswalk |
| `mapping/quarantine.parquet` | 498 | rejected mapping identities and reasons |

Curve fits retain AUC/EC50/IC50/R2 and source curve parameters. Dose observations retain dose, screen, source native matrix row, Broad ID, log2 fold change, and viability. Dose observations do not invent AUC, EC50, IC50, curve R2, STR QC, or exposure time.

## Edge-candidate QC policy

The edge surface is intentionally separate from the complete feature surface. A curve can create `cell_line_responds_to_molecule` only when:

1. the cell line is an exact canonical ACH endpoint;
2. the Broad ID maps through one unique exact InChIKey;
3. MTS010 is preferred over an older screen for the same canonical cell-line/molecule pair when present;
4. source `passed_str_profiling` is true;
5. AUC is at most `0.70`;
6. curve R2 is at least `0.80`.

The response score is `1 - AUC`; evidence preserves AUC as effect size, EC50, IC50, R2, screen, assay, native IDs, mapping method, release, license, and the full compact source record. This is a conservative source-specific candidate policy, not canonical promotion authorization.

## Validation evidence

The build ran twice on `txgnn-worker`; all seven producer payloads were byte-identical across runs. The staged readback gate checks:

- exact input schema and source hashes;
- exact cell-line and molecule endpoint anti-joins;
- feature-record identity uniqueness;
- edge and evidence duplicate keys;
- unsupported edges and evidence-without-edge gaps;
- curve-feature parity;
- null curve metrics on dose observations (no invented AUC/EC50/IC50);
- 20Q2-only release/source values;
- staged GCS payload SHA-256 equality with the deterministic local build.

Producer validation reports 0 endpoint anti-joins, 0 duplicate edges, 0 duplicate evidence records, 0 unsupported edges, and 0 evidence-without-edge rows. Final version-2 readback hashes and the duplicate-feature count are recorded in the staged `reports/` files for independent review.

The post-upload readback report passed with 0 duplicate feature IDs, 0 endpoint anti-joins, 0 null logfold/derived-viability dose values, 0 invented dose-level AUC/EC50/IC50/STR values, and SHA-256 equality for every staged producer payload. Key deterministic payload hashes are:

| Payload | SHA-256 |
| --- | --- |
| feature Parquet | `3080db1e11de1b3b8af200af2d659f682f6e27b939c1ab9149b20d989d9e3159` |
| edge Parquet | `90938c92e9d9aed8407547119b657e3be616869841656a43309bad9f209e3e78` |
| evidence Parquet | `4401dfe2ed676359a56c4f3a2d464a2423a59b5ea1ea803805e9549e53e97763` |
| crosswalk Parquet | `28348e074cd557059f3f699a63ce2b8ea6f62c7a6839cd02c1258f9064108e2e` |
| quarantine Parquet | `5db9f85359486bc6e1042060f63b439d67831837c3fd1fb05d137d61f6e9637a` |

## Residual risks and review questions

- Exact InChIKey mapping is authoritative but narrower than a future reviewed Repurposing Hub alias/identifier crosswalk; 447 source structures remain unmatched.
- Structure collisions are quarantined rather than resolved by name or heuristic.
- The AUC/R2 thresholds are a conservative candidate policy and need scientific review before any promotion.
- The source README/schema discrepancy prevents convergence from being used as a QC gate.
- MTS010 preference is applied at canonical cell-line/molecule-pair level; reviewers should confirm that conservative suppression of older screens is desired.
- The complete feature table is a dense assay surface and must not be treated as graph adjacency without the explicit threshold policy.
- No canonical write, LaminDB write, relation composition, CRISPR dependency, or 24Q2 ingestion occurred.
