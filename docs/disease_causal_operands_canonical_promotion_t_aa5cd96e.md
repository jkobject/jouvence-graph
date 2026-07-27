# `disease_associated_protein` causal operands canonical promotion — `t_aa5cd96e`

Date: 2026-07-22  
Status: **canonical promoted / review-required**

## Scope and code identity

This promotion integrated reviewed PR #39 exact head `042472c9bbe06153a9712e919af5a9b27372a9ce`. PR #39 was retargeted from its already-merged stacked base to `main`, re-read as `MERGEABLE` with the same immutable head, and squash-merged as main commit `343bb533f2d7d99b41a5c8149a18376bebecf593`.

Only the accepted `disease_associated_protein` edge and evidence candidates were promoted. No relation-name variant, LaminDB write, enhancer expansion, optional source materialization, or inferred-edge rewrite occurred.

## Authorization and preflight

The Kanban card authorized a SMALL local-to-GCS create-only promotion with no VM. The exact canonical targets were absent before writing, so no broad table was replaced and no backup object was required. The immutable preflight/rollback receipt is:

- `gs://jouvencekb/kg/v2/metadata/disease_causal_operands_promotion_t_aa5cd96e_preflight.json`
- generation `1784726652536984`

The preflight acquired a create-only global canonical-writer lock, re-read both exact targets as absent, and recorded generation-matched deletion as rollback for only task-created objects. The lock was released and fresh readback now returns 404.

Fresh memory admission passed with 17,179,869,184 total bytes, 51% free, and an 8,761,733,283-byte available estimate, above the required 8 GiB. Endpoint validation used bounded direct GCS downloads, never macOS GCS-FUSE:

- `nodes/protein.parquet`: generation `1781017349893351`, SHA-256 `b235efe7fd7e30bfd7ef221a56904dc52d2a41acc6ec0ab03db985a578d4c99f`;
- `nodes/disease.parquet`: generation `1781177660230499`, SHA-256 `db28a8b83fd2945c5bf0c8e968178516f79afb69f77486abc193c9857ebcadbd`.

UniProt's effective official license page states that copyrightable database parts are licensed under CC BY 4.0: <https://www.uniprot.org/help/license> (page last modified 2024-12-17). Source attribution and releases remain in evidence; the canonical receipts record the license verification.

## Canonical objects

| Object | Generation | Rows | Bytes | SHA-256 |
| --- | ---: | ---: | ---: | --- |
| `gs://jouvencekb/kg/v2/edges/disease_associated_protein.parquet` | `1784726656613727` | 3,243 | 59,222 | `98c1b444efa2e3a0fcaecabed948c9752b52100d8cc8b9ec012c478a3f453dc2` |
| `gs://jouvencekb/kg/v2/evidence/disease_associated_protein.parquet` | `1784726654659180` | 35,839 | 4,093,409 | `0e125397708f8f8678984eeeeb99b803669e3838357b9025a2d1af5a4c8b9d12` |

Both writes used `if-generation-match=0`. The evidence object was created first, then the edge object. Exact downloaded canonical SHA-256 values equal the independently accepted staged hashes.

The single marker was created last:

- `gs://jouvencekb/kg/v2/metadata/disease_causal_operands_canonical_release_t_aa5cd96e.json`
- generation `1784726666721784`
- SHA-256 `c15191aac3573bd6fe2800af0a76b39c83447112fbb7ea6da4c0f95c843a4112`

## Live semantic readback

Independent direct-GCS readback produced `artifacts/reports/t_aa5cd96e_independent_live_validation.json` with `ok=true` and:

- edge rows: 3,243;
- evidence rows: 35,839;
- duplicate edge keys: 0;
- reconstructed evidence-key mismatches: 0;
- evidence keys without edge: 0;
- edge keys without evidence: 0;
- protein endpoint anti-join: 0;
- disease endpoint anti-join: 0;
- mechanism status: `single=1`, `unknown=3,242`, `conflicting=0`;
- effect-direction status: `single=712`, `consensus=1,531`, `unknown=1,000`, `conflicting=0`;
- both operands known: 1;
- source rows: UniProtKB 15,006 and UniProtKB/humsavar 20,833;
- contract version: exactly `disease-causal-operands-v1`;
- staged/canonical hash mismatch: 0.

The promotion utility was replayed after publication and returned `replay_noop=true`, with the same marker generation and no create call. The independent live validator and replay each returned exit code 0.

## Preserved inferred release

The existing formal zero-row inferred release was not rewritten. Its completion marker remains:

- `gs://jouvencekb/kg/v2/edges_inferred/formal-relation-inference-v2/releases/post-operand-12fe3286f509-zero-rows/COMPLETED.json`
- generation `1784712720315973`
- SHA-256 `a19dc1ad0d31a81447a1f9ae615b0b315ce6d4aea19dbdb5b7237a2329b5a597`

These values exactly match its accepted promotion receipt.

## Commands and tests

- `uv run pytest -q tests/test_materialize_disease_causal_operands.py tests/test_validate_staged_disease_causal_operands.py` — 38 passed;
- `uv run ruff check manage_db/materialize_disease_causal_operands.py scripts/validate_staged_disease_causal_operands.py tests/test_materialize_disease_causal_operands.py tests/test_validate_staged_disease_causal_operands.py` — passed;
- `uv run python scripts/validate_staged_disease_causal_operands.py artifacts/staged/t_causal_disease_operands` — `ok=true`, `errors=[]`;
- task promotion script `py_compile` and Ruff — passed;
- independent live validation — exit 0;
- replay no-op validation — exit 0.

## Residual risk and review notes

Independent acceptance is still required before product credit. This materialization improves disease-direction coverage substantially but has only one edge with a known causal mechanism, and that edge is outside the current 701 molecule-target-protein-disease joins; fully signed joined paths therefore remain zero.

The first promotion process printed the complete successful marker payload and then returned nonzero during its generation-matched writer-lock cleanup wrapper. Fresh readback proved the lock had in fact been deleted, all canonical generations/hashes were correct, and subsequent independent validation and replay no-op both returned exit 0. The task-local wrapper was hardened to accept object absence as the lock-release postcondition; canonical data did not require repair or rollback.
