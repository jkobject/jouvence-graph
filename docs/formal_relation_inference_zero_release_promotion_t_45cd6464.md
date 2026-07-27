# Formal inference v2 zero-row canonical promotion — `t_45cd6464`

Status: **canonical promoted / review-required**

Promotion time: 2026-07-22 11:32 CEST. This release promotes the independently accepted producer `t_50a6f3ce` at exact revision `12fe3286f5091bd1a69a8287649e02e169737402`. The dedicated branch `ops/t_aeaea066-inference-promotion` was clean at `bdb82d7db5aa3779432c3b04a7cee59278ff4a86`, then fast-forwarded to the accepted revision before the focused promotion implementation. The accepted producer revision is based on reviewed main `1615f6a55f608974758396b4f2cf3cf73c2b331c`.

## Observable release

Immutable release identity:

`post-operand-12fe3286f509-zero-rows`

Versioned roots:

- `gs://jouvencekb/kg/v2/edges_inferred/formal-relation-inference-v2/releases/post-operand-12fe3286f509-zero-rows/`
- `gs://jouvencekb/kg/v2/evidence_inferred/formal-relation-inference-v2/releases/post-operand-12fe3286f509-zero-rows/`

Single completion marker:

`gs://jouvencekb/kg/v2/edges_inferred/formal-relation-inference-v2/releases/post-operand-12fe3286f509-zero-rows/COMPLETED.json`

The release contains 11 immutable JSON objects, zero inferred edge rows, zero inferred evidence rows, and zero Parquets. It intentionally contains no placeholder row artifact. All 24 accepted templates and the full accepted zero-output explanation remain in the version-local manifests. The signed-protein rules preserve the formal product `action × disease_mechanism × disease_direction`: 701 joined paths, 377 with known action sign, 596 with known disease direction, zero with known disease mechanism, zero fully signed paths, and therefore zero outputs. Missing, unknown, or conflicting operands continue to abstain.

## Exact canonical inventory

Every upload used `if_generation_match=0` and immutable SHA-256 metadata. Data/manifests were published before the two receipts, and the sole completion marker was published last.

| Canonical object | Generation | Bytes | SHA-256 |
| --- | ---: | ---: | --- |
| `gs://jouvencekb/kg/v2/edges_inferred/formal-relation-inference-v2/releases/post-operand-12fe3286f509-zero-rows/manifest/input_manifest.json` | `1784712716836808` | 5,066 | `c71eddb953d069a914ecf1ded11844d39e8237fab718dcafaafd4796cc4cdcde` |
| `gs://jouvencekb/kg/v2/edges_inferred/formal-relation-inference-v2/releases/post-operand-12fe3286f509-zero-rows/manifest/pilot_report.json` | `1784712717008677` | 363,350 | `9392e83bc8c94143031d73a3bb78acb6211129643936f9e707ac70e44c0405b2` |
| `gs://jouvencekb/kg/v2/edges_inferred/formal-relation-inference-v2/releases/post-operand-12fe3286f509-zero-rows/manifest/release_manifest.json` | `1784712717196586` | 1,889 | `b446b4fbcfc2b04cd6b51804880a376d677e0beb39e88e549c2b2aa54c1ca0e5` |
| `gs://jouvencekb/kg/v2/edges_inferred/formal-relation-inference-v2/releases/post-operand-12fe3286f509-zero-rows/manifest/template_registry_v2.json` | `1784712717334650` | 12,168 | `e23e34dfe51e3a568b2ee9e928ed6ddce4c8395bcac540c2b03dfcf03bb45e1a` |
| `gs://jouvencekb/kg/v2/evidence_inferred/formal-relation-inference-v2/releases/post-operand-12fe3286f509-zero-rows/manifest/input_manifest.json` | `1784712717488155` | 5,066 | `c71eddb953d069a914ecf1ded11844d39e8237fab718dcafaafd4796cc4cdcde` |
| `gs://jouvencekb/kg/v2/evidence_inferred/formal-relation-inference-v2/releases/post-operand-12fe3286f509-zero-rows/manifest/pilot_report.json` | `1784712717651567` | 363,350 | `9392e83bc8c94143031d73a3bb78acb6211129643936f9e707ac70e44c0405b2` |
| `gs://jouvencekb/kg/v2/evidence_inferred/formal-relation-inference-v2/releases/post-operand-12fe3286f509-zero-rows/manifest/release_manifest.json` | `1784712717804877` | 1,889 | `b446b4fbcfc2b04cd6b51804880a376d677e0beb39e88e549c2b2aa54c1ca0e5` |
| `gs://jouvencekb/kg/v2/evidence_inferred/formal-relation-inference-v2/releases/post-operand-12fe3286f509-zero-rows/manifest/template_registry_v2.json` | `1784712717917104` | 12,168 | `e23e34dfe51e3a568b2ee9e928ed6ddce4c8395bcac540c2b03dfcf03bb45e1a` |
| `gs://jouvencekb/kg/v2/edges_inferred/formal-relation-inference-v2/releases/post-operand-12fe3286f509-zero-rows/manifest/promotion_receipt.json` | `1784712719639727` | 2,832 | `d2f12c85cd25d8ccfe4dc1be62d19c8ae0810f983e6c90bf0d3b4d85224331f7` |
| `gs://jouvencekb/kg/v2/evidence_inferred/formal-relation-inference-v2/releases/post-operand-12fe3286f509-zero-rows/manifest/promotion_receipt.json` | `1784712719810426` | 2,832 | `d2f12c85cd25d8ccfe4dc1be62d19c8ae0810f983e6c90bf0d3b4d85224331f7` |
| `gs://jouvencekb/kg/v2/edges_inferred/formal-relation-inference-v2/releases/post-operand-12fe3286f509-zero-rows/COMPLETED.json` | `1784712720315973` | 3,373 | `a19dc1ad0d31a81447a1f9ae615b0b315ce6d4aea19dbdb5b7237a2329b5a597` |

Independent bounded Google Cloud Storage readback downloaded all 11 objects, recomputed every SHA-256, checked byte sizes and immutable metadata, and proved the marker creation time (`2026-07-22T09:32:00.327000+00:00`) is later than every manifest/receipt. The expected inventory is exact: no extra, missing, `.tmp`, staging, or Parquet object exists below either release root.

## Source equality and preserved contract

The three accepted producer files were copied byte-for-byte into each inferred namespace:

| Source file | Producer SHA-256 | Edge copy | Evidence copy |
| --- | --- | --- | --- |
| `manifest/input_manifest.json` | `c71eddb953d069a914ecf1ded11844d39e8237fab718dcafaafd4796cc4cdcde` | equal | equal |
| `manifest/pilot_report.json` | `9392e83bc8c94143031d73a3bb78acb6211129643936f9e707ac70e44c0405b2` | equal | equal |
| `manifest/template_registry_v2.json` | `e23e34dfe51e3a568b2ee9e928ed6ddce4c8395bcac540c2b03dfcf03bb45e1a` | equal | equal |

The input manifest preserves exact operand Parquet hashes, source releases, columns, counts, and the semantic manifest digest `3c053cb8cfdb7af16f53d879031acb337d215428aa6df1696b1d328fddc591e2`. The pilot report preserves all premise/evidence IDs, typed rejected-path samples, source records/releases, conflict accounting, per-template epistemic class and counts, canonical/staged anti-join accounting, rejected motifs, and the fail-closed explanation. The registry preserves all 24 reviewed templates. The version-local release manifest explicitly records the empty expected row-artifact inventory and zero row/Parquet counts.

## Safety and replay probes

- First promotion: `outcome=created`, 11 canonical objects.
- Identical replay: `outcome=verified-no-op`; generations, hashes, sizes, and inventory remained unchanged and no create call was made.
- Same-identity conflicting revision probe: failed closed on the divergent `release_manifest.json` before mutation; inventory equality before/after was true.
- Observed namespaces: exact release-identity probes under `kg/v2/edges/` and `kg/v2/evidence/` returned no objects.
- Latest pointers: exact `latest*` probes under both inferred families returned no objects.
- Residue: no temporary/staging object exists under either release prefix; the implementation creates no local or cloud temporary object.
- Infrastructure: no VM, GCS-FUSE, broad canonical relation read, observed-edge write, delete, overwrite, latest-pointer mutation, or LaminDB operation occurred.

## Implementation and validation

Changed implementation files:

- `scripts/promote_inferred_zero_release.py`
- `tests/test_promote_inferred_zero_release.py`
- this report

The utility validates the exact accepted source inventory/hashes, enforces the zero-row/no-Parquet contract, constrains writes to the two inferred namespaces, uses create-only generation preconditions, performs exact canonical readback, rechecks exact inventory immediately before publishing one marker last, verifies inventory again after publication and during replay, treats an identical replay as a no-op, and rejects partial, concurrently changed, or byte-conflicting same-identity state. The marker explicitly enumerates the only release members. Focused probes prove an unexpected object before marker publication prevents the marker from being created and a sibling release whose identity shares the same textual prefix is excluded by the trailing-slash listing boundary.

Validation commands:

- `uv run --no-sync pytest -q tests/test_promote_inferred_zero_release.py tests/test_relation_composition_allowlist.py` — `60 passed` (`6` promotion tests plus `54` formal inference tests)
- `uvx ruff check scripts/promote_inferred_zero_release.py tests/test_promote_inferred_zero_release.py` — PASS
- `uv run --no-sync python -m compileall -q scripts/promote_inferred_zero_release.py tests/test_promote_inferred_zero_release.py` — PASS
- bounded generated-file, diff, and secret checks — recorded in the final Kanban handoff

## Residual risk

The release is an honest negative/abstention result, not proof that no molecule–disease relationship exists. Disease mechanism remains unknown on every currently joined protein path. Any future nonzero release requires new source-backed operands, a new immutable identity, deterministic rerun, and independent review; weakening the formal gate or mutating this release is not authorized.
