# 01 — LaminDB

_Status snapshot: 2026-07-19 CEST._

Kanban board `txgnn` remains the live source of truth. Counter evidence below is explicitly dated; this mirror does not imply a fresh database read.

Heavy-job guardrail: full/bulk LaminDB syncs or registry scans must run on `txgnn-worker` or another explicitly approved in-region worker with source `gs://jouvencekb/main`. Do not run heavy LaminDB reads/writes from the Mac through `/Users/jkobject/mnt/gcs/...` / macOS GCS-FUSE.

## Canonical source denominator

Canonical Parquet inventory currently contains:

- **15/15 node files**;
- **55,523,691 physical node rows** total;
- **52,565,491 model/biomedical node rows** when graph-disconnected `paper` (2,958,199) and `dataset` (1) metadata nodes are excluded;
- **100,080,390 canonical edge rows**;
- **76,565,213 canonical evidence rows**;
- **230,874,162 rows** in the current Lamin ingestion denominator: 52,565,491 nodes + 101,743,458 edge target rows + 76,565,213 evidence target rows.

The edge ingestion denominator is larger than the older 100,080,390 snapshot because later reviewed/promoted relation material is included in the Lamin target contract. Do not silently mix these dated inventories.

## Live `jkobject/jouvencekb` ingestion

Latest durable accepted ledger, sealed in 2026-07-18 task evidence:

| Layer | Accepted | Denominator | Status |
| --- | ---: | ---: | --- |
| nodes | 3,771,054 | 52,565,491 | partial |
| edges | 3,971,264 | 101,743,458 | partial |
| evidence | 3,929,167 | 76,565,213 | partial |
| **total** | **11,671,485** | **230,874,162** | **partial** |

The latest sealed physical readback is also from 2026-07-18: 3,771,054 nodes + 4,141,291 edges + 4,099,167 evidence = **12,011,512 physical rows**. The +170,027 edge and +170,000 evidence difference is physical but uncredited. No newer mismatch-0 database readback is claimed here.

## Human ENSG-only denominator change in progress

`t_8b9cdabc` produced a `staged-only` / `review-required` human Gene candidate at commit `8714378`; its PR and independent review remain outstanding:

- current canonical Gene source: 267,830 IDs = 81,715 human ENSG + 27,610 human NCBI + 158,505 non-human Ensembl homologues;
- target canonical Gene identity: 81,715 human ENSG only;
- NCBI IDs are aliases/provenance, with authoritative endpoint remap or explicit quarantine required before removal;
- non-human homologue nodes and `gene_ortholog_gene` are excluded from the human canonical candidate;
- no canonical KG or LaminDB promotion is claimed.

`t_ce839966` and `t_075f5353` are superseded historical +158,505 Lamin Gene plans. They remain inert and must not run. Their 2026-07-18 readbacks remain useful as dated counter evidence only.

## Current boundary

Do not schedule a new bulk Lamin Gene wave against the old 267,830-row denominator. First complete independent review and any explicit promotion of the ENSG-only candidate, then produce a fresh source↔Lamin denominator and mismatch-0 readback. Keep accepted and physical counters separate until that review closes.

## What is and is not complete

- Canonical Parquet node inventory: **complete for the 15 active node types**.
- Canonical relation inventory: **40/67 declared relations physically canonical** in the latest schema snapshot; relation review/backlog is separate from Lamin ingestion.
- LaminDB artifact catalog/schema activation: implemented and reviewed in bounded form.
- Full row-level Lamin node/edge/evidence ingestion: **not complete** (11,671,485/230,874,162 accepted as of the dated 2026-07-18 evidence).
- Query helpers and full exact-ID coverage: not a completed global acceptance gate.

## Definition of done

1. Exact `jkobject/jouvencekb` identity and `lnschema_txgnn` activation verified.
2. The target denominator is rebased after the reviewed ENSG-only decision, then every included row is durably ingested or explicitly excluded.
3. Every credited wave has `rc=0`, hash-bound acknowledgements, selected-live edge/evidence equality and mismatch 0.
4. Exact-ID node/edge/evidence/feature query probes pass.
5. Independent review accepts the final ledger, query surface and evidence packet.
