"""Build bounded staged RNA/transcript-RBP CLIP audit artifacts.

This module is intentionally conservative: it stages only direct RBP/RNA CLIP
source rows and refuses to materialize KG edges unless both endpoints are
source-native KG endpoint IDs (transcript ENST/RefSeq mapped policy + protein
ENSP/UniProt mapped policy).  ENCORI/starBase RBPTarget currently exposes RBP
symbols and target gene IDs/names plus coordinates, so the default build writes
feature/rejected-candidate artifacts and zero canonical edges.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from manage_db import kg_evidence
from manage_db.kg_schema import EDGE_PARQUET_COLUMNS
from manage_db.kg_storage import open_kg_root, write_edges

RELATION = "transcript_interacts_protein"
DISPLAY_RELATION = "interacts with"
ENCORI_RBP_API = "https://rnasysu.com/encori/api/RBPTarget/"
ENCORI_MODULE_DOWNLOAD = "https://rnasysu.com/encori/moduleDownload.php"
ENCORI_REFERENCE_ZIP = "https://rnasysu.com/encori/api/ref/ENCORI_referenceData.zip"
ENCORI_PMID = "PMID:42185542"
ENCORI_LICENSE = "CC-BY-4.0 per ENCORI page HTML comment (https://creativecommons.org/licenses/by/4.0/)"

EDGE_COLUMNS = [name for name, _ in EDGE_PARQUET_COLUMNS]
EVIDENCE_COLUMNS = [name for name, _ in kg_evidence.EVIDENCE_PARQUET_COLUMNS]

CANDIDATE_COLUMNS = [
    "candidate_relation",
    "source",
    "source_dataset",
    "source_url",
    "source_record_id",
    "assembly",
    "rna_id",
    "rna_id_namespace",
    "rna_name",
    "rna_class",
    "rbp_name",
    "protein_id",
    "protein_id_namespace",
    "chromosome",
    "narrow_start",
    "narrow_end",
    "broad_start",
    "broad_end",
    "strand",
    "clip_method",
    "clip_region_type",
    "cluster_num",
    "total_clip_exp_num",
    "total_clip_site_num",
    "clip_exp_num",
    "cell_context",
    "pancancer_num",
    "pmid",
    "license",
    "evidence_strength",
    "raw_metadata_json",
]

REJECTED_COLUMNS = CANDIDATE_COLUMNS + ["reject_reason"]


@dataclass(frozen=True)
class SourceRequest:
    gene_type: str
    target: str
    rbp: str = "all"
    assembly: str = "hg38"
    clip_exp_num: int = 1
    pancancer_num: int = 0
    cell_type: str = "all"

    @property
    def candidate_relation(self) -> str:
        if self.gene_type == "lncRNA":
            return "lncrna_interacts_protein"
        return RELATION

    @property
    def source_dataset(self) -> str:
        return f"encori_rbptarget_{self.assembly}_{self.gene_type}"

    def api_url(self) -> str:
        query = urllib.parse.urlencode(
            {
                "assembly": self.assembly,
                "geneType": self.gene_type,
                "RBP": self.rbp,
                "clipExpNum": self.clip_exp_num,
                "pancancerNum": self.pancancer_num,
                "target": self.target,
                "cellType": self.cell_type,
            }
        )
        return f"{ENCORI_RBP_API}?{query}"

    def module_download_url(self) -> str:
        value = f"{self.assembly};{self.gene_type};{self.rbp};{self.clip_exp_num};{self.pancancer_num};{self.target}"
        query = urllib.parse.urlencode({"source": "rbpClipRNA", "type": "txt", "value": value})
        return f"{ENCORI_MODULE_DOWNLOAD}?{query}"


def _clean(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text == "NA" else text


def _fetch_text(url: str, *, timeout: int = 60) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 txgnn-source-audit"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # nosec - audited public source URL
        return resp.read().decode("utf-8", errors="replace")


def _parse_encori_tsv(text: str) -> pd.DataFrame:
    lines = [line for line in text.splitlines() if line and not line.startswith("#")]
    if not lines:
        return pd.DataFrame()
    return pd.read_csv(io.StringIO("\n".join(lines)), sep="\t", dtype=str, keep_default_na=False)


def fetch_encori_rows(request: SourceRequest, *, max_rows: int | None = None) -> pd.DataFrame:
    df = _parse_encori_tsv(_fetch_text(request.api_url()))
    if max_rows is not None:
        df = df.head(max_rows)
    df["_source_url"] = request.api_url()
    df["_assembly"] = request.assembly
    df["_source_dataset"] = request.source_dataset
    df["_candidate_relation"] = request.candidate_relation
    return df


def row_to_candidate(row: pd.Series, row_index: int) -> dict[str, object]:
    rbp = _clean(row.get("RBP"))
    cluster_id = _clean(row.get("clusterID"))
    gene_id = _clean(row.get("geneID"))
    gene_name = _clean(row.get("geneName"))
    gene_type = _clean(row.get("geneType"))
    source_dataset = _clean(row.get("_source_dataset"))
    source_record_id = cluster_id or f"{source_dataset}:{rbp}:{gene_id or gene_name}:{row_index}"
    source_url = _clean(row.get("_source_url"))
    raw = {k: _clean(v) for k, v in row.items() if not str(k).startswith("_")}
    clip_exp = _clean(row.get("clipExpNum"))
    total_clip_exp = _clean(row.get("totalClipExpNum"))
    total_clip_site = _clean(row.get("totalClipSiteNum"))
    strength = ""
    if clip_exp:
        strength = f"clipExpNum={clip_exp}"
    elif total_clip_exp:
        strength = f"totalClipExpNum={total_clip_exp}"
    return {
        "candidate_relation": _clean(row.get("_candidate_relation")),
        "source": "ENCORI/starBase",
        "source_dataset": source_dataset,
        "source_url": source_url,
        "source_record_id": source_record_id,
        "assembly": _clean(row.get("_assembly")),
        "rna_id": gene_id,
        "rna_id_namespace": "Ensembl gene (not transcript endpoint)",
        "rna_name": gene_name,
        "rna_class": gene_type,
        "rbp_name": rbp,
        "protein_id": "",
        "protein_id_namespace": "RBP gene symbol only; no approved gene-to-protein projection",
        "chromosome": _clean(row.get("chromosome")),
        "narrow_start": _clean(row.get("narrowStart")),
        "narrow_end": _clean(row.get("narrowEnd")),
        "broad_start": _clean(row.get("broadStart")),
        "broad_end": _clean(row.get("broadEnd")),
        "strand": _clean(row.get("strand")),
        "clip_method": "CLIP-seq aggregate; API parameter does not expose specific PAR-CLIP/eCLIP/HITS-CLIP method in RBPTarget response",
        "clip_region_type": "cluster/peak; clusterID present when using API endpoint",
        "cluster_num": _clean(row.get("clusterNum")),
        "total_clip_exp_num": total_clip_exp,
        "total_clip_site_num": total_clip_site,
        "clip_exp_num": clip_exp,
        "cell_context": _clean(row.get("cellline/tissue")),
        "pancancer_num": _clean(row.get("pancancerNum")),
        "pmid": ENCORI_PMID,
        "license": ENCORI_LICENSE,
        "evidence_strength": strength,
        "raw_metadata_json": json.dumps(raw, sort_keys=True, ensure_ascii=False),
    }


def classify_candidate(candidate: dict[str, object]) -> str:
    relation = _clean(candidate.get("candidate_relation"))
    rna_ns = _clean(candidate.get("rna_id_namespace"))
    protein_id = _clean(candidate.get("protein_id"))
    if relation == "lncrna_interacts_protein":
        return "relation_not_in_schema_and_lncRNA_node_type_missing"
    if "not transcript endpoint" in rna_ns:
        return "rna_endpoint_is_gene_or_coordinate_not_source_native_transcript"
    if not protein_id:
        return "rbp_endpoint_is_symbol_without_approved_protein_mapping"
    return ""


def empty_edges() -> pd.DataFrame:
    df = pd.DataFrame(columns=EDGE_COLUMNS)
    for col in EDGE_COLUMNS:
        df[col] = df[col].astype("object")
    return df


def empty_evidence() -> pd.DataFrame:
    df = pd.DataFrame(columns=EVIDENCE_COLUMNS)
    for col in EVIDENCE_COLUMNS:
        df[col] = df[col].astype("object")
    return df


def build_candidates(requests: Iterable[SourceRequest], *, max_rows_per_request: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    candidates: list[dict[str, object]] = []
    source_requests = []
    for req in requests:
        source_requests.append(
            {
                "api_url": req.api_url(),
                "module_download_url": req.module_download_url(),
                "gene_type": req.gene_type,
                "target": req.target,
                "clip_exp_num": req.clip_exp_num,
                "cell_type": req.cell_type,
            }
        )
        rows = fetch_encori_rows(req, max_rows=max_rows_per_request)
        for i, (_, row) in enumerate(rows.iterrows(), start=1):
            candidates.append(row_to_candidate(row, i))
    candidate_df = pd.DataFrame(candidates, columns=CANDIDATE_COLUMNS)
    rejected_rows = []
    for row in candidates:
        reason = classify_candidate(row)
        if reason:
            rejected = dict(row)
            rejected["reject_reason"] = reason
            rejected_rows.append(rejected)
    rejected_df = pd.DataFrame(rejected_rows, columns=REJECTED_COLUMNS)
    audit = {
        "sources": {
            "encori_starbase": {
                "rbp_api_endpoint": ENCORI_RBP_API,
                "rbp_api_example": SourceRequest(gene_type="mRNA", target="TP53", clip_exp_num=5, cell_type="HeLa").api_url(),
                "module_download_endpoint": ENCORI_MODULE_DOWNLOAD,
                "module_download_value_shape": "value={assembly};{geneType};{RBP};{clipExpNum};{pancancerNum};{target}",
                "reference_zip": ENCORI_REFERENCE_ZIP,
                "license": ENCORI_LICENSE,
                "citation": "Zhou et al., Nature Methods 2026, PMID:42185542",
                "observed_rbp_fields": list(candidate_df.columns) if not candidate_df.empty else [],
                "api_observed_columns": [
                    "RBP",
                    "geneID",
                    "geneName",
                    "geneType",
                    "clusterNum",
                    "totalClipExpNum",
                    "totalClipSiteNum",
                    "clusterID",
                    "chromosome",
                    "narrowStart",
                    "narrowEnd",
                    "broadStart",
                    "broadEnd",
                    "strand",
                    "clipExpNum",
                    "HepG2(shRNA)",
                    "K562(shRNA)",
                    "HepG2(CRISPR)",
                    "K562(CRISPR)",
                    "pancancerNum",
                    "cellline/tissue",
                ],
                "endpoint_namespace_decision": "API rows are direct RBP/RNA CLIP evidence but expose RNA as geneID/geneName and RBP as symbol; no ENST/ENSP/UniProt endpoint is source-native in tested output.",
            },
            "postar3": {
                "home": "http://111.198.139.65/ (postar.ncrnalab.org redirects here)",
                "rbp_module": "http://111.198.139.65/RBP.html",
                "rbs_module": "http://111.198.139.65/RBS.html",
                "ajax_endpoints": [
                    "http://111.198.139.65/script/ajax/RBP.php",
                    "http://111.198.139.65/script/ajax/RBS.php",
                    "http://111.198.139.65/script/ajax/RBS_show_CLIP_result.php",
                    "http://111.198.139.65/script/ajax/RBS_show_CLIP_result_predict.php",
                ],
                "species_gene_lists": [
                    "http://111.198.139.65/data/postar3/human_RBP_genelist.txt.js",
                    "http://111.198.139.65/data/postar3/human_RBS_genelist.txt.js",
                    "http://111.198.139.65/data/postar3/human_RBS_circRNA_genelist.txt.js",
                ],
                "bulk_download_note": "RBP.html says positions of all binding sites are downloadable through a link/request, but no direct bulk href was present in fetched HTML; survey form https://wj.qq.com/s2/10477617/4f84/v is exposed.",
                "license": "not found in fetched POSTAR3 HTML; citation/permission gate remains open",
                "citation": "POSTAR3 NAR 2022 PMID:34403477; POSTAR2 PMID:30239819; CLIPdb PMID:25652745",
                "clip_evidence_fields_seen_in_ui": ["species", "geneName/key", "RBP_name", "table", "CLIP method table families par/hits/iclip/eclip/pip plus predicted deepbind/fimo/tess", "coordinates in RBS result tables via AJAX"],
            },
        },
        "source_requests": source_requests,
    }
    return candidate_df, rejected_df, audit


def write_outputs(candidate_df: pd.DataFrame, rejected_df: pd.DataFrame, audit: dict, output_dir: Path, *, node_root: str = "") -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    for sub in ["edges", "evidence", "reports", "metadata"]:
        (output_dir / sub).mkdir(parents=True, exist_ok=True)

    root = open_kg_root(str(output_dir))
    write_edges(root, RELATION, empty_edges())
    kg_evidence.write_evidence(root, RELATION, empty_evidence())

    candidate_path = output_dir / "reports" / "candidate_clip_evidence.parquet"
    rejected_path = output_dir / "reports" / "rejected_rows.parquet"
    candidate_df.to_parquet(candidate_path, index=False)
    rejected_df.to_parquet(rejected_path, index=False)

    validation = {
        "ok": False,
        "staging_only": True,
        "canonical_promotion": False,
        "recommendation": "feature-only / no canonical promotion until endpoint policy is approved and node prerequisites are available",
        "candidate_rows": int(len(candidate_df)),
        "edge_rows": 0,
        "evidence_rows": 0,
        "rejected_rows": int(len(rejected_df)),
        "rejected_by_reason": rejected_df["reject_reason"].value_counts().to_dict() if not rejected_df.empty else {},
        "endpoint_anti_joins": {
            "node_root": node_root,
            "checked": False,
            "blocked_reason": "canonical node root is unavailable or insufficient; /mnt/gcs/jouvencekb/main was not mounted in this run, lncrna node type is absent from schema, and source rows lack source-native transcript/protein IDs",
        },
        "policy_checks": {
            "direct_clip_evidence_only": True,
            "gene_to_protein_projection_used": False,
            "coordinate_to_transcript_projection_used": False,
            "canonical_kg_writes": False,
        },
    }
    audit = dict(audit)
    audit["validation"] = validation
    audit["created_at"] = datetime.now(timezone.utc).isoformat()
    audit["relation"] = RELATION
    audit["candidate_relations_seen"] = sorted(candidate_df["candidate_relation"].dropna().unique().tolist()) if not candidate_df.empty else []

    audit_path = output_dir / "reports" / "source_audit.json"
    validation_path = output_dir / "reports" / "validation.json"
    manifest_path = output_dir / "MANIFEST.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    validation_path.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n")

    manifest = {
        "created_at": audit["created_at"],
        "relation": RELATION,
        "staging_only": True,
        "canonical_promotion": False,
        "outputs": {
            "edges": str(output_dir / "edges" / f"{RELATION}.parquet"),
            "evidence": str(output_dir / "evidence" / f"{RELATION}.parquet"),
            "candidate_clip_evidence": str(candidate_path),
            "rejected_rows": str(rejected_path),
            "source_audit": str(audit_path),
            "validation": str(validation_path),
        },
        "validation": validation,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def parse_request(text: str) -> SourceRequest:
    parts = dict(part.split("=", 1) for part in text.split(",") if "=" in part)
    return SourceRequest(
        gene_type=parts.get("geneType", parts.get("gene_type", "mRNA")),
        target=parts.get("target", "TP53"),
        rbp=parts.get("RBP", parts.get("rbp", "all")),
        assembly=parts.get("assembly", "hg38"),
        clip_exp_num=int(parts.get("clipExpNum", parts.get("clip_exp_num", "1"))),
        pancancer_num=int(parts.get("pancancerNum", parts.get("pancancer_num", "0"))),
        cell_type=parts.get("cellType", parts.get("cell_type", "all")),
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", action="append", default=[], help="Comma-separated ENCORI request, e.g. geneType=lncRNA,target=MALAT1,clipExpNum=1")
    parser.add_argument("--max-rows-per-request", type=int, default=50)
    parser.add_argument("--node-root", default="/mnt/gcs/jouvencekb/main")
    parser.add_argument("--output-dir", default="artifacts/staged/rbp-rna-clip-encori-pilot")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    requests = [parse_request(item) for item in args.request] or [
        SourceRequest(gene_type="mRNA", target="TP53", clip_exp_num=5, cell_type="HeLa"),
        SourceRequest(gene_type="lncRNA", target="MALAT1", clip_exp_num=1, cell_type="all"),
    ]
    candidate_df, rejected_df, audit = build_candidates(requests, max_rows_per_request=args.max_rows_per_request)
    manifest = write_outputs(candidate_df, rejected_df, audit, Path(args.output_dir), node_root=args.node_root)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
