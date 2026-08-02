#!/usr/bin/env bash
set -euo pipefail

LEASE_ID=${1:?lease id required}
GENERATION=${2:?generation required}
EXPECTED_COMMIT=${3:?expected repository commit required}
TASK_ROOT=/home/jkobject/t_03bf9e27
REPO="$TASK_ROOT/repo"
HEARTBEAT="$TASK_ROOT/payload-heartbeat.json"
BUILDER_OUT="$TASK_ROOT/v2-recovery-builder"
RECOVERED_SOURCE="$TASK_ROOT/v2-recovered-source"
CANDIDATE_DIR="$TASK_ROOT/v2-candidate"
BASE_DIR="$TASK_ROOT/v1-adopted-base"
RECOVERY_REPORT="$TASK_ROOT/v2-recovery-source-report.json"
GCS_ROOT=gs://jouvencekb/staging/gene-genomic-sequence-embeddings-20260722-t_03bf9e27-v2
SOURCE_ORIGIN=gs://jouvencekb/staging/gene-genomic-sequence-20260625-t_720528ea
BASE_ROOT=gs://jouvencekb/staging/gene-genomic-sequence-embeddings-20260721-t_03bf9e27
HEARTBEAT_SOURCE="$TASK_ROOT/source/features/gene_genomic_sequence.parquet"

write_heartbeat() {
  "$REPO/.venv/bin/python" "$REPO/scripts/write_gene_genomic_embedding_heartbeat.py" \
    --path "$HEARTBEAT" --lease-id "$LEASE_ID" --generation "$GENERATION" \
    --manifest "$BUILDER_OUT/manifest.json" --target "$GCS_ROOT" \
    --source "$HEARTBEAT_SOURCE" \
    --canonical-gene "$TASK_ROOT/preflight/gene.parquet" --payload-pid "$PAYLOAD_PID"
}

heartbeat_loop() {
  while kill -0 "$PAYLOAD_PID" 2>/dev/null; do
    if ! write_heartbeat; then
      kill -TERM "$PAYLOAD_PID" 2>/dev/null || true
      return 1
    fi
    sleep 30
  done
}

[[ "$(hostname)" == "txgnn-worker" ]]
[[ ! -e /Users/jkobject/mnt/gcs ]]
[[ "$(curl -fsS -H Metadata-Flavor:Google http://metadata.google.internal/computeMetadata/v1/instance/id)" == "4268456364292488510" ]]
export HERMES_KANBAN_TASK=t_03bf9e27
export PYTHONPATH="$REPO"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
cd "$REPO"

rm -rf "$BUILDER_OUT"
PAYLOAD_PID=$$
write_heartbeat
heartbeat_loop &
HEARTBEAT_PID=$!
trap 'kill "$HEARTBEAT_PID" 2>/dev/null || true; wait "$HEARTBEAT_PID" 2>/dev/null || true' EXIT

[[ "$(git rev-parse HEAD)" == "$EXPECTED_COMMIT" ]]
[[ "$(.venv/bin/python -c 'import transformers; print(transformers.__version__)')" == "4.55.4" ]]

rm -rf "$CANDIDATE_DIR" "$TASK_ROOT/readback" "$RECOVERED_SOURCE"
mkdir -p "$RECOVERED_SOURCE/features" "$RECOVERED_SOURCE/reports" "$BASE_DIR"
curl -fL --retry 3 -o "$TASK_ROOT/source/Homo_sapiens.GRCh38.114.gtf.gz" \
  https://ftp.ensembl.org/pub/release-114/gtf/homo_sapiens/Homo_sapiens.GRCh38.114.gtf.gz
curl -fL --retry 3 -o "$TASK_ROOT/source/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz" \
  https://ftp.ensembl.org/pub/release-114/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz
.venv/bin/python -m manage_db.recover_gene_genomic_overlength_windows \
  --original-sequence "$TASK_ROOT/source/features/gene_genomic_sequence.parquet" \
  --interval "$TASK_ROOT/source/features/gene_genomic_interval.parquet" \
  --fasta "$TASK_ROOT/source/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz" \
  --output "$RECOVERED_SOURCE/features/gene_genomic_sequence.parquet" \
  --report "$RECOVERY_REPORT"
cp "$TASK_ROOT/source/features/gene_genomic_interval.parquet" "$RECOVERED_SOURCE/features/"
cp "$TASK_ROOT/source/reports/gene_genomic_sequence_feature_report.json" "$RECOVERED_SOURCE/reports/"
gcloud storage cp --quiet \
  "$BASE_ROOT/embeddings/gene_genomic_sequence_nt.parquet" \
  "$BASE_DIR/gene_genomic_sequence_nt.parquet"
gcloud storage cp --quiet \
  "$BASE_ROOT/manifest.json" \
  "$BASE_DIR/manifest.json"
kill "$HEARTBEAT_PID" 2>/dev/null || true
wait "$HEARTBEAT_PID" 2>/dev/null || true
HEARTBEAT_SOURCE="$RECOVERED_SOURCE/features/gene_genomic_sequence.parquet"
write_heartbeat
heartbeat_loop &
HEARTBEAT_PID=$!
.venv/bin/python -m manage_db.build_gene_genomic_sequence_embeddings \
  --source-root "$RECOVERED_SOURCE" \
  --output-dir "$BUILDER_OUT" \
  --no-limit \
  --batch-size 16 \
  --part-size 480 \
  --max-nucleotides-per-window 1000 \
  --window-stride 1000 \
  --device cpu \
  --row-start 78164 \
  --clean

.venv/bin/python -m manage_db.finalize_gene_genomic_embedding_candidate \
  --task-root "$TASK_ROOT" \
  --builder-output "$BUILDER_OUT" \
  --source-root "$RECOVERED_SOURCE" \
  --canonical-gene "$TASK_ROOT/preflight/gene.parquet" \
  --canonical-gene-origin "gs://jouvencekb/main/nodes/gene.parquet" \
  --ensembl-gtf "$TASK_ROOT/source/Homo_sapiens.GRCh38.114.gtf.gz" \
  --adopted-base-embedding "$BASE_DIR/gene_genomic_sequence_nt.parquet" \
  --adopted-base-manifest "$BASE_DIR/manifest.json" \
  --adopted-base-origin "gs://jouvencekb/staging/gene-genomic-sequence-embeddings-20260721-t_03bf9e27/embeddings/gene_genomic_sequence_nt.parquet" \
  --recovery-report "$RECOVERY_REPORT" \
  --candidate-dir "$CANDIDATE_DIR" \
  --gcs-root "$GCS_ROOT" \
  --source-origin-root "$SOURCE_ORIGIN" \
  --repository-commit "$(git rev-parse HEAD)"
