#!/usr/bin/env bash
set -euo pipefail
REPOSITORY=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPOSITORY"
PYTHON=${PYTHON:-python}
NUM_GPUS=${NUM_GPUS:-1}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/vidhalluc}
CONFIG=${CONFIG:-configs/spod.json}
if [[ ! "$NUM_GPUS" =~ ^[1-9][0-9]*$ ]]; then
  echo 'NUM_GPUS must be a positive integer.' >&2
  exit 2
fi
mkdir -p "$OUTPUT_DIR"
EXTRA=()
if [[ "${SMOKE:-0}" == 1 ]]; then EXTRA+=(--smoke); fi
PIDS=()
for ((i=0;i<NUM_GPUS;i++)); do
  CUDA_VISIBLE_DEVICES=$i "$PYTHON" -m vidhalluc infer --config "$CONFIG" --output-dir "$OUTPUT_DIR" --prompts-per-batch 1 --shard "$i" --shards "$NUM_GPUS" "${EXTRA[@]}" > "$OUTPUT_DIR/shard_${i}.log" 2>&1 &
  PIDS+=("$!")
done
RC=0
for pid in "${PIDS[@]}"; do wait "$pid" || RC=1; done
exit "$RC"
