#!/bin/bash
# =============================================================================
# SPOD Theory Analysis Launcher
# =============================================================================

set -e

SUBSET="${1:-perception}"
NUM_SAMPLES="${2:-30}"
PILLAR="${3:-FDA,TDS}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

if [[ "$SUBSET" == "test" ]]; then
    SUBSET="perception"
    NUM_SAMPLES=3
fi

# Default: Qwen2.5-VL-7B
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-VL-7B-Instruct}"
VIDEO_ROOT="${VIDEO_ROOT:-/root/autodl-tmp/DATA/VideoMMMU}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/theory}"

echo "============================================================"
echo "  SPOD Theory Analysis"
echo "  Model:  ${MODEL_PATH}"
echo "  Subset: ${SUBSET} | Samples: ${NUM_SAMPLES}"
echo "  Pillars: ${PILLAR}"
echo "  FDA:    L8-L16 | Deep: L22"
echo "============================================================"

python -m src.experiments.theory.analyze \
    --subset "${SUBSET}" \
    --num-samples "${NUM_SAMPLES}" \
    --pillar "${PILLAR}" \
    --output-dir "${OUTPUT_DIR}" \
    --device cuda \
    --model-path "${MODEL_PATH}" \
    --video-root "${VIDEO_ROOT}" \
    --fda-layer-start 8 \
    --fda-layer-end 16 \
    --fda-alpha 0.5 \
    --deep-layer 22 \
    --temporal-dims 4 \
    --prior-dims 4
