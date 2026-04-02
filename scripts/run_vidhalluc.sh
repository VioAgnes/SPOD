#!/bin/bash
# =============================================================================
# SPOD VidHalluc Evaluation Launcher
# =============================================================================

set -e

MODE="${1:-spod}"
SUBSET="${2:-all}"
shift 2 || true
EXTRA_FLAGS="$@"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# Default: Qwen2.5-VL-7B
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-VL-7B-Instruct}"
VIDEO_ROOT="${VIDEO_ROOT:-/root/autodl-tmp/DATA/VidHalluc/videos}"
QUESTION_ROOT="${QUESTION_ROOT:-/root/autodl-tmp/DATA/VidHalluc}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/vidhalluc}"

echo "============================================================"
echo "  SPOD VidHalluc Evaluation"
echo "  Model:  ${MODEL_PATH}"
echo "  Mode:   ${MODE}"
echo "  Subset: ${SUBSET}"
echo "============================================================"

python run_evaluation_vidhalluc.py \
    --mode "${MODE}" \
    --subset "${SUBSET}" \
    --model-path "${MODEL_PATH}" \
    --video-root "${VIDEO_ROOT}" \
    --question-root "${QUESTION_ROOT}" \
    --output-dir "${OUTPUT_DIR}" \
    --device cuda \
    --num-frames 16 \
    --fda-layer-start 8 \
    --fda-layer-end 16 \
    --fda-alpha 0.5 \
    --deep-layer 22 \
    --temporal-dims 4 \
    --prior-dims 4 \
    --gamma-temporal 1.2 \
    --gamma-prior 0.5 \
    ${EXTRA_FLAGS}
