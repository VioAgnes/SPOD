#!/bin/bash
# =============================================================================
# SPOD VideoMMMU Evaluation Launcher
# =============================================================================

set -e

MODE="${1:-spod}"
SUBSET="${2:-all}"
shift 2 || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

TEST_FLAG=""
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/videommmu}"

if [[ "$MODE" == "test" ]]; then
    TEST_FLAG="--test-mode"
    MODE="spod"
    OUTPUT_DIR="${OUTPUT_DIR}_test"
fi

# Default: Qwen2.5-VL-7B
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-VL-7B-Instruct}"
VIDEO_ROOT="${VIDEO_ROOT:-/root/autodl-tmp/DATA/VideoMMMU}"

echo "============================================================"
echo "  SPOD VideoMMMU Evaluation"
echo "  Model:  ${MODEL_PATH}"
echo "  Mode:   ${MODE}"
echo "  Subset: ${SUBSET}"
echo "  FDA:    L8-L16 | Deep: L22"
echo "============================================================"

python -m src.experiments.downstream.evaluate \
    --model-path "${MODEL_PATH}" \
    --subset "${SUBSET}" \
    --mode "${MODE}" \
    --output-dir "${OUTPUT_DIR}" \
    --device cuda \
    --num-frames 16 \
    --max-new-tokens 1024 \
    --video-root "${VIDEO_ROOT}" \
    --fda-layer-start 8 \
    --fda-layer-end 16 \
    --fda-alpha 0.5 \
    --deep-layer 22 \
    --temporal-dims 4 \
    --prior-dims 4 \
    --gamma-temporal 1.2 \
    --gamma-prior 0.5 \
    --gamma-residual 1.0 \
    ${TEST_FLAG}
