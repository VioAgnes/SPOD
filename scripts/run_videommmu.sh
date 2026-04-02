#!/bin/bash
# =============================================================================
# SPOD VideoMMMU Evaluation Launcher
# =============================================================================
# Usage:
#   bash run_videommmu.sh <mode> <subset> [--extra-flags]
#
# Modes:
#   spod       SPOD-enhanced evaluation (default)
#   baseline   Standard evaluation without SPOD
#   test       Quick test mode (3 samples per subset)
#
# Subsets:
#   perception      Visual information extraction questions
#   comprehension   Concept understanding questions
#   adaptation      Knowledge application questions
#   all             All subsets combined
#
# Example:
#   bash run_videommmu.sh spod all
#   bash run_videommmu.sh baseline perception
#   bash run_videommmu.sh test all
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

MODE="${1:-spod}"
SUBSET="${2:-all}"

# Handle test mode
TEST_FLAG=""
FINAL_MODE="${MODE}"
FINAL_OUTPUT_DIR="${OUTPUT_DIR:-./outputs/videommmu}"

if [[ "${MODE}" == "test" ]]; then
    TEST_FLAG="--test-mode"
    FINAL_MODE="spod"
    FINAL_OUTPUT_DIR="${OUTPUT_DIR:-./outputs/videommmu_test}"
fi

# Environment defaults
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-VL-7B-Instruct}"
VIDEO_ROOT="${VIDEO_ROOT:-./DATA/VideoMMMU}"

echo "============================================================"
echo "  SPOD — VideoMMMU Evaluation"
echo "  Mode:   ${FINAL_MODE}"
echo "  Subset: ${SUBSET}"
echo "  Model:  ${MODEL_PATH}"
echo "  FDA:    L8-L16 | Deep: L22"
echo "============================================================"

python -m src.experiments.downstream.run_videommmu \
    --model-path "${MODEL_PATH}" \
    --subset "${SUBSET}" \
    --mode "${FINAL_MODE}" \
    --output-dir "${FINAL_OUTPUT_DIR}" \
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
