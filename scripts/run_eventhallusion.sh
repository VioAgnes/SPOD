#!/bin/bash
# =============================================================================
# SPOD EventHallusion Evaluation Launcher
# =============================================================================
# Usage:
#   bash run_eventhallusion.sh <mode> <subset> [--extra-flags]
#
# Modes:
#   spod       SPOD-enhanced evaluation (default)
#   baseline   Standard evaluation without SPOD
#
# Subsets:
#   entire      Videos with a single event throughout
#   interleave  Videos with multiple interleaved events
#   misleading  Videos designed to mislead with prior biases
#   all         All subsets combined
#
# Example:
#   bash run_eventhallusion.sh spod all
#   bash run_eventhallusion.sh baseline entire
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

MODE="${1:-spod}"
SUBSET="${2:-all}"
EXTRA_FLAGS="${@:3}"

# Environment defaults
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-VL-7B-Instruct}"
VIDEO_ROOT="${VIDEO_ROOT:-./DATA/eventhallusion}"
QUESTION_ROOT="${QUESTION_ROOT:-./DATA/EventHallusion/questions}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/eventhallusion}"

echo "============================================================"
echo "  SPOD — EventHallusion Evaluation"
echo "  Mode:   ${MODE}"
echo "  Subset: ${SUBSET}"
echo "  Model:  ${MODEL_PATH}"
echo "============================================================"

python -m src.experiments.downstream.run_eventhallusion \
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
