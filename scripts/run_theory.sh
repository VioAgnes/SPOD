#!/bin/bash
# =============================================================================
# SPOD Theory Analysis Launcher
# =============================================================================
# Usage:
#   bash run_theory.sh <subset> <num_samples> [pillar] [--extra-flags]
#
# Subsets:
#   perception      Perception-level questions (default)
#   comprehension   Comprehension-level questions
#   adaptation      Adaptation-level questions
#   all             All subsets
#
# Num_samples: Number of samples to analyze (default: 30)
#
# Pillars:
#   FDA    Analyze Frame Distinctiveness Analysis
#   TDS    Analyze Temporal Delta Subspace
#   FDA,TDS Both pillars (default)
#
# Special modes:
#   bash run_theory.sh test 3    Quick 3-sample test
#
# Example:
#   bash run_theory.sh perception 30
#   bash run_theory.sh all 100 FDA,TDS
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

SUBSET="${1:-perception}"
NUM_SAMPLES="${2:-30}"
PILLAR="${3:-FDA,TDS}"

# Quick test mode
if [[ "${SUBSET}" == "test" ]]; then
    SUBSET="perception"
    NUM_SAMPLES=3
fi

# Environment defaults
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-VL-7B-Instruct}"
VIDEO_ROOT="${VIDEO_ROOT:-./DATA/VideoMMMU}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/theory}"

echo "============================================================"
echo "  SPOD — Theory Analysis"
echo "  Subset: ${SUBSET} | Samples: ${NUM_SAMPLES}"
echo "  Pillars: ${PILLAR}"
echo "  Model:  ${MODEL_PATH}"
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
