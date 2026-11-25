#!/usr/bin/env bash
set -euo pipefail
REPOSITORY=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPOSITORY"
PYTHON=${PYTHON:-python}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/vidhalluc}
CONFIG=${CONFIG:-configs/spod.json}
"$PYTHON" -m vidhalluc evaluate "$OUTPUT_DIR" --config "$CONFIG" "$@"
