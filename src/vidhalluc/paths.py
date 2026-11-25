"""Locations of external data, checkpoints, and generated files."""

import os
from pathlib import Path


def resource_path(variable, default):
    return Path(os.environ.get(variable, str(default))).expanduser().resolve()


DATA_DIR = resource_path("VIDHALLUC_DATA_DIR", Path("data/VidHalluc"))
MODEL_DIR = resource_path("VIDHALLUC_MODEL_DIR", Path("models"))
OUTPUT_DIR = resource_path("VIDHALLUC_OUTPUT_DIR", Path("outputs"))
DECODER_CACHE = resource_path("VIDHALLUC_DECODER_CACHE", Path(".cache/vidhalluc"))
OFFICIAL_DIR = Path(__file__).resolve().parent / "evaluators"
