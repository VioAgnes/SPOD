"""Load the rank-8 method settings and benchmark prompts."""

import json
from importlib.resources import files
from pathlib import Path


def load_benchmark():
    resource = files("vidhalluc").joinpath("resources/benchmark.json")
    return json.loads(resource.read_text(encoding="utf-8"))


def load_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    fixed = {
        "name": "spod",
        "source_layer": 22,
        "rank": 4,
        "prior_rank": 4,
        "spatial_rank": 8,
    }
    for key, value in fixed.items():
        if config.get(key) != value:
            raise ValueError(f"{key} must be {value!r}.")
    if config.get("weight") != "fused":
        raise ValueError("weight must be fused.")
    if any(config.get(key, 0) <= 0 for key in ["gT", "gP", "gR", "video_gain"]):
        raise ValueError("All gains must be positive.")
    if tuple(config[key] for key in ["gT", "gP", "gR"]) == (1, 1, 1):
        raise ValueError("Projection gains must enable the method.")
    if (
        not 1
        <= config.get("attention_start", 0)
        <= config.get("attention_end", 0)
        <= 28
    ):
        raise ValueError("Invalid attention layer range.")
    return config
