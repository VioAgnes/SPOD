"""Check local assets and CUDA before starting a long experiment."""

import argparse
import importlib.util
import sys
from pathlib import Path

from vidhalluc.paths import DATA_DIR, MODEL_DIR


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("demo", "full", "full4"), default="demo")
    args = parser.parse_args()
    errors = []
    for name in (
        "torch",
        "transformers",
        "numpy",
        "decord",
        "av",
        "accelerate",
        "PIL",
        "sklearn",
    ):
        if importlib.util.find_spec(name) is None:
            errors.append(f"Missing Python dependency: {name}")
    for name in ("sth", "tsh", "ach_binaryqa", "ach_mcq"):
        path = DATA_DIR / f"{name}.json"
        if not path.is_file():
            errors.append(f"Missing annotation file: {path}")
    for name in ("ACH", "STH", "TSH"):
        path = DATA_DIR / "videos" / name
        if not path.is_dir():
            errors.append(f"Missing video directory: {path}")
    models = ["Qwen2.5-VL-7B-Instruct"]
    if args.mode != "demo":
        models.append("sup-simcse-roberta-large")
    for name in models:
        path = MODEL_DIR / name / "config.json"
        if not path.is_file():
            errors.append(f"Missing model config: {path}")
    if importlib.util.find_spec("torch"):
        try:
            import torch

            required = 4 if args.mode == "full4" else 1
            if torch.cuda.device_count() < required:
                errors.append(
                    f"Mode {args.mode} requires at least {required} visible CUDA GPU(s)."
                )
        except Exception as exc:
            errors.append(f"Could not load PyTorch: {exc}")
    if errors:
        print(
            "Environment check failed:\n" + "\n".join(f"- {e}" for e in errors),
            file=sys.stderr,
        )
        return 1
    print(
        "Environment check passed; inference validates annotation hashes and video files."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
