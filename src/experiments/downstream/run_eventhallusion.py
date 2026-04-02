#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SPOD EventHallusion Evaluation
============================
"""
import argparse
import sys
import time
from pathlib import Path

import torch

from src.evaluation.evaluate_eventhallusion import evaluate_eventhallusion


def parse_args():
    parser = argparse.ArgumentParser(description="SPOD EventHallusion Evaluation")
    parser.add_argument("--model-path", type=str,
                        default="/root/autodl-tmp/MODELS/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--subset", type=str, default="all",
                        choices=["entire", "interleave", "misleading", "all"])
    parser.add_argument("--mode", type=str, default="spod",
                        choices=["spod", "baseline"])
    parser.add_argument("--video-root", type=str,
                        default="/root/autodl-tmp/DATA/eventhallusion")
    parser.add_argument("--question-root", type=str,
                        default="/root/autodl-tmp/EventHallusion/questions")
    parser.add_argument("--output-dir", type=str, default="./outputs/eventhallusion")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--attn-impl", type=str, default="flash_attention_2")
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--desc-max-tokens", type=int, default=512)
    # FDA hyperparameters
    parser.add_argument("--fda-layer-start", type=int, default=9)
    parser.add_argument("--fda-layer-end", type=int, default=28)
    parser.add_argument("--fda-alpha", type=float, default=0.7)
    parser.add_argument("--smooth-window", type=int, default=3)
    # TDS hyperparameters
    parser.add_argument("--deep-layer", type=int, default=22)
    parser.add_argument("--temporal-dims", type=int, default=16)
    parser.add_argument("--prior-dims", type=int, default=8)
    # SPOD scaling
    parser.add_argument("--gamma-temporal", type=float, default=1.0)
    parser.add_argument("--gamma-prior", type=float, default=0.5)
    parser.add_argument("--gamma-residual", type=float, default=1.0)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--test-mode", action="store_true")
    return parser.parse_args()


def resolve_splits(subset_arg):
    if subset_arg == "all":
        return ["entire", "interleave", "misleading"]
    return [subset_arg]


def main():
    args = parse_args()
    if args.test_mode:
        args.subset = "all"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = resolve_splits(args.subset)

    print("=" * 70)
    print("  SPOD EventHallusion Evaluation")
    print(f"  Mode: {args.mode.upper()}")
    print(f"  Subset: {args.subset} -> {splits}")
    print(f"  Model: {args.model_path}")
    print(f"  FDA layers: {args.fda_layer_start}–{args.fda_layer_end}")
    print(f"  TDS: r={args.temporal_dims}, q={args.prior_dims}")
    print(f"  SPOD: gamma_T={args.gamma_temporal}, gamma_P={args.gamma_prior}")
    print("=" * 70)

    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    print(f"[MODEL] Loading {args.model_path} ...")
    t0 = time.time()

    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            device_map=args.device,
            attn_implementation=args.attn_impl,
        )
    except Exception:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            device_map=args.device,
            attn_implementation="eager",
        )

    processor = AutoProcessor.from_pretrained(args.model_path)
    model.eval()
    print(f"[MODEL] Loaded in {time.time()-t0:.1f}s")

    results = evaluate_eventhallusion(
        model=model,
        processor=processor,
        args=args,
        mode=args.mode,
        splits=splits,
    )

    print(f"\n[DONE] Overall Accuracy: {results['overall_accuracy']:.4f}")


if __name__ == "__main__":
    main()
