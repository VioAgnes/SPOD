#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SPOD VideoMMMU Evaluation Entry Point
====================================
"""
import argparse
import sys
import time
from pathlib import Path

import torch

from src.evaluation.evaluator import evaluate_videommmu


def parse_args():
    parser = argparse.ArgumentParser(description="SPOD VideoMMMU Evaluation")
    parser.add_argument("--model-path", type=str,
                        default="./MODELS/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--subset", type=str, default="all",
                        choices=["perception", "comprehension", "adaptation", "all"])
    parser.add_argument("--mode", type=str, default="spod",
                        choices=["spod", "baseline"])
    parser.add_argument("--output-dir", type=str, default="./outputs/videommmu")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--attn-impl", type=str, default="flash_attention_2")
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--video-root", type=str,
                        default="./DATA/VideoMMMU")
    parser.add_argument("--num-samples", type=int, default=None)
    # FDA hyperparameters
    parser.add_argument("--fda-layer-start", type=int, default=8)
    parser.add_argument("--fda-layer-end", type=int, default=16)
    parser.add_argument("--fda-alpha", type=float, default=0.5)
    parser.add_argument("--smooth-window", type=int, default=3)
    # TDS hyperparameters
    parser.add_argument("--deep-layer", type=int, default=22)
    parser.add_argument("--temporal-dims", type=int, default=4)
    parser.add_argument("--prior-dims", type=int, default=4)
    # SPOD enhancement scaling
    parser.add_argument("--gamma-temporal", type=float, default=1.2)
    parser.add_argument("--gamma-prior", type=float, default=0.5)
    parser.add_argument("--gamma-residual", type=float, default=1.0)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--test-mode", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.test_mode:
        args.num_samples = 3
        args.output_dir = args.output_dir + "_test"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  SPOD — VideoMMMU Evaluation")
    print(f"  Mode: {args.mode.upper()}")
    print(f"  Subset: {args.subset}")
    print(f"  Model: {args.model_path}")
    print(f"  FDA layers: {args.fda_layer_start}–{args.fda_layer_end}")
    print(f"  TDS dims: r={args.temporal_dims}, q={args.prior_dims}")
    print(f"  SPOD: gamma_T={args.gamma_temporal}, gamma_P={args.gamma_prior}, gamma_R={args.gamma_residual}")
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
    except Exception as e:
        print(f"[WARN] Failed to load with {args.attn_impl}: {e}")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            device_map=args.device,
            attn_implementation="eager",
        )

    processor = AutoProcessor.from_pretrained(args.model_path)
    model.eval()
    print(f"[MODEL] Loaded in {time.time()-t0:.1f}s")

    results = evaluate_videommmu(
        model=model,
        processor=processor,
        args=args,
        mode=args.mode,
        subset=args.subset,
    )

    out_file = output_dir / args.subset / f"{args.mode}_predictions.json"
    print(f"\n[DONE] Results saved to: {out_file}")
    print(f"[METRIC] Accuracy: {results['accuracy']:.4f}")


if __name__ == "__main__":
    main()
