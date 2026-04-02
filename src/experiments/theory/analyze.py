#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SPOD Theory Analysis Entry Point
================================

Runs the three-pillar theoretical experiments validating SPOD:
- FDA Signal Exists: Hidden-state variance analysis
- FDA Signal Valuable: FDA vs random frame comparison
- TDS Construction Valid: Temporal subspace analysis
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tqdm import tqdm

from src.models.video_io import load_video, find_local_video
from src.theory.fda import analyze_fda_signal
from src.theory.fda_value import analyze_fda_frame_contribution
from src.theory.tds import analyze_tds_subspace
from src.models.spod import compute_fda_frame_weights


SUBSET_CONFIG = {
    "perception":    ("Perception",    9,  28, 22, 16, 8),
    "comprehension": ("Comprehension", 9,  25, 20, 16, 8),
    "adaptation":    ("Adaptation",    9,  22, 18, 12, 6),
    "all":           ("all",           9,  28, 22, 16, 8),
}


def parse_args():
    parser = argparse.ArgumentParser(description="SPOD Theory Analysis")
    parser.add_argument("--subset", type=str, default="perception",
                        choices=["perception", "comprehension", "adaptation", "all"])
    parser.add_argument("--num-samples", type=int, default=30)
    parser.add_argument("--pillar", type=str, default="FDA,TDS")
    parser.add_argument("--output-dir", type=str, default="./outputs/theory")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model-path", type=str,
                        default="/root/autodl-tmp/MODELS/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--video-root", type=str,
                        default="/root/autodl-tmp/DATA/VideoMMMU")
    parser.add_argument("--attn-impl", type=str, default="flash_attention_2")
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--temporal-dims", type=int, default=16)
    parser.add_argument("--prior-dims", type=int, default=8)
    parser.add_argument("--smooth-window", type=int, default=3)
    parser.add_argument("--deep-layer", type=int, default=22)
    parser.add_argument("--fda-layer-start", type=int, default=9)
    parser.add_argument("--fda-layer-end", type=int, default=28)
    parser.add_argument("--test-mode", action="store_true")
    return parser.parse_args()


def load_model(model_path: str, device: str, attn_impl: str):
    """Load Qwen2.5-VL model and processor."""
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    print(f"[MODEL] Loading {model_path} ...")
    t0 = time.time()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        attn_implementation=attn_impl,
    )
    processor = AutoProcessor.from_pretrained(model_path)
    print(f"[MODEL] Loaded in {time.time()-t0:.1f}s")
    return model, processor


def run_theory_analysis(args) -> dict:
    """Run the complete theory analysis pipeline."""
    sub_name, layer_start, layer_end, deep_layer, temporal_dims, prior_dims = \
        SUBSET_CONFIG.get(args.subset, SUBSET_CONFIG["perception"])

    video_root = Path(args.video_root)
    parquet_file = video_root / sub_name / "test-00000-of-00001.parquet"

    if not parquet_file.exists():
        print(f"[ERROR] Parquet file not found: {parquet_file}")
        return {}

    import pyarrow.parquet as pq
    table = pq.read_table(str(parquet_file))
    n_rows = table.num_rows
    n_samples = min(args.num_samples, n_rows)
    if args.test_mode:
        n_samples = min(3, n_rows)

    pillars = [p.strip() for p in args.pillar.split(",")]

    model, processor = load_model(args.model_path, args.device, args.attn_impl)
    model.eval()

    fda_all, tds_all = [], []

    for idx in tqdm(range(n_samples), desc=f"Theory [{args.subset}]"):
        row = table.slice(idx, 1).to_pydict()
        sample = {k: (v[0] if isinstance(v, list) and len(v) == 1 else v)
                  for k, v in row.items()}
        sample_id = sample.get("id", f"{args.subset}_{idx}")
        video_url = sample.get("video", "")

        video_path = find_local_video(sample_id, video_root, subset=args.subset)
        if video_path is None:
            video_path = video_url

        try:
            frames = load_video(video_path, num_frames=args.num_frames)
        except Exception as e:
            print(f"[WARN] Failed to load {sample_id}: {e}")
            continue

        from PIL import Image
        images = [Image.fromarray(f).convert("RGB") for f in frames]

        messages = [{"role": "user", "content": [
            {"type": "image", "image": images[0]},
            {"type": "text", "text": "Describe the visual content briefly."},
        ]}]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=images, return_tensors="pt").to(model.device)

        hidden_outputs = {}

        def make_hook(lidx):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    hidden_outputs[lidx] = output[0].detach()
                elif torch.is_tensor(output):
                    hidden_outputs[lidx] = output.detach()
            return hook

        hooks = []
        capture_range = list(range(layer_start, layer_end + 1))
        for lidx in capture_range:
            if lidx < len(model.model.layers):
                h = model.model.layers[lidx].register_forward_hook(make_hook(lidx))
                hooks.append(h)

        try:
            with torch.no_grad():
                _ = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    pixel_values=inputs["pixel_values"],
                    output_hidden_states=True,
                )
        finally:
            for h in hooks:
                h.remove()

        if not hidden_outputs:
            continue

        vision_seq_len = args.num_frames
        hidden_tuple = tuple(hidden_outputs[k] for k in sorted(hidden_outputs.keys()))

        if "FDA" in pillars:
            fda = analyze_fda_signal(
                hidden_states=hidden_tuple,
                vision_seq_len=vision_seq_len,
                num_frames=args.num_frames,
                layer_start=layer_start,
                layer_end=layer_end,
                smooth_window=args.smooth_window,
            )
            fda["sample_id"] = sample_id
            fda_all.append(fda)

        if "TDS" in pillars:
            tds = analyze_tds_subspace(
                hidden_states=hidden_tuple,
                vision_seq_len=vision_seq_len,
                num_frames=args.num_frames,
                deep_layer=deep_layer,
                temporal_dims=temporal_dims,
                prior_dims=prior_dims,
                layer_start=layer_start,
                layer_end=layer_end,
            )
            tds["sample_id"] = sample_id
            tds_all.append(tds)

    return {
        "fda_results": fda_all,
        "tds_results": tds_all,
        "config": vars(args),
    }


def summarize_results(results: dict):
    """Print a summary of theory experiment results."""
    n_fda = len(results.get("fda_results", []))
    n_tds = len(results.get("tds_results", []))

    print("\n" + "=" * 70)
    print(f"  Theory Analysis Summary  (FDA: {n_fda}, TDS: {n_tds})")
    print("=" * 70)

    if n_fda > 0:
        all_vr = []
        for r in results["fda_results"]:
            all_vr.extend(list(r.get("variance_ratio", {}).values()))
        if all_vr:
            print(f"\n  [FDA] Variance Ratio (High-phi / Low-phi):")
            print(f"         Mean ± Std : {np.mean(all_vr):.2f} ± {np.std(all_vr):.2f}")
            print(f"         > 1.5x ratio: {sum(1 for v in all_vr if v > 1.5)/len(all_vr):.1%}")

    if n_tds > 0:
        eff_ranks = [r["effective_rank"] for r in results["tds_results"]]
        print(f"\n  [TDS] Effective Rank:")
        print(f"         Mean: {np.mean(eff_ranks):.2f} ± {np.std(eff_ranks):.2f}")

    print()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / args.subset / "analysis_results.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  SPOD Theory Analysis")
    print(f"  Subset: {args.subset}, Samples: {args.num_samples}")
    print(f"  Pillars: {args.pillar}")
    print(f"  Model: {args.model_path}")
    print("=" * 70)

    results = run_theory_analysis(args)

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to: {out_file}")

    summarize_results(results)


if __name__ == "__main__":
    main()
