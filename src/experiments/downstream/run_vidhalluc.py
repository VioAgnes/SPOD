#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SPOD VidHalluc Evaluation
========================
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from src.models.spod import (
    compute_fda_frame_weights,
    build_tds_subspaces,
    enhance_hidden_state,
)
from src.models.video_io import load_video


def parse_args():
    parser = argparse.ArgumentParser(description="SPOD VidHalluc Evaluation")
    parser.add_argument("--model-path", type=str,
                        default="./MODELS/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--subset", type=str, default="all",
                        choices=["ach_binaryqa", "ach_mcq", "sth", "tsh", "all"])
    parser.add_argument("--mode", type=str, default="spod",
                        choices=["spod", "baseline"])
    parser.add_argument("--video-root", type=str,
                        default="./DATA/VidHalluc/videos")
    parser.add_argument("--question-root", type=str,
                        default="./DATA/VidHalluc")
    parser.add_argument("--output-dir", type=str, default="./outputs/vidhalluc")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--attn-impl", type=str, default="flash_attention_2")
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=256)
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


SUBSET_FILES = {
    "ach_binaryqa": "ach_binaryqa.json",
    "ach_mcq": "ach_mcq.json",
    "sth": "sth.json",
    "tsh": "tsh.json",
}


class SPODGenerator:
    def __init__(self, model, processor, args):
        self.model = model
        self.processor = processor
        self.args = args
        self.V_r = None
        self.P_q = None
        self.frame_weights = None
        self.vision_seq_len = 0
        self.num_frames = 0
        self._subspaces_ready = False

    @property
    def mode(self):
        return getattr(self.args, "mode", "spod")

    def set_vision_params(self, vision_seq_len, num_frames):
        self.vision_seq_len = vision_seq_len
        self.num_frames = num_frames
        self._subspaces_ready = False

    def _build_subspaces(self, hidden_states):
        if self._subspaces_ready:
            return
        device = hidden_states[0].device

        self.frame_weights = compute_fda_frame_weights(
            hidden_states=hidden_states,
            vision_seq_len=self.vision_seq_len,
            num_frames=self.num_frames,
            layer_start=self.args.fda_layer_start,
            layer_end=self.args.fda_layer_end,
            alpha=self.args.fda_alpha,
            smooth_window=self.args.smooth_window,
            device=device,
        )

        self.V_r, self.P_q = build_tds_subspaces(
            hidden_states=hidden_states,
            vision_seq_len=self.vision_seq_len,
            num_frames=self.num_frames,
            frame_weights=self.frame_weights,
            deep_layer=self.args.deep_layer,
            temporal_dims=self.args.temporal_dims,
            prior_dims=self.args.prior_dims,
        )
        self._subspaces_ready = True

    def generate(self, messages, images=None, max_new_tokens=None, temperature=0.0):
        max_new_tokens = max_new_tokens or getattr(self.args, "max_new_tokens", 256)
        do_sample = temperature > 0

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text],
            images=images,
            return_tensors="pt",
            padding=True,
        )
        device = self.model.device
        inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}

        image_grid_thw = inputs.get("image_grid_thw")
        num_frames = 1
        if image_grid_thw is not None:
            num_frames = image_grid_thw.shape[0]

        vision_seq_len = 0
        if image_grid_thw is not None:
            for i in range(num_frames):
                thw = image_grid_thw[i]
                T_i = int(thw[0].item()) if thw.dim() > 0 else 1
                H_i = int(thw[1].item()) if thw.dim() > 0 else int(thw[0].item())
                W_i = int(thw[2].item()) if thw.dim() > 0 else int(thw[1].item())
                vision_seq_len += T_i * (H_i // 2) * (W_i // 2)

        self.set_vision_params(vision_seq_len, num_frames)

        if self.mode != "baseline" and vision_seq_len > 0:
            with torch.inference_mode():
                outputs = self.model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    pixel_values=inputs.get("pixel_values"),
                    image_grid_thw=inputs.get("image_grid_thw"),
                    output_hidden_states=True,
                    use_cache=True,
                    return_dict=True,
                )

            self._build_subspaces(outputs.hidden_states)

            h_final = outputs.hidden_states[-1][:, -1, :]
            h_enhanced = enhance_hidden_state(
                h_final, self.V_r, self.P_q,
                gamma_T=self.args.gamma_temporal,
                gamma_P=self.args.gamma_prior,
                gamma_R=self.args.gamma_residual,
            )
            logits = torch.matmul(h_enhanced, self.model.lm_head.weight.t())
            if self.model.lm_head.bias is not None:
                logits = logits + self.model.lm_head.bias

            if do_sample:
                probs = torch.softmax(logits / temperature, dim=-1)
                first_token = torch.multinomial(probs.squeeze(1), num_samples=1)
            else:
                first_token = torch.argmax(logits.squeeze(1), dim=-1, keepdim=True)

            generated_ids = torch.cat([inputs["input_ids"], first_token], dim=1)
            full_attn_mask = torch.cat([
                inputs["attention_mask"],
                torch.ones(1, 1, dtype=inputs["attention_mask"].dtype, device=device)
            ], dim=1)

            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids=generated_ids,
                    attention_mask=full_attn_mask,
                    past_key_values=outputs.past_key_values,
                    max_new_tokens=max_new_tokens - 1,
                    temperature=temperature if temperature > 0 else None,
                    do_sample=do_sample,
                )
        else:
            with torch.inference_mode():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature if temperature > 0 else None,
                    do_sample=do_sample,
                )

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs["input_ids"], output_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
        )[0]

        return {"response": response}


def extract_answer(response, options=None):
    if response is None:
        return None
    resp = response.strip()
    if options:
        for i, opt in enumerate(options):
            letter = chr(65 + i)
            if resp.upper().startswith(letter) or resp.upper().startswith(f"{letter}."):
                return letter
    resp_lower = resp.lower()
    if resp_lower.startswith("yes"):
        return "Yes"
    if resp_lower.startswith("no"):
        return "No"
    return resp.split()[0] if resp else None


def find_video_path(video_id, video_root):
    video_root = Path(video_root)
    for subfolder in ["ACH_ffmpeg_fast", "ACH_reencoded_test", "STH", "TSH"]:
        subfolder_path = video_root / subfolder
        if not subfolder_path.exists():
            continue
        video_path = subfolder_path / f"{video_id}.mp4"
        if video_path.exists():
            return str(video_path)
        parts = video_id.rsplit("_clip_", 1)
        if len(parts) == 2:
            video_path = subfolder_path / f"{parts[0]}.mp4"
            if video_path.exists():
                return str(video_path)
    return None


def evaluate_vidhalluc(model, processor, args, mode, subsets):
    video_root = getattr(args, "video_root", "./DATA/VidHalluc/videos")
    question_root = getattr(args, "question_root", "./DATA/VidHalluc")
    num_frames = getattr(args, "num_frames", 8)
    max_new_tokens = getattr(args, "max_new_tokens", 256)
    output_dir = Path(getattr(args, "output_dir", "./outputs/vidhalluc"))
    output_dir.mkdir(parents=True, exist_ok=True)

    generator = SPODGenerator(model=model, processor=processor, args=args)

    all_predictions = []
    total_correct = 0
    total_questions = 0
    per_subset_correct = {}
    per_subset_total = {}

    for subset in subsets:
        json_file = SUBSET_FILES.get(subset)
        if not json_file:
            continue

        question_file = Path(question_root) / json_file
        if not question_file.exists():
            continue

        with open(question_file, "r") as f:
            data = json.load(f)

        if getattr(args, "test_mode", False):
            data = dict(list(data.items())[:3])

        n_samples = len(data)
        print(f"\n{'='*60}\n  Subset: {subset} | Videos: {n_samples}\n{'='*60}")

        subset_correct = 0
        subset_total = 0

        for video_key, questions in tqdm(data.items(), desc=f"{mode}/{subset}"):
            for qa in questions:
                question_text = qa["q"]
                answers = qa["a"]

                for video_id, gt_answer in answers.items():
                    video_path = find_video_path(video_id, video_root)
                    if not video_path:
                        continue

                    try:
                        frames = load_video(video_path, num_frames=num_frames)
                        images = [Image.fromarray(f).convert("RGB") for f in frames]
                    except:
                        images = []

                    generator._subspaces_ready = False

                    full_question = f"{question_text}\nPlease answer yes or no."
                    messages = [{"role": "user", "content": [
                        *[{"type": "image", "image": img} for img in images],
                        {"type": "text", "text": full_question},
                    ]}]

                    try:
                        result = generator.generate(messages, images=images, max_new_tokens=max_new_tokens)
                        response = result["response"]
                    except Exception as e:
                        response = f"[ERROR] {str(e)}"

                    pred_answer = extract_answer(response)

                    is_correct = 0
                    if pred_answer and str(pred_answer).lower() == str(gt_answer).lower():
                        is_correct = 1

                    subset_correct += is_correct
                    subset_total += 1
                    total_correct += is_correct
                    total_questions += 1

                    all_predictions.append({
                        "video_id": video_id,
                        "subset": subset,
                        "question": question_text,
                        "ground_truth": gt_answer,
                        "prediction": pred_answer,
                        "correct": bool(is_correct),
                    })

        subset_acc = subset_correct / subset_total if subset_total > 0 else 0.0
        per_subset_correct[subset] = subset_correct
        per_subset_total[subset] = subset_total

        print(f"\n  {subset} Accuracy: {subset_correct}/{subset_total} correct, acc={subset_acc:.4f}")

        subset_out = output_dir / subset
        subset_out.mkdir(parents=True, exist_ok=True)
        with open(subset_out / f"{mode}_predictions.json", "w") as f:
            json.dump({"subset": subset, "mode": mode, "correct": subset_correct,
                      "total": subset_total, "accuracy": subset_acc}, f)

    overall_accuracy = total_correct / total_questions if total_questions > 0 else 0.0

    print(f"\n{'='*60}\n  OVERALL ({mode})\n{'='*60}")
    print(f"  Accuracy: {total_correct}/{total_questions} correct, acc={overall_accuracy:.4f}")

    results = {
        "mode": mode,
        "overall_accuracy": overall_accuracy,
        "overall_correct": total_correct,
        "overall_total": total_questions,
        "predictions": all_predictions,
    }

    combined_out = output_dir / f"{'_'.join(subsets)}_{mode}_combined.json"
    with open(combined_out, "w") as f:
        json.dump(results, f, indent=2)

    return results


def main():
    args = parse_args()
    if args.test_mode:
        args.subset = "all"

    subsets = ["ach_binaryqa", "ach_mcq", "sth", "tsh"] if args.subset == "all" else [args.subset]
    mode = args.mode

    print("=" * 70)
    print("  SPOD VidHalluc Evaluation")
    print(f"  Mode: {mode.upper()}")
    print(f"  Subsets: {subsets}")
    print("=" * 70)

    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    print(f"\n[MODEL] Loading {args.model_path} ...")
    t0 = time.time()

    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16, device_map=args.device,
            attn_implementation=args.attn_impl,
        )
    except Exception:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16, device_map=args.device,
            attn_implementation="eager",
        )

    processor = AutoProcessor.from_pretrained(args.model_path)
    model.eval()
    print(f"[MODEL] Loaded in {time.time()-t0:.1f}s")

    results = evaluate_vidhalluc(model, processor, args, mode, subsets)
    print(f"\n[DONE] Overall Accuracy: {results['overall_accuracy']:.4f}")


if __name__ == "__main__":
    main()
