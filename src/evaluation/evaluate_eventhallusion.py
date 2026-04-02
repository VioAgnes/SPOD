"""
SPOD Evaluation for EventHallusion Dataset
=====================================

Evaluates Qwen2.5-VL with SPOD on the EventHallusion benchmark.
"""
from __future__ import annotations

import json
import os
import base64
import io
from pathlib import Path
from typing import Any, Optional
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from ..models.spod import (
    compute_fda_frame_weights,
    build_tds_subspaces,
    enhance_hidden_state,
)
from ..models.video_io import load_video


# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------

def img_to_data_url(img: Image.Image, format: str = "JPEG", quality: int = 95) -> str:
    """Convert PIL Image to base64 data URL."""
    buffer = io.BytesIO()
    img.save(buffer, format=format, quality=quality)
    img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/{format.lower()};base64,{img_str}"


JSON_SPLIT_TO_FOLDER = {
    "entire": "entire",
    "interleave": "interleave",
    "misleading": "misleading",
}

QUESTION_FILE_MAP = {
    "entire": "entire_questions.json",
    "interleave": "mix_questions.json",
    "misleading": "misleading_questions.json",
}

VIDEO_FILENAME_PREFIX = {
    "entire": "entire",
    "interleave": "interleave",
    "misleading": "misleading",
}


def extract_yes_no(response: str) -> Optional[str]:
    """Extract Yes/No from model response."""
    if response is None:
        return None
    resp = response.lower().strip()
    if resp.startswith("yes"):
        return "Yes."
    if resp.startswith("no"):
        return "No."
    return None


# ----------------------------------------------------------------------
# SPOD-aware generator
# ----------------------------------------------------------------------

@dataclass
class SPODGenerator:
    """SPOD-augmented generator for EventHallusion binary QA."""

    model: Any
    processor: Any
    args: Any
    V_r: torch.Tensor | None = None
    P_q: torch.Tensor | None = None
    frame_weights: torch.Tensor | None = None
    vision_seq_len: int = 0
    num_frames: int = 0
    _subspaces_ready: bool = False
    _debug_buffer: dict | None = None

    @property
    def fda_layer_start(self) -> int:
        return getattr(self.args, "fda_layer_start", 9)

    @property
    def fda_layer_end(self) -> int:
        return getattr(self.args, "fda_layer_end", 28)

    @property
    def fda_alpha(self) -> float:
        return getattr(self.args, "fda_alpha", 0.7)

    @property
    def smooth_window(self) -> int:
        return getattr(self.args, "smooth_window", 3)

    @property
    def deep_layer(self) -> int:
        return getattr(self.args, "deep_layer", 22)

    @property
    def temporal_dims(self) -> int:
        return getattr(self.args, "temporal_dims", 16)

    @property
    def prior_dims(self) -> int:
        return getattr(self.args, "prior_dims", 8)

    @property
    def gamma_T(self) -> float:
        return getattr(self.args, "gamma_temporal", 1.0)

    @property
    def gamma_P(self) -> float:
        return getattr(self.args, "gamma_prior", 0.5)

    @property
    def gamma_R(self) -> float:
        return getattr(self.args, "gamma_residual", 1.0)

    def set_vision_params(self, vision_seq_len: int, num_frames: int):
        self.vision_seq_len = vision_seq_len
        self.num_frames = num_frames
        self._subspaces_ready = False

    def _build_subspaces(self, hidden_states: tuple, attentions: tuple | None = None,
                          mm_token_type_ids: torch.Tensor | None = None):
        if self._subspaces_ready:
            return
        device = hidden_states[0].device

        self.frame_weights = compute_fda_frame_weights(
            hidden_states=hidden_states,
            vision_seq_len=self.vision_seq_len,
            num_frames=self.num_frames,
            layer_start=self.fda_layer_start,
            layer_end=self.fda_layer_end,
            alpha=self.fda_alpha,
            smooth_window=self.smooth_window,
            device=device,
            attentions=attentions,
            mm_token_type_ids=mm_token_type_ids,
        )

        self.V_r, self.P_q = build_tds_subspaces(
            hidden_states=hidden_states,
            vision_seq_len=self.vision_seq_len,
            num_frames=self.num_frames,
            frame_weights=self.frame_weights,
            deep_layer=self.deep_layer,
            temporal_dims=self.temporal_dims,
            prior_dims=self.prior_dims,
        )
        self._subspaces_ready = True

        self._debug_buffer = {
            "frame_weights": self.frame_weights.detach().cpu().tolist(),
        }

    def _enhanced_logits(self, hidden_states: tuple) -> torch.Tensor:
        """Compute SPOD-enhanced logits from the final hidden state."""
        h_final = hidden_states[-1][:, -1, :]

        if not self._subspaces_ready:
            logits = torch.matmul(h_final, self.model.lm_head.weight.t())
            if self.model.lm_head.bias is not None:
                logits = logits + self.model.lm_head.bias
            return logits.unsqueeze(1)

        h_enhanced = enhance_hidden_state(
            h_final, self.V_r, self.P_q,
            gamma_T=self.gamma_T,
            gamma_P=self.gamma_P,
            gamma_R=self.gamma_R,
        )

        logits = torch.matmul(h_enhanced, self.model.lm_head.weight.t())
        if self.model.lm_head.bias is not None:
            logits = logits + self.model.lm_head.bias
        return logits.unsqueeze(1)

    def generate(
        self,
        messages: list,
        images: list | None = None,
        max_new_tokens: int | None = None,
        temperature: float = 0.0,
    ) -> dict:
        """SPOD-augmented generation for EventHallusion."""
        max_new_tokens = max_new_tokens or getattr(self.args, "max_new_tokens", 256)
        temperature = temperature or getattr(self.args, "temperature", 0.0)
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
        inputs = {
            k: (v.to(device) if torch.is_tensor(v) else v)
            for k, v in inputs.items()
        }

        image_grid_thw = inputs.get("image_grid_thw")
        num_frames = 1
        if image_grid_thw is not None:
            num_frames = image_grid_thw.shape[0]
        else:
            num_frames = getattr(self.args, "num_frames", 4)

        vision_seq_len = 0
        if image_grid_thw is not None:
            for i in range(num_frames):
                thw = image_grid_thw[i]
                T_i = int(thw[0].item()) if thw.dim() > 0 else 1
                H_i = int(thw[1].item()) if thw.dim() > 0 else int(thw[0].item())
                W_i = int(thw[2].item()) if thw.dim() > 0 else int(thw[1].item())
                vision_seq_len += T_i * (H_i // 2) * (W_i // 2)
        elif inputs.get("pixel_values") is not None:
            vision_seq_len = inputs["pixel_values"].shape[0]

        self.set_vision_params(vision_seq_len, num_frames)

        if self.mode != "baseline" and vision_seq_len > 0:
            mm_ids = inputs.get("mm_token_type_ids")

            with torch.inference_mode():
                outputs = self.model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    position_ids=inputs.get("position_ids"),
                    pixel_values=inputs.get("pixel_values"),
                    image_grid_thw=inputs.get("image_grid_thw"),
                    video_grid_thw=inputs.get("video_grid_thw"),
                    mm_token_type_ids=mm_ids,
                    output_hidden_states=True,
                    output_attentions=False,
                    use_cache=True,
                    return_dict=True,
                )

            self._build_subspaces(outputs.hidden_states, attentions=None,
                                  mm_token_type_ids=mm_ids)
            logits_enhanced = self._enhanced_logits(outputs.hidden_states)

            if do_sample:
                probs = torch.softmax(logits_enhanced / temperature, dim=-1)
                first_token = torch.multinomial(probs.squeeze(1), num_samples=1)
            else:
                first_token = torch.argmax(logits_enhanced.squeeze(1), dim=-1, keepdim=True)

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
                    use_cache=True,
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
                    use_cache=True,
                )

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs["input_ids"], output_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return {
            "response": response,
            "debug": self._debug_buffer.copy() if self._debug_buffer else {},
        }

    def generate_description(
        self,
        images: list | None = None,
        max_new_tokens: int | None = None,
        temperature: float = 0.0,
    ) -> dict:
        """Generate a detailed description of the video content."""
        max_new_tokens = max_new_tokens or 512

        if images:
            content = [
                *[{'type': 'image', 'image': img} for img in images],
                {'type': 'text', 'text': 'Please describe this video in detail.'}
            ]
        else:
            content = [{"type": "text", "text": "Please describe this video in detail."}]

        messages = [{"role": "user", "content": content}]
        self._subspaces_ready = False

        return self.generate(
            messages=messages,
            images=images,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

    @property
    def mode(self) -> str:
        return getattr(self.args, "mode", "spod")


# ----------------------------------------------------------------------
# Main evaluation function
# ----------------------------------------------------------------------

def evaluate_eventhallusion(
    model,
    processor,
    args,
    mode: str = "spod",
    splits: list[str] | None = None,
    generate_description: bool = True,
) -> dict:
    """Evaluate SPOD on EventHallusion.

    Args:
        model: Loaded Qwen2.5-VL model.
        processor: Loaded Qwen2.5-VL processor.
        args: Parsed argument namespace.
        mode: "spod" (with SPOD enhancement) or "baseline" (without).
        splits: List of splits to evaluate.
        generate_description: Whether to generate video descriptions.

    Returns:
        Dict with comprehensive metrics.
    """
    if splits is None:
        splits = ["entire", "interleave", "misleading"]

    if hasattr(args, "mode") and args.mode:
        mode = args.mode

    video_root = getattr(args, "video_root", "./DATA/eventhallusion")
    question_root = getattr(args, "question_root", "./DATA/EventHallusion/questions")
    num_frames = getattr(args, "num_frames", 16)
    max_new_tokens = getattr(args, "max_new_tokens", 256)
    desc_max_tokens = getattr(args, "desc_max_tokens", 512)
    output_dir = Path(getattr(args, "output_dir", "./outputs/eventhallusion"))
    output_dir.mkdir(parents=True, exist_ok=True)

    generator = SPODGenerator(model=model, processor=processor, args=args)

    all_predictions = []
    total_correct = 0
    total_questions = 0
    total_non_matching = 0
    per_split_correct = {}
    per_split_total = {}
    per_split_non_matching = {}

    for split in splits:
        json_name = QUESTION_FILE_MAP.get(split, f"{split}_questions.json")
        question_file = Path(question_root) / json_name
        video_folder = Path(video_root) / "videos" / JSON_SPLIT_TO_FOLDER.get(split, split)

        if not question_file.exists():
            print(f"[WARN] Question file not found: {question_file}, skipping.")
            continue
        if not video_folder.exists():
            print(f"[WARN] Video folder not found: {video_folder}, skipping.")
            continue

        with open(question_file, "r", encoding="utf-8") as f:
            question_data = json.load(f)

        if getattr(args, "test_mode", False):
            question_data = question_data[:3]

        n_samples = len(question_data)
        print(f"\n{'='*60}")
        print(f"  Split: {split} | Videos: {n_samples}")
        print(f"{'='*60}")

        split_correct = 0
        split_total = 0
        split_non_matching = 0

        for video_info in tqdm(question_data, desc=f"{mode}/{split}"):
            vid = video_info["id"]
            questions = video_info.get("questions", [])
            event_info = video_info.get("event_info", {})

            file_prefix = VIDEO_FILENAME_PREFIX.get(split, split)
            suffix = vid.rsplit("_", 1)[-1]
            video_filename = f"{file_prefix}_{suffix}.mp4"
            video_path = video_folder / video_filename
            if not video_path.exists():
                print(f"[WARN] Video not found: {video_path}")
                continue

            try:
                frames = load_video(str(video_path), num_frames=num_frames)
                images = [Image.fromarray(f).convert("RGB") for f in frames]
            except Exception as e:
                print(f"[WARN] Failed to load video {vid}: {e}")
                images = []

            generator._subspaces_ready = False
            generator.V_r = None
            generator.P_q = None
            generator.frame_weights = None
            generator._debug_buffer = None

            pred_description = ""
            if generate_description and split != "misleading" and images:
                try:
                    desc_result = generator.generate_description(
                        images=images,
                        max_new_tokens=desc_max_tokens,
                        temperature=0.0,
                    )
                    pred_description = desc_result.get("response", "")
                except Exception as e:
                    print(f"[ERROR] Description generation failed for {vid}: {e}")
                    pred_description = ""

            generator._subspaces_ready = False

            for qa in questions:
                question_text = qa["question"]
                full_question = f"{question_text}\nPlease answer yes or no."
                gt_answer = qa["answer"]

                if images:
                    content = [
                        *[{ "type": "image", "image": img } for img in images],
                        {"type": "text", "text": full_question},
                    ]
                else:
                    content = [{"type": "text", "text": full_question}]

                messages = [{"role": "user", "content": content}]
                result = None
                try:
                    result = generator.generate(
                        messages=messages,
                        images=images,
                        max_new_tokens=max_new_tokens,
                        temperature=0.0,
                    )
                    response = result["response"]
                except Exception as e:
                    response = f"[ERROR] {str(e)}"
                    print(f"[ERROR] Generation failed for {vid}: {e}")

                pred_answer = extract_yes_no(response)

                is_correct = 0
                if pred_answer is not None:
                    if pred_answer == gt_answer:
                        is_correct = 1
                else:
                    split_non_matching += 1

                split_correct += is_correct
                split_total += 1
                total_correct += is_correct
                total_questions += 1

                all_predictions.append({
                    "video_id": vid,
                    "split": split,
                    "question": question_text,
                    "ground_truth": gt_answer,
                    "prediction": pred_answer,
                    "response": response,
                    "correct": bool(is_correct),
                    "description": pred_description,
                    "event_caption": event_info.get("caption", ""),
                    "event_unexpected": event_info.get("unexpected", ""),
                })

        split_acc = split_correct / split_total if split_total > 0 else 0.0
        per_split_correct[split] = split_correct
        per_split_total[split] = split_total
        per_split_non_matching[split] = split_non_matching

        print(f"\n  {split} Accuracy: {split_correct}/{split_total} correct, "
              f"acc={split_acc:.4f}, non_match={split_non_matching}")

        split_out = output_dir / split
        split_out.mkdir(parents=True, exist_ok=True)

        with open(split_out / f"{mode}_predictions.json", "w", encoding="utf-8") as f:
            json.dump({
                "split": split,
                "mode": mode,
                "correct": split_correct,
                "total": split_total,
                "accuracy": split_acc,
                "non_matching": split_non_matching,
            }, f, ensure_ascii=False, indent=2)

    overall_accuracy = total_correct / total_questions if total_questions > 0 else 0.0

    print(f"\n{'='*60}")
    print(f"  OVERALL RESULTS ({mode})")
    print(f"{'='*60}")
    print(f"  OVERALL Accuracy: {total_correct}/{total_questions} correct, "
          f"acc={overall_accuracy:.4f}")

    results = {
        "mode": mode,
        "splits": splits,
        "overall_accuracy": overall_accuracy,
        "overall_correct": total_correct,
        "overall_total": total_questions,
        "predictions": all_predictions,
    }

    combined_out = output_dir / f"{'_'.join(splits)}_{mode}_combined.json"
    with open(combined_out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nCombined results saved to: {combined_out}")
    return results
