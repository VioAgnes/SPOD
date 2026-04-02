"""
SPOD Evaluation for VideoMMMU Dataset
==================================

SPOD-augmented inference engine for VideoMMMU benchmark evaluation.
Supports Qwen2.5-VL, VideoLlama2, VideoLlava, and Llama-3.2-Vision models.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
import json
import time

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from ..models.video_io import load_video, find_local_video, estimate_vision_seq_len, detect_model_arch
from ..models.spod import (
    compute_fda_frame_weights,
    build_tds_subspaces,
    decompose_hidden_state,
    enhance_hidden_state,
)
from ..models.model_config import get_model_config, ModelConfig


@dataclass
class EvaluationConfig:
    """Global evaluation configuration."""
    model_path: str
    device: str = "cuda"
    attn_impl: str = "flash_attention_2"
    num_frames: int = 16
    max_new_tokens: int = 1024
    temperature: float = 0.0
    output_dir: str = "./outputs"


@dataclass
class SPODGenerator:
    """SPOD-augmented inference engine.
    
    Supports any video-LLM that:
    1. Returns hidden_states through output_hidden_states=True
    2. Has an lm_head linear layer for logits computation
    3. Uses standard HuggingFace generate() method
    """

    model: Any
    processor: Any
    args: Any
    model_config: Optional[ModelConfig] = None
    V_r: torch.Tensor | None = None
    P_q: torch.Tensor | None = None
    frame_weights: torch.Tensor | None = None
    vision_seq_len: int = 0
    num_frames: int = 0
    _subspaces_ready: bool = False
    _debug_buffer: dict = field(default_factory=dict)

    def __post_init__(self):
        # Auto-detect model config if not provided
        if self.model_config is None:
            model_path = getattr(self.args, "model_path", "")
            try:
                self.model_config = get_model_config(model_path)
            except ValueError:
                # Use defaults for unknown models
                self.model_config = ModelConfig(
                    model_id=model_path,
                    arch=detect_model_arch(model_path),
                    num_layers=28,
                    hidden_dim=3584,
                    fda_layer_start=9,
                    fda_layer_end=28,
                    tds_deep_layer=22,
                    temporal_dims=16,
                    prior_dims=8,
                )

    # Cached config derived from args (with model-specific defaults)
    @property
    def fda_layer_start(self) -> int:
        default = self.model_config.fda_layer_start if self.model_config else 8
        return getattr(self.args, "fda_layer_start", default)

    @property
    def fda_layer_end(self) -> int:
        default = self.model_config.fda_layer_end if self.model_config else 16
        return getattr(self.args, "fda_layer_end", default)

    @property
    def fda_alpha(self) -> float:
        # Paper default: alpha=0.5
        default = self.model_config.fda_alpha if self.model_config else 0.5
        return getattr(self.args, "fda_alpha", default)

    @property
    def smooth_window(self) -> int:
        default = self.model_config.smooth_window if self.model_config else 3
        return getattr(self.args, "smooth_window", default)

    @property
    def deep_layer(self) -> int:
        default = self.model_config.tds_deep_layer if self.model_config else 22
        return getattr(self.args, "deep_layer", default)

    @property
    def temporal_dims(self) -> int:
        # Paper default: r=4
        default = self.model_config.temporal_dims if self.model_config else 4
        return getattr(self.args, "temporal_dims", default)

    @property
    def prior_dims(self) -> int:
        # Paper default: q=4
        default = self.model_config.prior_dims if self.model_config else 4
        return getattr(self.args, "prior_dims", default)

    @property
    def gamma_T(self) -> float:
        # Paper default: gamma_T=1.2
        return getattr(self.args, "gamma_temporal", 1.2)

    @property
    def gamma_P(self) -> float:
        # Paper default: gamma_P=0.5
        return getattr(self.args, "gamma_prior", 0.5)

    @property
    def gamma_R(self) -> float:
        # Paper default: gamma_R=1.0
        return getattr(self.args, "gamma_residual", 1.0)

    @property
    def arch(self) -> str:
        if self.model_config:
            return self.model_config.arch
        model_path = getattr(self.args, "model_path", "")
        return detect_model_arch(model_path)

    def set_vision_params(self, vision_seq_len: int, num_frames: int):
        self.vision_seq_len = vision_seq_len
        self.num_frames = num_frames
        self._subspaces_ready = False

    def _build_subspaces(self, hidden_states: tuple):
        if self._subspaces_ready:
            return
        device = hidden_states[0].device

        # FDA: Compute frame weights
        self.frame_weights = compute_fda_frame_weights(
            hidden_states=hidden_states,
            vision_seq_len=self.vision_seq_len,
            num_frames=self.num_frames,
            layer_start=self.fda_layer_start,
            layer_end=self.fda_layer_end,
            alpha=self.fda_alpha,
            smooth_window=self.smooth_window,
            device=device,
        )

        # TDS: Build temporal and non-temporal subspaces
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

        # Debug: store FDA weights
        self._debug_buffer["frame_weights"] = (
            self.frame_weights.detach().cpu().tolist()
        )

    def _apply_spod_enhancement(self, hidden_states: tuple) -> torch.Tensor:
        """Apply SPOD enhancement to final hidden state and return logits."""
        h_final = hidden_states[-1][:, -1, :]

        if not self._subspaces_ready:
            logits = torch.matmul(h_final, self.model.lm_head.weight.t())
            if self.model.lm_head.bias is not None:
                logits = logits + self.model.lm_head.bias
            return logits.unsqueeze(1)

        # Decompose and enhance
        h_enhanced = enhance_hidden_state(
            h_final, self.V_r, self.P_q,
            gamma_T=self.gamma_T,
            gamma_P=self.gamma_P,
            gamma_R=self.gamma_R,
        )

        # Apply language model head
        logits = torch.matmul(h_enhanced, self.model.lm_head.weight.t())
        if self.model.lm_head.bias is not None:
            logits = logits + self.model.lm_head.bias

        return logits.unsqueeze(1)

    def _get_layer_range(self) -> range:
        """Get the range of layers to capture for FDA analysis."""
        return range(self.fda_layer_start, self.fda_layer_end + 1)

    def _get_model_layers(self):
        """Get the model layers for hook registration.
        
        Different model architectures have different layer access patterns:
        - Qwen2.5-VL: model.model.layers
        - VideoLlama2: model.model.layers
        - VideoLlava: model.language_model.model.layers
        - Llama-3.2-Vision: model.language_model.model.layers
        """
        model = self.model
        
        # Try Qwen2.5-VL pattern
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            return model.model.layers
        
        # Try VideoLlama2/VideoLlava pattern
        if hasattr(model, 'language_model'):
            if hasattr(model.language_model, 'model') and hasattr(model.language_model.model, 'layers'):
                return model.language_model.model.layers
            if hasattr(model.language_model, 'layers'):
                return model.language_model.layers
        
        # Fallback
        if hasattr(model, 'layers'):
            return model.layers
        
        raise ValueError(f"Cannot find model layers for architecture: {self.arch}")

    def _forward_with_hooks(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        **kwargs,
    ) -> Any:
        """Forward pass with hidden state capture for FDA+TDS."""
        debug = getattr(self.args, "debug", False)
        capture_layers = list(self._get_layer_range())

        hidden_outputs = {}

        def make_hook(lidx):
            def hook(module, input, output):
                if isinstance(output, tuple) and len(output) > 0:
                    hidden_outputs[lidx] = output[0].detach()
                elif torch.is_tensor(output):
                    hidden_outputs[lidx] = output.detach()
            return hook

        hooks = []
        model_layers = self._get_model_layers()
        
        for lidx in capture_layers:
            if lidx < len(model_layers):
                hook = model_layers[lidx].register_forward_hook(make_hook(lidx))
                hooks.append(hook)

        try:
            outputs = self.model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                pixel_values=pixel_values,
                output_hidden_states=True,
                **kwargs,
            )
        finally:
            for h in hooks:
                h.remove()

        if hidden_outputs:
            sorted_keys = sorted(hidden_outputs.keys())
            hs_tuple = tuple(hidden_outputs[k] for k in sorted_keys)
            self._build_subspaces(hs_tuple)

        if debug:
            self._debug_buffer["hidden_layers_captured"] = len(hidden_outputs)
            self._debug_buffer["V_r_shape"] = (
                list(self.V_r.shape) if self.V_r is not None else None
            )
            self._debug_buffer["P_q_shape"] = (
                list(self.P_q.shape) if self.P_q is not None else None
            )

        return outputs

    def generate(self, messages: list, **kwargs) -> dict:
        """Run SPOD-augmented generation for a single video question."""
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images = kwargs.pop("images", None)

        inputs = self.processor(
            text=[text],
            images=images,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.model.device) if torch.is_tensor(v) else v
                  for k, v in inputs.items()}

        vision_info = kwargs.pop("vision_info", None)
        video_grid_thw = None
        if vision_info:
            try:
                from qwen_vl_utils import process_vision_info
                _, video_grid_thw_v = process_vision_info(vision_info)
                if video_grid_thw_v is not None:
                    video_grid_thw = video_grid_thw_v
            except Exception:
                pass

        # Estimate vision sequence length with architecture awareness
        if pixel_values := inputs.get("pixel_values"):
            video_inputs = kwargs.get("video_inputs")
            vl = estimate_vision_seq_len(video_grid_thw, video_inputs, arch=self.arch)
            if vl > 0 and self.num_frames > 0:
                self.set_vision_params(vl, self.num_frames)

        input_ids = inputs["input_ids"]
        pixel_values = inputs.get("pixel_values")
        attention_mask = inputs.get("attention_mask")
        position_ids = inputs.get("position_ids")

        # Single forward pass for SPOD
        if self.mode != "baseline" and self.vision_seq_len > 0:
            outputs = self._forward_with_hooks(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **kwargs,
            )

            # SPOD-enhanced logits
            logits_enhanced = self._apply_spod_enhancement(outputs.hidden_states)

            # Generate with enhanced first token
            if kwargs.get("temperature", 0.0) > 0:
                probs = torch.softmax(logits_enhanced / kwargs["temperature"], dim=-1)
                first_token = torch.multinomial(probs.squeeze(1), num_samples=1)
            else:
                first_token = torch.argmax(logits_enhanced.squeeze(1), dim=-1, keepdim=True)

            generated_ids = torch.cat([input_ids, first_token], dim=1)
            full_attn_mask = torch.cat([
                attention_mask,
                torch.ones(1, 1, dtype=attention_mask.dtype, device=self.model.device)
            ], dim=1)

            generated_ids = self.model.generate(
                input_ids=generated_ids,
                attention_mask=full_attn_mask,
                past_key_values=outputs.past_key_values,
                max_new_tokens=kwargs.get("max_new_tokens", self.args.max_new_tokens) - 1,
                **kwargs,
            )
        else:
            # Baseline: standard generate
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=kwargs.get("max_new_tokens", self.args.max_new_tokens),
                **kwargs,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        return {
            "response": response,
            "debug": self._debug_buffer.copy() if getattr(self.args, "debug", False) else {},
        }

    @property
    def mode(self) -> str:
        return getattr(self.args, "mode", "spod")


def evaluate_videommmu(
    model,
    processor,
    args,
    mode: str = "spod",
    subset: str = "all",
) -> dict:
    """Run SPOD or baseline evaluation on VideoMMMU.

    Args:
        model: Loaded HuggingFace model.
        processor: Loaded HuggingFace processor.
        args: Parsed argument namespace.
        mode: One of "spod" or "baseline".
        subset: One of "perception", "comprehension", "adaptation", "all".

    Returns:
        Dict with keys: predictions, accuracy, per_subset_accuracy.
    """
    import pyarrow.parquet as pq

    video_root = args.video_root
    num_frames = args.num_frames
    max_new_tokens = args.max_new_tokens
    output_dir = Path(args.output_dir) / f"{subset}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{mode}_predictions.json"

    # Load dataset
    subset_config = {
        "perception":    f"{video_root}/Perception/test-00000-of-00001.parquet",
        "comprehension": f"{video_root}/Comprehension/test-00000-of-00001.parquet",
        "adaptation":    f"{video_root}/Adaptation/test-00000-of-00001.parquet",
    }

    subsets_to_eval = (
        list(subset_config.keys())
        if subset == "all"
        else [subset]
    )

    all_predictions = []
    correct = 0
    total = 0
    per_subset_correct = {s: 0 for s in subsets_to_eval}
    per_subset_total = {s: 0 for s in subsets_to_eval}

    generator = SPODGenerator(model, processor, args)

    for sub_name in subsets_to_eval:
        print(f"\n--- Subset: {sub_name} ---")
        pq_file = subset_config[sub_name]
        if not Path(pq_file).exists():
            print(f"[WARN] Parquet file not found: {pq_file}, skipping.")
            continue

        table = pq.read_table(pq_file)
        n_rows = table.num_rows
        n_eval = args.num_samples if getattr(args, "num_samples", None) else n_rows
        n_eval = min(n_eval, n_rows)

        for idx in tqdm(range(n_eval), desc=f"{mode}/{sub_name}"):
            row = table.slice(idx, 1).to_pydict()
            sample = {k: (v[0] if isinstance(v, list) and len(v) == 1 else v)
                      for k, v in row.items()}

            sample_id = sample.get("id", f"{sub_name}_{idx}")
            question = sample.get("question", "")
            options = sample.get("options", [])
            if isinstance(options, str):
                try:
                    options = json.loads(options)
                except Exception:
                    options = [options]
            answer_key = sample.get("answer", "").strip()
            video_url = sample.get("video", "")

            # Build question text
            if options and len(options) > 0:
                if isinstance(options[0], dict):
                    opt_text = "\n".join(
                        f"{o.get('choice', o.get('text', str(o)))}"
                        for o in options
                    )
                else:
                    opt_text = "\n".join(f"{chr(65+i)}. {o}" for i, o in enumerate(options))
                full_question = f"{question}\n{opt_text}"
            else:
                full_question = question

            # Load video
            video_path = find_local_video(sample_id, video_root, subset=sub_name)
            if video_path is None:
                video_path = video_url

            try:
                frames = load_video(video_path, num_frames=num_frames)
                images = [Image.fromarray(f).convert("RGB") for f in frames]
            except Exception as e:
                print(f"[WARN] Failed to load video {sample_id}: {e}")
                images = []

            messages = [{"role": "user", "content": [
                {"type": "video", "image": images} if images else {"type": "image", "image": images[0] if images else None},
                {"type": "text", "text": full_question},
            ]}]

            try:
                result = generator.generate(messages)
                response = result["response"]
            except Exception as e:
                response = f"[ERROR] {str(e)}"
                print(f"[ERROR] Generation failed for {sample_id}: {e}")

            # Extract prediction
            pred = response.strip()
            if answer_key.upper() in ["A", "B", "C", "D", "E"]:
                pred_key = pred[0].upper() if pred else ""
            else:
                pred_key = pred.split()[0] if pred else ""

            is_correct = int(pred_key == answer_key.upper())
            correct += is_correct
            total += 1
            per_subset_correct[sub_name] += is_correct
            per_subset_total[sub_name] += 1

            all_predictions.append({
                "id": sample_id,
                "subset": sub_name,
                "question": question,
                "ground_truth": answer_key,
                "prediction": pred,
                "correct": bool(is_correct),
            })

    # Save results
    accuracy = correct / total if total > 0 else 0.0
    per_subset_acc = {
        s: per_subset_correct[s] / per_subset_total[s]
        if per_subset_total[s] > 0 else 0.0
        for s in subsets_to_eval
    }

    results = {
        "mode": mode,
        "subset": subset,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "per_subset_accuracy": per_subset_acc,
        "predictions": all_predictions,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n[METRIC] Overall accuracy: {accuracy:.4f} ({correct}/{total})")
    for s, acc in per_subset_acc.items():
        c = per_subset_correct[s]
        t = per_subset_total[s]
        print(f"[METRIC] {s}: {acc:.4f} ({c}/{t})")

    return results
