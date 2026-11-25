"""FDA and question-attention evidence collected during a read-only prefill."""

import math
import torch
import torch.nn.functional as F
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    apply_multimodal_rotary_pos_emb,
    repeat_kv,
)


class Evidence:

    def __init__(self, model, inputs, question_ids):
        self.model = model
        self.handles = []
        self.early = None
        self.frames = []
        self.attn = []
        self.weights = None
        ids = inputs.input_ids[0]
        self.visual = ids.eq(model.config.video_token_id).nonzero().flatten()
        self.groups = int(inputs.video_grid_thw[0, 0])
        assert len(self.visual) % self.groups == 0
        seq = ids.tolist()
        matches = [
            i
            for i in range(len(seq) - len(question_ids) + 1)
            if seq[i : i + len(question_ids)] == question_ids
        ]
        assert len(matches) == 1, matches
        self.question = torch.arange(
            matches[0], matches[0] + len(question_ids), device=ids.device
        )
        assert int(self.question.min()) > int(self.visual.max())
        self.length = len(ids)
        self.frame18 = None
        self.text18 = None

    def attention(self, module, args, kw):
        h = kw["hidden_states"] if "hidden_states" in kw else args[0]
        if h.shape[1] != self.length:
            return
        h = h[:1]
        n = h.shape[1]
        q = module.q_proj(h).view(1, n, -1, module.head_dim).transpose(1, 2)
        k = module.k_proj(h).view(1, n, -1, module.head_dim).transpose(1, 2)
        cos, sin = kw["position_embeddings"]
        q, k = apply_multimodal_rotary_pos_emb(
            q, k, cos[:, :1], sin[:, :1], module.rope_scaling["mrope_section"]
        )
        k = repeat_kv(k, module.num_key_value_groups)
        scores = (
            q[:, :, self.question] @ k.transpose(-2, -1) / math.sqrt(module.head_dim)
        )
        mask = kw.get("attention_mask")
        if mask is not None:
            scores = scores + mask[:1, :, self.question, :n]
        else:
            scores = scores.masked_fill(
                torch.arange(n, device=h.device)[None, :] > self.question[:, None],
                -torch.inf,
            )
        prob = scores.float().softmax(-1)
        relevance = (
            prob[0, :, :, self.visual].mean((0, 1)).reshape(self.groups, -1).sum(-1)
        )
        self.attn.append(relevance)

    def middle(self, module, args, out):
        if out[0].shape[1] != self.length:
            return
        f = (
            out[0][0, self.visual]
            .float()
            .reshape(self.groups, -1, out[0].shape[-1])
            .mean(1)
        )
        self.frames.append(f)

    def capture18(self, module, args, out):
        self.early = out[0]
        if out[0].shape[1] != self.length:
            return
        h = self.model.model.norm(out[0][:1])[0].float()
        self.frame18 = h[self.visual].reshape(self.groups, -1, h.shape[-1]).mean(1)
        self.text18 = h[self.question]
        assert len(self.frames) == len(self.attn) == 9
        phi = torch.stack(
            [
                (F.normalize(f, dim=-1) @ F.normalize(f, dim=-1).T).var(
                    -1, unbiased=False
                )
                for f in self.frames
            ]
        ).mean(0)
        phi = F.avg_pool1d(
            F.pad(phi[None, None], (1, 1), mode="replicate"), 3, stride=1
        ).flatten()
        w = phi.softmax(0)
        a = torch.stack(self.attn).mean(0)
        mass = float(a.sum())
        a = a / a.sum().clamp_min(1e-12)
        self.weights = {
            "uniform": torch.ones_like(w) / len(w),
            "fda": w,
            "attention": a,
            "fused": 0.5 * w + 0.5 * a,
        }
        self.audit = {
            "question_tokens": len(self.question),
            "video_tokens": len(self.visual),
            "temporal_groups": self.groups,
            "attention_layers": list(range(8, 17)),
            "video_attention_mass": mass,
            "weights": {k: v.tolist() for k, v in self.weights.items()},
        }
        assert mass > 0 and all(
            (torch.isfinite(v).all() for v in self.weights.values())
        )
        self.frames.clear()
        self.attn.clear()

    def __enter__(self):
        for i in range(7, 16):
            self.handles.append(
                self.model.model.layers[i].self_attn.register_forward_pre_hook(
                    self.attention, with_kwargs=True
                )
            )
            self.handles.append(
                self.model.model.layers[i].register_forward_hook(self.middle)
            )
        self.handles.append(
            self.model.model.layers[17].register_forward_hook(self.capture18)
        )
        return self

    def __exit__(self, *args):
        for h in self.handles:
            h.remove()


class RowEvidence(Evidence):

    def __init__(self, model, inputs, question_ids, start, width):
        super().__init__(model, inputs, question_ids)
        self.start = start
        self.width = width

    def attention(self, module, args, kw):
        kw = dict(kw)
        h = kw["hidden_states"] if "hidden_states" in kw else args[0]
        kw["hidden_states"] = h[self.start : self.start + self.width]
        cos, sin = kw["position_embeddings"]
        kw["position_embeddings"] = (
            cos[:, self.start : self.start + self.width],
            sin[:, self.start : self.start + self.width],
        )
        if kw.get("attention_mask") is not None and kw["attention_mask"].shape[0] > 1:
            kw["attention_mask"] = kw["attention_mask"][
                self.start : self.start + self.width
            ]
        return super().attention(module, (), kw)

    def middle(self, module, args, out):
        return super().middle(
            module, args, (out[0][self.start : self.start + self.width],)
        )

    def capture18(self, module, args, out):
        return super().capture18(
            module, args, (out[0][self.start : self.start + self.width],)
        )


class PromptEvidence(RowEvidence):

    def capture22(self, module, args, out):
        if out[0].shape[1] != self.length:
            return
        x = self.model.model.norm(out[0][self.start : self.start + 1])[0].float()
        self.visual22 = x[self.visual]
        self.frame22 = x[self.visual].reshape(self.groups, -1, x.shape[-1]).mean(1)
        self.text22 = x[self.question]

    def __enter__(self):
        super().__enter__()
        self.handles.append(
            self.model.model.layers[21].register_forward_hook(self.capture22)
        )
        return self
