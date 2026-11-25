"""Rank-8 evidence projection with video-key attention enhancement."""

from types import SimpleNamespace
import torch
from .attention import wrap_attention
from .projection import decompose, precise_projection, svd_basis


class Intervention:
    def __init__(self, model, evidence, config):
        if config["spatial_rank"] != 8:
            raise ValueError("This implementation requires spatial rank 8.")
        self.model = model
        self.evidence = evidence
        self.config = config
        self.parts = [SimpleNamespace(records=[]) for _ in evidence]
        self.bases = {}
        self.calls = {}
        self.pending = []

    def build_bases(self, prompt):
        config = self.config
        frame, text = prompt.frame22, prompt.text22
        weights = prompt.weights[config["weight"]]
        temporal = svd_basis(
            (frame[1:] - frame[:-1]) * weights[1:, None], config["rank"]
        )
        visual = prompt.visual22
        question = text.mean(0)
        relevance = (
            torch.nn.functional.normalize(visual, dim=-1)
            @ torch.nn.functional.normalize(question, dim=0)
            / 1.5
        ).softmax(0)
        relevance = relevance * weights.repeat_interleave(len(visual) // prompt.groups)
        relevance = relevance / relevance.sum()
        centered = visual - (relevance[:, None] * visual).sum(0, keepdim=True)
        complement = centered - (centered @ temporal) @ temporal.T
        spatial = svd_basis(complement * relevance.sqrt()[:, None], 8)
        spatial = svd_basis(spatial.T - (spatial.T @ temporal) @ temporal.T, 8)
        temporal = torch.linalg.qr(
            torch.cat([temporal, spatial], dim=1), mode="reduced"
        ).Q
        prior = svd_basis(text - (text @ temporal) @ temporal.T, config["prior_rank"])
        prior = svd_basis(
            prior.T - (prior.T @ temporal) @ temporal.T, config["prior_rank"]
        )
        orthogonality = float((temporal.T @ prior).abs().max())
        assert orthogonality < 1e-4
        prompt.audit["subspaces"] = dict(
            visual_rank=temporal.shape[1],
            prior_rank=prior.shape[1],
            cross_orthogonality_max=orthogonality,
        )
        return temporal, prior

    @precise_projection
    def final(self, module, args, output):
        hidden = output.last_hidden_state
        result, counts, checks = [], [], []
        config = self.config
        for index, prompt in enumerate(self.evidence):
            x = hidden[index].float()
            if index not in self.bases:
                self.bases[index] = self.build_bases(prompt)
            temporal, prior = self.bases[index]
            hT, hP, hR = decompose(x, temporal, prior)
            reconstruction = (
                (hT + hP + hR - x).norm(dim=-1) / (x.norm(dim=-1) + 1e-8)
            ).max()
            energy = x.square().sum(-1).clamp_min(1e-8)
            fractions = torch.stack(
                [part.square().sum(-1) / energy for part in (hT, hP, hR)], -1
            )
            checks.extend(
                [reconstruction < 1e-5, (fractions.sum(-1) - 1).abs().max() < 1e-4]
            )
            edited = (
                x
                + (config["gT"] - 1) * hT
                + (config["gP"] - 1) * hP
                + (config["gR"] - 1) * hR
            ).to(hidden.dtype)
            counts.append(((edited.float() - x).abs().amax(-1) > 0).sum())
            result.append(edited)
        output.last_hidden_state = torch.stack(result)
        checks.append(torch.isfinite(output.last_hidden_state).all())
        self.pending.append(
            (list(hidden.shape), torch.stack(counts), torch.stack(checks).all())
        )
        return output

    def __enter__(self):
        self.handle = self.model.model.register_forward_hook(self.final)
        self.originals = []
        for layer in range(
            self.config["attention_start"], self.config["attention_end"] + 1
        ):
            module = self.model.model.layers[layer - 1].self_attn
            self.originals.append((module, module.forward))
            module.forward = wrap_attention(
                module.forward, layer, self.evidence, self.config, self.calls
            )
        return self

    def __exit__(self, *exc):
        for module, original in self.originals:
            module.forward = original
        self.handle.remove()
        if exc[0] is not None:
            return
        for shape, counts, check in self.pending:
            assert bool(check), "Projection reconstruction/finite check failed"
            for index, changed in enumerate(counts.cpu().tolist()):
                assert changed == shape[1], (changed, shape)
                self.parts[index].records.append(
                    dict(shape=[1, shape[1], shape[2]], changed=[changed])
                )
        assert all(count == len(self.pending) for count in self.calls.values())
        for prompt in self.evidence:
            prompt.audit["attention_layer_forward_counts"] = dict(self.calls)
        self.pending.clear()
