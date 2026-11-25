"""Video-key attention enhancement for the rank-8 method."""

import functools
import math
import torch
import torch.nn.functional as F


def wrap_attention(original, layer, evidence, config, calls):
    @functools.wraps(original)
    def forward(*args, **kwargs):
        sdpa = F.scaled_dot_product_attention

        def edited(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, **options):
            batch, _, length, _ = q.shape
            keys = k.shape[-2]
            if attn_mask is None:
                mask = torch.zeros(
                    (batch, 1, length, keys), device=q.device, dtype=q.dtype
                )
            else:
                mask = attn_mask.expand(batch, -1, -1, -1).clone()
                if mask.dtype == torch.bool:
                    mask = torch.zeros_like(mask, dtype=q.dtype).masked_fill(
                        ~mask, -torch.inf
                    )
            if is_causal:
                causal = torch.arange(keys, device=q.device)[None, :] > (
                    torch.arange(length, device=q.device)[:, None] + keys - length
                )
                mask.masked_fill_(causal[None, None], -torch.inf)
            for index, prompt in enumerate(evidence):
                mask[index, :, :, prompt.visual] += math.log(config["video_gain"])
            return sdpa(
                q, k, v, attn_mask=mask, dropout_p=dropout_p, is_causal=False, **options
            )

        F.scaled_dot_product_attention = edited
        try:
            return original(*args, **kwargs)
        finally:
            F.scaled_dot_product_attention = sdpa
            calls[layer] = calls.get(layer, 0) + 1

    return forward
