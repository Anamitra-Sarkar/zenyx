"""FP8-ish KV/weight storage with BF16 compute restoration."""

from __future__ import annotations

import torch


def quantize_to_fp8_storage(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Store tensor as int8 + scale to emulate FP8 storage budget."""
    scale = tensor.abs().amax().clamp_min(1e-8) / 127.0
    q = torch.clamp((tensor / scale).round(), -127, 127).to(torch.int8)
    return q, scale


def dequantize_for_bf16_compute(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return (q.float() * scale).to(torch.bfloat16)


__all__ = ["quantize_to_fp8_storage", "dequantize_for_bf16_compute"]
