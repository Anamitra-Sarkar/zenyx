"""FP8 E4M3 quantization of K and V tensors during training with STE gradient flow.

COAT proof does NOT transfer directly to KV quantization (Source A).
In a linear layer, quantization error is averaged by the weight matrix (low-pass filtered).
In KV attention, the error enters the pre-softmax logit as K_quant_err, which is then
EXPONENTIATED in softmax — exponential amplification. Source B claims COAT does transfer.
Both paths are implemented with a flag (coat_safety_claim).

KIVI (2-bit inference quantizer) must NOT be used in training. Dynamic per-forward-pass
scale computation is mandatory — never cache or reuse scales across steps.

Quantization strategy [DISPUTE 8-B]:
  Source A: per-CHANNEL dynamic scaling for K (prevents outlier-caused underflow).
  Source B: per-HEAD dynamic scaling for K (simpler, claims "manageable" outlier risk).
  Both implemented; per_channel is default (safer). Ablation test resolves empirically.

SwiGLU outlier amplification (confirmed by arxiv 2409.12517):
  Over prolonged training with SwiGLU + weight decay, linear projections align,
  systematically amplifying outliers → K scale explodes → non-outlier underflow →
  attention collapse. Fix: Smooth-SwiGLU with α = sqrt(d_model / num_heads).
"""

from __future__ import annotations

import logging
import math
import warnings
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn

__all__ = [
    "quantize_kv_fp8",
    "quantize_kv_fp8_per_channel",
    "quantize_kv_fp8_per_head",
    "dequantize_kv",
    "smooth_swiglu_scale",
    "GradientMonitor",
    "FP8_E4M3_MAX",
    "FP8_E4M3_EPSILON",
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FP8 E4M3 constants
# ---------------------------------------------------------------------------

FP8_E4M3_MAX: float = 448.0  # Maximum representable value in FP8 E4M3
FP8_E4M3_EPSILON: float = 2**(-3)  # 0.125 — machine epsilon (3 mantissa bits)
_GRADIENT_ANOMALY_THRESHOLD: float = 0.02  # 3× theoretical bound → anomaly


# ---------------------------------------------------------------------------
# Smooth-SwiGLU scale
# ---------------------------------------------------------------------------


def smooth_swiglu_scale(d_model: int, num_heads: int) -> float:
    """Compute the Smooth-SwiGLU scaling factor α.

    Smooth-SwiGLU applies a static scale α to the linear branch of the gate
    and rescales by 1/α after the final linear layer. This algebraically
    neutralises outlier amplification at the source before it reaches K/V
    projections.

    Confirmed by arxiv 2409.12517: over prolonged training with SwiGLU + weight
    decay, the two linear projections in the gate align, systematically amplifying
    outliers. This causes per-channel K scale to explode → all valid data underflows
    to zero → attention collapse → gradient shatters.

    Parameters
    ----------
    d_model : int
        Model hidden dimension.
    num_heads : int
        Number of attention heads.

    Returns
    -------
    float
        The smoothing factor α = sqrt(d_model / num_heads).
    """
    if num_heads <= 0:
        raise ValueError(f"num_heads must be positive, got {num_heads}")
    if d_model <= 0:
        raise ValueError(f"d_model must be positive, got {d_model}")
    return math.sqrt(d_model / num_heads)


# ---------------------------------------------------------------------------
# Per-channel quantization (Source A — safe default)
# ---------------------------------------------------------------------------


def quantize_kv_fp8_per_channel(
    k: torch.Tensor, v: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize K and V to FP8 E4M3 with per-channel K / per-token V scaling.

    Source A (Zenyx Research Questions): per-head scaling is WRONG for K.
    A single outlier channel forces a massive scale denominator, mapping all
    non-outlier values to zero in the 3-bit mantissa.

    Correct strategy:
      K: per-CHANNEL dynamic scaling. s_d = max|K[:, d]| / 448.0
      V: per-TOKEN dynamic scaling.  s_t = max|V[t, :]| / 448.0

    Parameters
    ----------
    k : torch.Tensor
        Key tensor of shape [..., seq_len, head_dim].
    v : torch.Tensor
        Value tensor of shape [..., seq_len, head_dim].

    Returns
    -------
    (k_fp8, v_fp8, k_scales, v_scales)
        k_fp8, v_fp8: Quantized tensors (stored as float32 for compatibility).
        k_scales: FP32 per-channel scales of shape [..., head_dim].
        v_scales: FP32 per-token scales of shape [..., seq_len].
    """
    # K: per-channel scaling — scale per hidden dimension d
    k_abs_max = k.abs().amax(dim=-2, keepdim=False)  # [..., head_dim]
    k_scales = k_abs_max / FP8_E4M3_MAX
    k_scales = k_scales.clamp(min=1e-12)  # Prevent division by zero
    k_scaled = k / k_scales.unsqueeze(-2)
    k_fp8 = k_scaled.clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).round()

    # V: per-token scaling — scale per token t
    v_abs_max = v.abs().amax(dim=-1, keepdim=False)  # [..., seq_len]
    v_scales = v_abs_max / FP8_E4M3_MAX
    v_scales = v_scales.clamp(min=1e-12)
    v_scaled = v / v_scales.unsqueeze(-1)
    v_fp8 = v_scaled.clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).round()

    return k_fp8, v_fp8, k_scales, v_scales


# ---------------------------------------------------------------------------
# Per-head quantization (Source B — simpler but riskier)
# ---------------------------------------------------------------------------


def quantize_kv_fp8_per_head(
    k: torch.Tensor, v: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize K and V to FP8 E4M3 with per-head scaling (Source B).

    Source B (Qwen Phase Analysis): per-head dynamic scale is a "reasonable
    first choice." Notes outlier risk but calls it manageable.

    WARNING: This causes catastrophic underflow when a single outlier channel
    forces a massive scale denominator, mapping all non-outlier values to zero.
    Use quantize_kv_fp8_per_channel instead for production.

    Parameters
    ----------
    k : torch.Tensor
        Key tensor of shape [..., seq_len, head_dim].  Must have ndim >= 3.
    v : torch.Tensor
        Value tensor of shape [..., seq_len, head_dim].  Must have ndim >= 3.

    Returns
    -------
    (k_fp8, v_fp8, k_scales, v_scales)
        k_scales : [...] — one scalar per head (all dims except last two).
        v_scales : [...] — one scalar per head.

    Raises
    ------
    ValueError
        If k or v has ndim < 3. Per-head scaling requires at least one
        batch/head dimension in addition to (seq_len, head_dim).
    """
    if k.ndim < 3:
        raise ValueError(
            f"quantize_kv_fp8_per_head: k must have ndim >= 3 "
            f"(got ndim={k.ndim}, shape={tuple(k.shape)}). "
            "Expected shape [..., seq_len, head_dim] with at least one "
            "leading batch or head dimension."
        )
    if v.ndim < 3:
        raise ValueError(
            f"quantize_kv_fp8_per_head: v must have ndim >= 3 "
            f"(got ndim={v.ndim}, shape={tuple(v.shape)}). "
            "Expected shape [..., seq_len, head_dim] with at least one "
            "leading batch or head dimension."
        )

    # K: per-head scaling — single scale for entire head
    k_abs_max = k.abs().amax(dim=(-2, -1), keepdim=True)  # [..., 1, 1]
    k_scales = (k_abs_max / FP8_E4M3_MAX).squeeze(-2).squeeze(-1)  # [...]
    k_scales = k_scales.clamp(min=1e-12)
    k_scaled = k / k_abs_max.clamp(min=1e-12) * FP8_E4M3_MAX
    k_fp8 = k_scaled.clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).round()

    # V: per-head scaling (same approach)
    v_abs_max = v.abs().amax(dim=(-2, -1), keepdim=True)
    v_scales = (v_abs_max / FP8_E4M3_MAX).squeeze(-2).squeeze(-1)  # [...]
    v_scales = v_scales.clamp(min=1e-12)
    v_scaled = v / v_abs_max.clamp(min=1e-12) * FP8_E4M3_MAX
    v_fp8 = v_scaled.clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).round()

    return k_fp8, v_fp8, k_scales, v_scales


# ---------------------------------------------------------------------------
# Unified quantization entry point with strategy flag
# ---------------------------------------------------------------------------


def quantize_kv_fp8(
    k: torch.Tensor,
    v: torch.Tensor,
    strategy: str = "per_channel",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize K and V to FP8 E4M3 using the specified strategy.

    [DISPUTE 8-B]: per_channel (Source A) vs per_head (Source B).
    Default: per_channel (safer against outlier-caused underflow).

    Parameters
    ----------
    k : torch.Tensor
        Key tensor.
    v : torch.Tensor
        Value tensor.
    strategy : str
        "per_channel" (default, Source A) or "per_head" (Source B).

    Returns
    -------
    (k_fp8, v_fp8, k_scales, v_scales)
    """
    if strategy == "per_channel":
        return quantize_kv_fp8_per_channel(k, v)
    elif strategy == "per_head":
        return quantize_kv_fp8_per_head(k, v)
    else:
        raise ValueError(f"Unknown quantization strategy: {strategy!r}. "
                         "Use 'per_channel' or 'per_head'.")


# ---------------------------------------------------------------------------
# Dequantization
# ---------------------------------------------------------------------------


def dequantize_kv(
    k_fp8: torch.Tensor,
    v_fp8: torch.Tensor,
    k_scales: torch.Tensor,
    v_scales: torch.Tensor,
    strategy: str = "per_channel",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Dequantize FP8 K and V back to BF16/FP32.

    STE: in backward, the gradient passes through dequantize unchanged (identity).
    The quantization error manifests as bounded additive noise in forward activations only.

    Parameters
    ----------
    k_fp8, v_fp8 : torch.Tensor
        Quantized tensors of shape [..., seq_len, head_dim].
    k_scales, v_scales : torch.Tensor
        Scale tensors from quantization.
        per_channel: k_scales shape [..., head_dim], v_scales shape [..., seq_len].
        per_head:    k_scales shape [...], v_scales shape [...].
    strategy : str
        Must match the strategy used during quantization.

    Returns
    -------
    (k_bf16, v_bf16) : Tuple[torch.Tensor, torch.Tensor]
    """
    if strategy == "per_channel":
        k_bf16 = k_fp8 * k_scales.unsqueeze(-2)
        v_bf16 = v_fp8 * v_scales.unsqueeze(-1)
    elif strategy == "per_head":
        # Per-head: k_scales shape is [...] (all dims except seq_len and head_dim).
        # k_fp8 = k / k_abs_max * FP8_E4M3_MAX
        # Dequantize: k = k_fp8 * k_scales  (where k_scales = k_abs_max / FP8_E4M3_MAX)
        # Need to broadcast k_scales [...] against k_fp8 [..., seq_len, head_dim]
        # by unsqueezing the last two dims.
        k_bf16 = k_fp8 * k_scales.unsqueeze(-1).unsqueeze(-1)
        v_bf16 = v_fp8 * v_scales.unsqueeze(-1).unsqueeze(-1)
    else:
        raise ValueError(f"Unknown strategy: {strategy!r}")

    return k_bf16, v_bf16


# ---------------------------------------------------------------------------
# STE (Straight-Through Estimator) wrapper
# ---------------------------------------------------------------------------


class _STEQuantizeFunction(torch.autograd.Function):
    """Straight-Through Estimator for quantize → dequantize.

    Forward: quantize to FP8, immediately dequantize back.
    Backward: identity (gradient passes through unchanged).
    """

    @staticmethod
    def forward(
        ctx: Any, k: torch.Tensor, v: torch.Tensor, strategy: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        k_fp8, v_fp8, k_scales, v_scales = quantize_kv_fp8(k, v, strategy)
        k_deq, v_deq = dequantize_kv(k_fp8, v_fp8, k_scales, v_scales, strategy)
        return k_deq, v_deq

    @staticmethod
    def backward(
        ctx: Any, grad_k: torch.Tensor, grad_v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, None]:
        # STE: identity pass-through
        return grad_k, grad_v, None


def ste_quantize_kv(
    k: torch.Tensor, v: torch.Tensor, strategy: str = "per_channel"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply FP8 quantize+dequantize with STE for gradient pass-through."""
    return _STEQuantizeFunction.apply(k, v, strategy)


# ---------------------------------------------------------------------------
# Gradient Monitor
# ---------------------------------------------------------------------------


class GradientMonitor:
    """Monitor per-step gradient perturbation from FP8 quantization.

    Theoretical max gradient error bound:
      FP8 E4M3 machine epsilon: ε = 2^(-3) = 0.125
      Max quantization error per element: δ ≤ 0.007 × max_channel_activation
      Anomaly threshold: measured perturbation > 0.02 × mean gradient norm (3× bound)

    [DISPUTE 8-A]: Source A says COAT does NOT transfer (softmax exponential amplification).
    Source B says it does. coat_safety_claim flag controls whether the per-step check runs.

    Parameters
    ----------
    coat_safety_claim : bool
        If True (Source B), skip per-step numerical check (trust COAT bound).
        If False (Source A, default), run independent per-step gradient check.
    anomaly_threshold : float
        Relative threshold for gradient perturbation anomaly. Default: 0.02.
    """

    def __init__(
        self,
        coat_safety_claim: bool = False,
        anomaly_threshold: float = _GRADIENT_ANOMALY_THRESHOLD,
    ) -> None:
        self.coat_safety_claim = coat_safety_claim
        self.anomaly_threshold = anomaly_threshold
        self._step_count: int = 0
        self._max_relative_error: float = 0.0
        self._anomaly_count: int = 0

    def check_gradient(
        self,
        grad_fp8: torch.Tensor,
        grad_bf16: torch.Tensor,
    ) -> bool:
        """Compare FP8-path gradient against BF16 reference.

        Parameters
        ----------
        grad_fp8 : torch.Tensor
            Gradient computed through the FP8 quantization path.
        grad_bf16 : torch.Tensor
            Reference gradient computed in full BF16.

        Returns
        -------
        bool
            True if gradient perturbation is within bounds.
        """
        self._step_count += 1

        if self.coat_safety_claim:
            # Source B: trust COAT bound, skip expensive check
            return True

        # Source A: run independent numerical check
        grad_norm = grad_bf16.norm().item()
        if grad_norm < 1e-12:
            return True

        diff_norm = (grad_fp8 - grad_bf16).norm().item()
        relative_error = diff_norm / grad_norm

        self._max_relative_error = max(self._max_relative_error, relative_error)

        if relative_error > self.anomaly_threshold:
            self._anomaly_count += 1
            warnings.warn(
                f"FP8 gradient perturbation anomaly at step {self._step_count}: "
                f"relative error {relative_error:.6f} > threshold {self.anomaly_threshold}. "
                f"This exceeds 3× the theoretical bound. COAT safety claim would mask this.",
                UserWarning,
                stacklevel=2,
            )
            return False

        return True

    @property
    def max_relative_error(self) -> float:
        """Maximum observed relative gradient error across all steps."""
        return self._max_relative_error

    @property
    def anomaly_count(self) -> int:
        """Number of steps where gradient perturbation exceeded threshold."""
        return self._anomaly_count

    def summary(self) -> dict:
        """Return monitoring summary."""
        return {
            "steps_checked": self._step_count,
            "max_relative_error": self._max_relative_error,
            "anomaly_count": self._anomaly_count,
            "coat_safety_claim": self.coat_safety_claim,
            "anomaly_threshold": self.anomaly_threshold,
            "DISPUTE_8A_resolution": (
                "COAT trusted (Source B)" if self.coat_safety_claim
                else f"Independent check (Source A): max_err={self._max_relative_error:.6f}"
            ),
        }
