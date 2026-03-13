"""Tests for Phase 8 — FP8 KV quantization with dispute resolution."""

from __future__ import annotations

import logging
import warnings

import pytest
import torch

from zenyx.train.fp8_kv import (
    FP8_E4M3_EPSILON,
    FP8_E4M3_MAX,
    GradientMonitor,
    dequantize_kv,
    quantize_kv_fp8,
    quantize_kv_fp8_per_channel,
    quantize_kv_fp8_per_head,
    smooth_swiglu_scale,
    ste_quantize_kv,
)


# ---------------------------------------------------------------------------
# DISPUTE 8-B ablation: per-channel vs per-head with 100× outlier
# ---------------------------------------------------------------------------


class TestDisputeB_OutlierAblation:
    """Empirically resolve DISPUTE 8-B: per-channel vs per-head for K.

    Create synthetic K tensor with a 100× outlier in one channel.
    - per_channel should preserve non-outlier values (no underflow).
    - per_head should cause >50% underflow in non-outlier channels.
    """

    def _make_outlier_k(self) -> torch.Tensor:
        """Create K tensor [1, 32, 64] with 100× outlier in channel 0."""
        k = torch.randn(1, 32, 64) * 0.1  # Normal channels: ~0.1 magnitude
        k[:, :, 0] = 100.0 * torch.randn(1, 32)  # Channel 0: 100× outlier
        return k

    def test_per_channel_preserves_non_outlier(self) -> None:
        """Per-channel scaling: non-outlier channels should survive quantization."""
        k = self._make_outlier_k()
        v = torch.randn(1, 32, 64) * 0.1

        k_fp8, v_fp8, k_scales, v_scales = quantize_kv_fp8_per_channel(k, v)
        k_deq, _ = dequantize_kv(k_fp8, v_fp8, k_scales, v_scales, "per_channel")

        # Non-outlier channels (1..63) should be close to original
        non_outlier_original = k[:, :, 1:]
        non_outlier_deq = k_deq[:, :, 1:]

        # Count how many non-outlier values are approximately zero (underflow)
        underflow_mask = non_outlier_deq.abs() < 1e-6
        underflow_fraction = underflow_mask.float().mean().item()

        # Per-channel: underflow should be minimal (< 10%)
        assert underflow_fraction < 0.10, (
            f"Per-channel underflow {underflow_fraction:.2%} — should be < 10%"
        )

    def test_per_head_causes_underflow(self) -> None:
        """Per-head scaling: outlier dominates scale → non-outlier channels underflow."""
        k = self._make_outlier_k()
        v = torch.randn(1, 32, 64) * 0.1

        k_fp8, v_fp8, k_scales, v_scales = quantize_kv_fp8_per_head(k, v)
        k_deq, _ = dequantize_kv(k_fp8, v_fp8, k_scales, v_scales, "per_head")

        # Non-outlier channels: many should underflow to zero
        non_outlier_original = k[:, :, 1:]
        non_outlier_deq = k_deq[:, :, 1:]

        underflow_mask = non_outlier_deq.abs() < 1e-6
        underflow_fraction = underflow_mask.float().mean().item()

        # Per-head: underflow should be >50% (catastrophic)
        assert underflow_fraction > 0.50, (
            f"Per-head underflow {underflow_fraction:.2%} — expected > 50%. "
            "DISPUTE 8-B: per-head may be safer than expected?"
        )

        # Log resolution
        logging.info(
            "DISPUTE_8B_RESOLVED: per_head underflow=%.2f%%, per_channel preserves data. "
            "Source A (per-channel) wins.",
            underflow_fraction * 100,
        )

    def test_unified_api_default_is_per_channel(self) -> None:
        """quantize_kv_fp8 default strategy should be per_channel."""
        k = torch.randn(1, 32, 64)
        v = torch.randn(1, 32, 64)
        # Should not raise
        k_fp8, v_fp8, k_s, v_s = quantize_kv_fp8(k, v)
        assert k_fp8.shape == k.shape
        assert v_fp8.shape == v.shape

    def test_unified_api_per_head(self) -> None:
        k = torch.randn(1, 32, 64)
        v = torch.randn(1, 32, 64)
        k_fp8, v_fp8, k_s, v_s = quantize_kv_fp8(k, v, strategy="per_head")
        assert k_fp8.shape == k.shape

    def test_invalid_strategy_raises(self) -> None:
        k = torch.randn(1, 32, 64)
        v = torch.randn(1, 32, 64)
        with pytest.raises(ValueError, match="Unknown quantization strategy"):
            quantize_kv_fp8(k, v, strategy="garbage")


# ---------------------------------------------------------------------------
# STE gradient tests
# ---------------------------------------------------------------------------


class TestSTEGradient:
    """Verify STE passes gradient through cleanly."""

    def test_ste_gradient_identity(self) -> None:
        """Gradient through STE should be approximately identity."""
        k = torch.randn(2, 16, 64, requires_grad=True)
        v = torch.randn(2, 16, 64, requires_grad=True)

        k_deq, v_deq = ste_quantize_kv(k, v)

        # Backward
        loss = k_deq.sum() + v_deq.sum()
        loss.backward()

        # STE: gradients should be exactly 1.0 (identity pass-through)
        assert k.grad is not None
        assert v.grad is not None
        assert torch.allclose(k.grad, torch.ones_like(k.grad))
        assert torch.allclose(v.grad, torch.ones_like(v.grad))

    def test_ste_quantize_dequantize_roundtrip(self) -> None:
        """STE output should be close to original (small quantization noise)."""
        k = torch.randn(2, 16, 64) * 0.5
        v = torch.randn(2, 16, 64) * 0.5

        k.requires_grad_(True)
        v.requires_grad_(True)

        k_deq, v_deq = ste_quantize_kv(k, v)

        # Quantization noise should be bounded
        k_diff = (k_deq - k).abs().max().item()
        v_diff = (v_deq - v).abs().max().item()

        # Noise should be small relative to data magnitude
        assert k_diff < 1.0, f"K quantization error too large: {k_diff}"
        assert v_diff < 1.0, f"V quantization error too large: {v_diff}"


# ---------------------------------------------------------------------------
# SwiGLU smoothing
# ---------------------------------------------------------------------------


class TestSmoothSwiGLU:
    """Verify Smooth-SwiGLU scaling factor."""

    def test_scale_formula(self) -> None:
        """α = sqrt(d_model / num_heads)"""
        alpha = smooth_swiglu_scale(d_model=4096, num_heads=32)
        expected = (4096 / 32) ** 0.5  # sqrt(128) ≈ 11.31
        assert abs(alpha - expected) < 1e-6

    def test_scale_reduces_outlier(self) -> None:
        """Applying α to a tensor with outliers should reduce outlier magnitude."""
        alpha = smooth_swiglu_scale(d_model=4096, num_heads=32)

        # Simulate: linear branch output with an outlier
        x = torch.randn(32, 64) * 0.1
        x[0, 0] = 100.0  # Outlier

        # Smooth: divide by α
        x_smooth = x / alpha

        # Outlier magnitude should be reduced by factor α
        assert x_smooth[0, 0].abs().item() < x[0, 0].abs().item()
        assert abs(x_smooth[0, 0].item() - 100.0 / alpha) < 1e-4

    def test_invalid_inputs(self) -> None:
        with pytest.raises(ValueError):
            smooth_swiglu_scale(d_model=0, num_heads=32)
        with pytest.raises(ValueError):
            smooth_swiglu_scale(d_model=4096, num_heads=0)

    def test_various_configs(self) -> None:
        """Test with multiple model configurations."""
        for d, h in [(512, 8), (1024, 16), (2048, 32), (4096, 64)]:
            alpha = smooth_swiglu_scale(d, h)
            assert alpha > 0
            assert abs(alpha - (d / h) ** 0.5) < 1e-6


# ---------------------------------------------------------------------------
# GradientMonitor
# ---------------------------------------------------------------------------


class TestGradientMonitor:
    """Test gradient monitoring with DISPUTE 8-A flag."""

    def test_coat_safety_skips_check(self) -> None:
        """When coat_safety_claim=True, check always returns True."""
        monitor = GradientMonitor(coat_safety_claim=True)
        grad_fp8 = torch.randn(10)
        grad_bf16 = torch.randn(10) * 100  # Wildly different
        assert monitor.check_gradient(grad_fp8, grad_bf16) is True

    def test_independent_check_catches_anomaly(self) -> None:
        """When coat_safety_claim=False, large perturbation triggers warning."""
        monitor = GradientMonitor(coat_safety_claim=False, anomaly_threshold=0.02)

        grad_bf16 = torch.ones(100)
        grad_fp8 = torch.ones(100) * 1.5  # 50% error — way over threshold

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = monitor.check_gradient(grad_fp8, grad_bf16)
            assert result is False
            assert len(w) >= 1
            assert "anomaly" in str(w[0].message).lower()

    def test_independent_check_passes_small_error(self) -> None:
        """Small perturbation should pass the check."""
        monitor = GradientMonitor(coat_safety_claim=False, anomaly_threshold=0.02)

        grad_bf16 = torch.ones(100) * 10.0
        grad_fp8 = grad_bf16 + torch.randn(100) * 0.001  # Tiny noise

        result = monitor.check_gradient(grad_fp8, grad_bf16)
        assert result is True

    def test_summary(self) -> None:
        monitor = GradientMonitor(coat_safety_claim=False)
        grad = torch.ones(10)
        monitor.check_gradient(grad, grad)

        summary = monitor.summary()
        assert summary["steps_checked"] == 1
        assert summary["coat_safety_claim"] is False
        assert "DISPUTE_8A_resolution" in summary

    def test_coat_empirical_validation_100_steps(self) -> None:
        """Run 100 steps with coat_safety_claim=False.

        Assert max relative gradient error < 0.02 (COAT-equivalent bound).
        If this passes, COAT is empirically validated for this implementation.
        """
        monitor = GradientMonitor(coat_safety_claim=False, anomaly_threshold=0.02)

        for step in range(100):
            # Simulate realistic K/V quantization gradient difference
            k = torch.randn(4, 8, 64) * 0.5
            v = torch.randn(4, 8, 64) * 0.5

            k.requires_grad_(True)
            v.requires_grad_(True)

            # BF16 path
            loss_bf16 = (k * v).sum()
            loss_bf16.backward()
            grad_k_bf16 = k.grad.clone()

            k.grad = None
            v.grad = None

            # FP8 path (STE)
            k_deq, v_deq = ste_quantize_kv(k, v)
            loss_fp8 = (k_deq * v_deq).sum()
            loss_fp8.backward()
            grad_k_fp8 = k.grad.clone()

            monitor.check_gradient(grad_k_fp8, grad_k_bf16)

        # If max error < 0.02, COAT is empirically validated
        assert monitor.max_relative_error < 0.02 or monitor.anomaly_count == 0, (
            f"COAT empirical validation: max_error={monitor.max_relative_error:.6f}, "
            f"anomalies={monitor.anomaly_count}. "
            "Source A (independent check needed) may be correct."
        )
        logging.info(
            "DISPUTE 8-A empirical result: max_relative_error=%.6f over 100 steps. "
            "COAT bound %s.",
            monitor.max_relative_error,
            "VALIDATED" if monitor.max_relative_error < 0.02 else "VIOLATED",
        )


# ---------------------------------------------------------------------------
# FP8 constants
# ---------------------------------------------------------------------------


class TestFP8Constants:
    def test_epsilon(self) -> None:
        assert FP8_E4M3_EPSILON == 0.125

    def test_max(self) -> None:
        assert FP8_E4M3_MAX == 448.0
