"""Zenyx bench — Memory budget calculator.

Computes a detailed memory budget for a given model configuration and
produces a beautifully formatted table with box-drawing characters.

Hardware presets cover H100, A100, H200, TPU_v5e, and RTX_4090.

Complexity
----------
``memory_budget()`` is O(1) — pure arithmetic on scalar inputs.
``BudgetReport.__str__()`` is O(1) — fixed-size string formatting.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional


# ── Hardware presets ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class HardwarePreset:
    """Immutable hardware specification for budget estimation.

    Attributes
    ----------
    name : str
        Short display name.
    description : str
        One-line description including memory technology.
    vram_gb : float
        T0 (VRAM / HBM) capacity in GB.
    mem_bandwidth_tb_s : float
        Memory bandwidth in TB/s.
    compute_tflops_fp16 : float
        Peak FP16 throughput in TFLOPS.
    nvlink_bw_gb_s : float
        NVLink / ICI bandwidth in GB/s (0 = none).
    nvme_bw_gb_s : float
        Typical NVMe read bandwidth in GB/s.
    """

    name: str
    description: str
    vram_gb: float
    mem_bandwidth_tb_s: float
    compute_tflops_fp16: float
    nvlink_bw_gb_s: float
    nvme_bw_gb_s: float = 7.0

    def __repr__(self) -> str:
        return (
            f"HardwarePreset({self.name!r}, {self.vram_gb:.0f}GB, "
            f"{self.compute_tflops_fp16:.0f} TFLOPS)"
        )


HARDWARE_PRESETS: Dict[str, HardwarePreset] = {
    "H100": HardwarePreset(
        name="H100",
        description="80GB HBM3",
        vram_gb=80.0,
        mem_bandwidth_tb_s=3.35,
        compute_tflops_fp16=989.0,
        nvlink_bw_gb_s=900.0,
    ),
    "A100": HardwarePreset(
        name="A100",
        description="80GB HBM2e",
        vram_gb=80.0,
        mem_bandwidth_tb_s=2.0,
        compute_tflops_fp16=312.0,
        nvlink_bw_gb_s=600.0,
    ),
    "H200": HardwarePreset(
        name="H200",
        description="141GB HBM3e",
        vram_gb=141.0,
        mem_bandwidth_tb_s=4.8,
        compute_tflops_fp16=989.0,
        nvlink_bw_gb_s=900.0,
    ),
    "TPU_v5e": HardwarePreset(
        name="TPU_v5e",
        description="16GB HBM",
        vram_gb=16.0,
        mem_bandwidth_tb_s=1.6,
        compute_tflops_fp16=197.0,
        nvlink_bw_gb_s=1600.0,  # ICI bidirectional
    ),
    "RTX_4090": HardwarePreset(
        name="RTX_4090",
        description="24GB GDDR6X",
        vram_gb=24.0,
        mem_bandwidth_tb_s=1.0,
        compute_tflops_fp16=330.0,
        nvlink_bw_gb_s=0.0,
    ),
}


# ── Budget report ────────────────────────────────────────────────────────


@dataclass
class BudgetReport:
    """Complete memory budget report.

    Attributes
    ----------
    model_params : float
        Number of model parameters.
    vocab_size : int
        Vocabulary size.
    context_len : int
        Maximum sequence length.
    hardware_name : str
        Hardware preset name.
    weights_gb : float
        Model weights size in GB.
    activations_gb : float
        Activation memory in GB.
    kv_cache_gb : float
        KV cache size in GB.
    optimizer_gb : float
        Optimizer state size in GB.
    peak_t0_gb : float
        Peak T0 (VRAM) usage in GB.
    peak_t1_gb : float
        Peak T1 (CPU) usage in GB.
    peak_t2_gb : float
        Peak T2 (NVMe) usage in GB.
    t0_capacity_gb : float
        T0 capacity in GB.
    estimated_load_time_s : float
        Estimated model load time in seconds.
    estimated_tokens_per_sec : float
        Estimated throughput in tokens/sec.
    oom_free : bool
        Whether the configuration fits without OOM.
    feasibility_margin : float
        Ratio of remaining capacity to total (higher = more headroom).
    """

    model_params: float
    vocab_size: int
    context_len: int
    hardware_name: str
    weights_gb: float
    activations_gb: float
    kv_cache_gb: float
    optimizer_gb: float
    peak_t0_gb: float
    peak_t1_gb: float
    peak_t2_gb: float
    t0_capacity_gb: float
    estimated_load_time_s: float
    estimated_tokens_per_sec: float
    oom_free: bool
    feasibility_margin: float

    def __repr__(self) -> str:
        params_str = _format_params(self.model_params)
        status = "OOM-free" if self.oom_free else "OOM-RISK"
        return (
            f"BudgetReport({params_str}, {self.hardware_name}, "
            f"peak_T0={self.peak_t0_gb:.2f}GB/{self.t0_capacity_gb:.0f}GB, "
            f"status={status}, margin={self.feasibility_margin:.2f}x)"
        )

    def __str__(self) -> str:
        """Produce the beautifully formatted memory budget table.

        Time: O(1).  Space: O(1).
        """
        W = 44  # inner width (between the outer ║ characters)

        params_str = _format_params(self.model_params)
        hw_preset = HARDWARE_PRESETS.get(self.hardware_name)
        hw_desc = f"{self.hardware_name} ({hw_preset.description})" if hw_preset else self.hardware_name

        oom_icon = "\u2705" if self.oom_free else "\u274c"
        tok_str = f"{self.estimated_tokens_per_sec:,.0f}"
        load_str = f"{self.estimated_load_time_s:.1f}"

        lines: list[str] = []

        # ── Top border ───────────────────────────────────────────────────
        lines.append(f"\u2554{'═' * W}\u2557")

        # ── Title ────────────────────────────────────────────────────────
        title = "Zenyx Memory Budget Report"
        lines.append(f"\u2551{title:^{W}}\u2551")

        # ── Header separator ─────────────────────────────────────────────
        lines.append(f"\u2560{'═' * W}\u2563")

        # ── Model info ───────────────────────────────────────────────────
        lines.append(_row(f" Model: {params_str} params", W))
        lines.append(_row(
            f" Vocab: {self.vocab_size:,} | Context: {self.context_len:,}",
            W,
        ))
        lines.append(_row(f" Hardware: {hw_desc}", W))

        # ── Component table header ───────────────────────────────────────
        lines.append(f"\u2560{'═' * W}\u2563")
        lines.append(_row(" Component          \u2502 Size (GB)", W))
        lines.append(_row(" \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2502\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500", W))

        # ── Component rows ───────────────────────────────────────────────
        lines.append(_row(f" Weights            \u2502 {self.weights_gb:>8.2f}", W))
        lines.append(_row(f" Activations        \u2502 {self.activations_gb:>8.2f}", W))
        lines.append(_row(f" KV Cache           \u2502 {self.kv_cache_gb:>8.2f}", W))
        lines.append(_row(f" Optimizer States   \u2502 {self.optimizer_gb:>8.2f}", W))

        # ── Tier table header ────────────────────────────────────────────
        lines.append(f"\u2560{'═' * W}\u2563")
        lines.append(_row(" Tier    \u2502 Peak (GB) \u2502 Capacity", W))
        lines.append(_row(" \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2502\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2502\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500", W))

        # ── Tier rows ────────────────────────────────────────────────────
        t0_cap_str = f"{self.t0_capacity_gb:.2f} GB"
        lines.append(_row(f" T0 VRAM \u2502 {self.peak_t0_gb:>8.2f} \u2502 {t0_cap_str}", W))
        lines.append(_row(f" T1 CPU  \u2502 {self.peak_t1_gb:>8.2f} \u2502 auto", W))
        lines.append(_row(f" T2 NVMe \u2502 {self.peak_t2_gb:>8.2f} \u2502 auto", W))

        # ── Summary ──────────────────────────────────────────────────────
        lines.append(f"\u2560{'═' * W}\u2563")
        lines.append(_row(
            f" Load time: {load_str}s | Throughput: {tok_str} tok/s",
            W,
        ))
        lines.append(_row(
            f" OOM-free: {oom_icon} (margin: {self.feasibility_margin:.2f}x)",
            W,
        ))

        # ── Bottom border ────────────────────────────────────────────────
        lines.append(f"\u255a{'═' * W}\u255d")

        return "\n".join(lines)


# ── Main function ────────────────────────────────────────────────────────


def memory_budget(
    params: float,
    vocab_size: int,
    context_len: int,
    hardware: str,
    d_model: Optional[int] = None,
    n_layers: Optional[int] = None,
    n_kv_heads: Optional[int] = None,
    dtype_bytes: int = 2,
) -> BudgetReport:
    """Compute a detailed memory budget for a model configuration.

    If ``d_model``, ``n_layers``, or ``n_kv_heads`` are not given they are
    estimated from *params* using standard scaling laws.

    Time: O(1).  Space: O(1).

    Parameters
    ----------
    params : float
        Number of model parameters (e.g. ``7e9``).
    vocab_size : int
        Vocabulary size.
    context_len : int
        Maximum sequence length.
    hardware : str
        Hardware preset key (``"H100"``, ``"A100"``, ``"H200"``,
        ``"TPU_v5e"``, ``"RTX_4090"``).
    d_model : int | None
        Hidden dimension.  Estimated from *params* if ``None``.
    n_layers : int | None
        Number of transformer layers.  Estimated if ``None``.
    n_kv_heads : int | None
        Number of key-value heads (GQA).  Estimated if ``None``.
    dtype_bytes : int
        Bytes per parameter (default 2 for FP16 / BF16).

    Returns
    -------
    BudgetReport
        Complete memory budget with formatted ``__str__`` output.
    """
    hw = HARDWARE_PRESETS.get(hardware)
    if hw is None:
        raise ValueError(
            f"Unknown hardware preset {hardware!r}. "
            f"Available: {list(HARDWARE_PRESETS.keys())}"
        )

    # ── Estimate architecture from params if needed ──────────────────────
    if d_model is None:
        d_model = int(math.sqrt(params / 12))
        # Round to nearest multiple of 128
        d_model = max(128, (d_model + 63) // 128 * 128)

    if n_layers is None:
        n_layers = max(1, round(params / (12 * d_model * d_model)))

    if n_kv_heads is None:
        # GQA heuristic: n_kv_heads ≈ d_model / 128, minimum 1
        n_kv_heads = max(1, d_model // 128)

    d_head = d_model // max(1, n_kv_heads * 8)  # Assume n_heads ≈ 8 × n_kv_heads
    if d_head <= 0:
        d_head = 128

    # ── Compute memory components ────────────────────────────────────────

    # Weights: params × dtype_bytes
    weights_gb = params * dtype_bytes / (1024 ** 3)

    # Activations: ~2× weights for training with selective checkpointing
    activations_gb = weights_gb * 2.0

    # KV cache: 2 × n_kv_heads × d_head × context_len × n_layers × dtype_bytes
    kv_cache_bytes = (
        2 * n_kv_heads * d_head * context_len * n_layers * dtype_bytes
    )
    kv_cache_gb = kv_cache_bytes / (1024 ** 3)

    # Optimizer states: 2× weights (Adam: first + second moment)
    optimizer_gb = weights_gb * 2.0

    # ── Tier assignment ──────────────────────────────────────────────────
    total_gb = weights_gb + activations_gb + kv_cache_gb + optimizer_gb

    vram = hw.vram_gb
    # T0 gets as much as fits (85% usable to leave headroom)
    usable_t0 = vram * 0.85
    peak_t0 = min(total_gb, usable_t0)
    spill_to_t1 = max(0.0, total_gb - usable_t0)
    peak_t1 = spill_to_t1
    peak_t2 = 0.0  # Only used when T1 also overflows (not modelled here)

    # ── OOM-free check ───────────────────────────────────────────────────
    oom_free = total_gb <= usable_t0
    if usable_t0 > 0:
        feasibility_margin = (usable_t0 - total_gb) / usable_t0 if oom_free else total_gb / usable_t0
        if oom_free:
            feasibility_margin = (usable_t0 - total_gb) / total_gb if total_gb > 0 else float("inf")
        else:
            feasibility_margin = usable_t0 / total_gb
    else:
        feasibility_margin = 0.0

    # ── Load time estimate ───────────────────────────────────────────────
    # model_size / NVMe bandwidth
    model_size_gb = weights_gb
    load_time_s = model_size_gb / hw.nvme_bw_gb_s if hw.nvme_bw_gb_s > 0 else float("inf")

    # ── Throughput estimate ──────────────────────────────────────────────
    # tokens/sec ≈ (compute_TFLOPS × 1e12 × MFU) / (6 × params)
    mfu = 0.45 if hw.nvlink_bw_gb_s > 0 else 0.35
    flops_per_token = 6 * params
    if flops_per_token > 0:
        tokens_per_sec = (hw.compute_tflops_fp16 * 1e12 * mfu) / flops_per_token
    else:
        tokens_per_sec = 0.0

    return BudgetReport(
        model_params=params,
        vocab_size=vocab_size,
        context_len=context_len,
        hardware_name=hardware,
        weights_gb=weights_gb,
        activations_gb=activations_gb,
        kv_cache_gb=kv_cache_gb,
        optimizer_gb=optimizer_gb,
        peak_t0_gb=peak_t0,
        peak_t1_gb=peak_t1,
        peak_t2_gb=peak_t2,
        t0_capacity_gb=vram,
        estimated_load_time_s=load_time_s,
        estimated_tokens_per_sec=tokens_per_sec,
        oom_free=oom_free,
        feasibility_margin=round(feasibility_margin, 4),
    )


# ── Formatting helpers ───────────────────────────────────────────────────


def _format_params(params: float) -> str:
    """Format parameter count as human-readable string (e.g. ``7.0B``)."""
    if params >= 1e12:
        return f"{params / 1e12:.1f}T"
    if params >= 1e9:
        return f"{params / 1e9:.1f}B"
    if params >= 1e6:
        return f"{params / 1e6:.1f}M"
    return f"{params:,.0f}"


def _row(content: str, width: int) -> str:
    """Format a single row: ``║ content (padded to *width*) ║``.

    Accounts for wide characters (emoji) that occupy two terminal columns.
    """
    display_len = _display_width(content)
    if display_len < width:
        padded = content + " " * (width - display_len)
    else:
        padded = content
    return f"\u2551{padded}\u2551"


def _display_width(text: str) -> int:
    """Return the display width of *text*, counting wide chars as 2."""
    w = 0
    for ch in text:
        cp = ord(ch)
        # Common emoji and wide character ranges
        if (
            0x1F300 <= cp <= 0x1F9FF  # Misc Symbols, Emoticons, etc.
            or 0x2600 <= cp <= 0x27BF  # Misc Symbols, Dingbats
            or 0x2700 <= cp <= 0x27BF
            or 0xFE00 <= cp <= 0xFE0F  # Variation selectors
            or 0x200D == cp  # ZWJ
        ):
            w += 2
        else:
            w += 1
    return w
