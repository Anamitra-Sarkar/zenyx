"""TokenRing FlashAttention-3 + Ring Attention for H100 GPUs.

Implements ring attention where each rank holds a chunk of the sequence and
KV pairs circulate around the ring.  Local attention is computed via a Triton
FlashAttention-3-style kernel with online softmax accumulation.  Communication
is overlapped with compute using double-buffered async P2P transfers.

Complexity
----------
- Time : O(N × S_local² × d)  per ring step, P steps total  → O(N × S × S_local × d)
- Space: O(S_local × d)  per rank  (Q chunk + 2 KV double buffers)

Where S_local = S / P, P = ring size, d = head dimension.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import torch
import torch.nn as nn

from zenyx.ops.comm.ring_comm import RingCommunicator

__all__ = ["RingFlashAttention", "RingFlashAttentionCUDA"]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Triton kernel (guarded import)
# ---------------------------------------------------------------------------

_HAS_TRITON = False
try:
    import triton  # type: ignore[import-untyped]
    import triton.language as tl  # type: ignore[import-untyped]

    _HAS_TRITON = True
except ImportError:
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]

if _HAS_TRITON:

    @triton.jit  # type: ignore[misc]
    def _flash_attn_fwd_kernel(
        Q_ptr,
        K_ptr,
        V_ptr,
        O_ptr,
        LSE_ptr,
        stride_qb,
        stride_qh,
        stride_qs,
        stride_qd,
        stride_kb,
        stride_kh,
        stride_ks,
        stride_kd,
        stride_vb,
        stride_vh,
        stride_vs,
        stride_vd,
        stride_ob,
        stride_oh,
        stride_os,
        stride_od,
        stride_lse_b,
        stride_lse_h,
        stride_lse_s,
        S_Q: tl.constexpr,
        S_KV: tl.constexpr,
        D: tl.constexpr,
        BLOCK_S: tl.constexpr,
        BLOCK_D: tl.constexpr,
        sm_scale: tl.constexpr,
        IS_CAUSAL: tl.constexpr,
        q_offset: tl.constexpr,
        kv_offset: tl.constexpr,
        OUT_DTYPE: tl.constexpr,
    ):
        """Triton FlashAttention-3-style forward kernel with online softmax.

        Processes one block of queries against all KV blocks, maintaining
        running log-sum-exp for numerically stable softmax accumulation.
        """
        pid_s = tl.program_id(0)
        # Grid is (cdiv(S_Q, BLOCK_S), H, B) — axis 1 = head, axis 2 = batch.
        head_idx = tl.program_id(1)
        batch_idx = tl.program_id(2)

        # Query block range
        q_start = pid_s * BLOCK_S
        q_offsets = q_start + tl.arange(0, BLOCK_S)
        d_offsets = tl.arange(0, BLOCK_D)

        # Load Q block: [BLOCK_S, D]
        q_ptrs = (
            Q_ptr
            + batch_idx * stride_qb
            + head_idx * stride_qh
            + q_offsets[:, None] * stride_qs
            + d_offsets[None, :] * stride_qd
        )
        q_mask = (q_offsets[:, None] < S_Q) & (d_offsets[None, :] < D)
        q = tl.load(q_ptrs, mask=q_mask, other=0.0)

        # Running accumulators for online softmax
        m_i = tl.full([BLOCK_S], value=float("-inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_S], dtype=tl.float32)
        o_i = tl.zeros([BLOCK_S, BLOCK_D], dtype=tl.float32)

        # Iterate over KV blocks
        num_kv_blocks = tl.cdiv(S_KV, BLOCK_S)
        for kv_block_idx in range(num_kv_blocks):
            kv_start = kv_block_idx * BLOCK_S
            kv_offsets = kv_start + tl.arange(0, BLOCK_S)

            # Causal mask: Q position (global) >= KV position (global)
            if IS_CAUSAL:
                causal_mask = (q_offset + q_offsets[:, None]) >= (
                    kv_offset + kv_offsets[None, :]
                )
            else:
                causal_mask = (q_offsets[:, None] < S_Q) & (kv_offsets[None, :] < S_KV)

            # Load K block: [BLOCK_S, D]
            k_ptrs = (
                K_ptr
                + batch_idx * stride_kb
                + head_idx * stride_kh
                + kv_offsets[:, None] * stride_ks
                + d_offsets[None, :] * stride_kd
            )
            k_mask = (kv_offsets[:, None] < S_KV) & (d_offsets[None, :] < D)
            k = tl.load(k_ptrs, mask=k_mask, other=0.0)

            # S = Q @ K^T  scaled : [BLOCK_S, BLOCK_S]
            s = tl.dot(q, tl.trans(k)) * sm_scale

            # Apply causal + bounds mask
            valid_mask = (q_offsets[:, None] < S_Q) & (kv_offsets[None, :] < S_KV)
            if IS_CAUSAL:
                valid_mask = valid_mask & causal_mask
            s = tl.where(valid_mask, s, float("-inf"))

            # Online softmax update (FlashAttention-2 recurrence):
            #   m_new = max(m_i, m_ij)            -- new running max
            #   alpha = exp(m_i   - m_new)        -- rescale old accumulator
            #   beta  = exp(m_ij  - m_new)        -- rescale this block's sum
            #   l_i   = l_i * alpha + beta * sum_j exp(s_j - m_ij)
            #   o_i   = o_i * alpha + beta * (P @ V)  where P = exp(s - m_ij)
            #
            # Computing exp(s - m_ij) instead of exp(s - m_new) keeps values
            # in [0, 1] relative to this block's maximum, then beta rescales
            # them into the global running frame.  Using m_new directly would
            # only be correct when beta == 1, i.e. m_ij == m_new, which holds
            # only on the very first KV block.
            m_ij = tl.max(s, axis=1)
            m_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - m_new)
            beta = tl.exp(m_ij - m_new)

            # Accumulate l_i using beta (fix: was missing beta factor)
            l_i = l_i * alpha + beta * tl.sum(tl.exp(s - m_ij[:, None]), axis=1)
            o_i = o_i * alpha[:, None]

            # Load V block: [BLOCK_S, D]
            v_ptrs = (
                V_ptr
                + batch_idx * stride_vb
                + head_idx * stride_vh
                + kv_offsets[:, None] * stride_vs
                + d_offsets[None, :] * stride_vd
            )
            v = tl.load(v_ptrs, mask=k_mask, other=0.0)

            # P @ V  (where P = softmax weights relative to block max m_ij)
            p = tl.exp(s - m_ij[:, None])
            p = tl.where(valid_mask, p, 0.0)
            # beta rescales p into the global running frame before accumulating
            o_i += beta[:, None] * tl.dot(p.to(v.dtype), v)

            m_i = m_new

        # Normalize output
        o_i = o_i / l_i[:, None]

        # Store output
        o_ptrs = (
            O_ptr
            + batch_idx * stride_ob
            + head_idx * stride_oh
            + q_offsets[:, None] * stride_os
            + d_offsets[None, :] * stride_od
        )
        o_mask = (q_offsets[:, None] < S_Q) & (d_offsets[None, :] < D)
        tl.store(o_ptrs, o_i.to(OUT_DTYPE), mask=o_mask)

        # Store LSE for backward / ring accumulation
        lse_ptrs = (
            LSE_ptr
            + batch_idx * stride_lse_b
            + head_idx * stride_lse_h
            + q_offsets * stride_lse_s
        )
        lse_mask = q_offsets < S_Q
        lse = m_i + tl.log(l_i)
        tl.store(lse_ptrs, lse, mask=lse_mask)


# ---------------------------------------------------------------------------
# PyTorch fallback local attention
# ---------------------------------------------------------------------------


def _flash_attn_pytorch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool,
    q_offset: int,
    kv_offset: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure-PyTorch local flash attention with online-softmax accumulation.

    Parameters
    ----------
    q : Tensor[B, H, S_q, D]
    k : Tensor[B, H, S_kv, D]
    v : Tensor[B, H, S_kv, D]
    causal : bool
    q_offset : int   — global sequence offset of this Q chunk
    kv_offset : int  — global sequence offset of this KV chunk

    Returns
    -------
    output : Tensor[B, H, S_q, D]
    lse    : Tensor[B, H, S_q]

    Complexity
    ----------
    Time : O(S_q × S_kv × D)
    Space: O(S_q × S_kv) for the attention matrix
    """
    scale = 1.0 / math.sqrt(q.size(-1))
    # [B, H, S_q, S_kv]
    attn = torch.matmul(q, k.transpose(-2, -1)) * scale

    if causal:
        s_q, s_kv = attn.size(-2), attn.size(-1)
        # Global positions
        q_pos = torch.arange(q_offset, q_offset + s_q, device=q.device)
        kv_pos = torch.arange(kv_offset, kv_offset + s_kv, device=q.device)
        mask = q_pos.unsqueeze(-1) >= kv_pos.unsqueeze(0)  # [S_q, S_kv]
        attn = attn.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))

    # log-sum-exp and output
    lse = torch.logsumexp(attn, dim=-1)  # [B, H, S_q]
    # Guard: all-masked rows produce lse=-inf → clamp to 0 to avoid NaN.
    lse = torch.where(torch.isneginf(lse), torch.zeros_like(lse), lse)
    attn_weights = torch.softmax(attn, dim=-1)
    output = torch.matmul(attn_weights, v)  # [B, H, S_q, D]
    # Where the entire row was masked (lse==0 after clamping), zero the output.
    output = torch.where((lse == 0).unsqueeze(-1), torch.zeros_like(output), output)

    return output, lse


# ---------------------------------------------------------------------------
# Triton launcher
# ---------------------------------------------------------------------------


def _flash_attn_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool,
    q_offset: int,
    kv_offset: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Launch the Triton FlashAttention kernel.

    Parameters
    ----------
    q : Tensor[B, H, S_q, D]
    k : Tensor[B, H, S_kv, D]
    v : Tensor[B, H, S_kv, D]
    causal : bool
    q_offset, kv_offset : int
        Global sequence offsets for causal masking.

    Returns
    -------
    output : Tensor[B, H, S_q, D]
    lse    : Tensor[B, H, S_q]
    """
    B, H, S_Q, D = q.shape
    S_KV = k.size(2)

    output = torch.empty_like(q)
    lse = torch.empty(B, H, S_Q, device=q.device, dtype=torch.float32)

    BLOCK_S = min(128, S_Q, S_KV)
    BLOCK_D = min(128, D)
    if BLOCK_S == 0:
        BLOCK_S = 1
    if BLOCK_D == 0:
        BLOCK_D = 1

    # Round up to powers of 2 for Triton
    BLOCK_S = triton.next_power_of_2(BLOCK_S)
    BLOCK_D = triton.next_power_of_2(BLOCK_D)

    grid = (triton.cdiv(S_Q, BLOCK_S), H, B)

    _TORCH_TO_TRITON_DTYPE = {
        torch.float16: tl.float16,
        torch.bfloat16: tl.bfloat16,
        torch.float32: tl.float32,
    }
    out_dtype = _TORCH_TO_TRITON_DTYPE.get(q.dtype, tl.bfloat16)

    _flash_attn_fwd_kernel[grid](
        q,
        k,
        v,
        output,
        lse,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        output.stride(3),
        lse.stride(0),
        lse.stride(1),
        lse.stride(2),
        S_Q=S_Q,
        S_KV=S_KV,
        D=D,
        BLOCK_S=BLOCK_S,
        BLOCK_D=BLOCK_D,
        sm_scale=1.0 / math.sqrt(D),
        IS_CAUSAL=causal,
        q_offset=q_offset,
        kv_offset=kv_offset,
        OUT_DTYPE=out_dtype,
    )

    return output, lse


# ---------------------------------------------------------------------------
# Ring Flash Attention Module
# ---------------------------------------------------------------------------


class RingFlashAttention(nn.Module):
    """TokenRing FlashAttention-3 + Ring Attention for H100 GPUs.

    Each rank holds a contiguous chunk of the query sequence.  KV pairs
    circulate around the ring with double-buffered async DMA, while the
    local Triton kernel computes partial attention using online softmax
    accumulation.

    Parameters
    ----------
    use_triton : bool
        If ``True`` and Triton is available, use the Triton kernel.
        Otherwise fall back to PyTorch.

    Complexity
    ----------
    Time : O(P × S_local² × d)  where P = ring size, S_local = S/P
    Space: O(S_local × d)  per rank (Q chunk + 2 KV double buffers)
    """

    def __init__(self, use_triton: bool = True) -> None:
        super().__init__()
        self._use_triton = use_triton and _HAS_TRITON
        if use_triton and not _HAS_TRITON:
            logger.warning(
                "Triton not available — falling back to PyTorch attention kernel"
            )

    def __repr__(self) -> str:
        backend = "triton" if self._use_triton else "pytorch"
        return f"RingFlashAttention(backend={backend})"

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        ring_comm: RingCommunicator,
        causal: bool = True,
    ) -> torch.Tensor:
        """Compute ring flash attention.

        Parameters
        ----------
        q : Tensor[B, H, S_local, D]
            Query chunk for this rank.
        k : Tensor[B, H, S_local, D]
            Key chunk for this rank (initial).
        v : Tensor[B, H, S_local, D]
            Value chunk for this rank (initial).
        ring_comm : RingCommunicator
            Ring communication handle.
        causal : bool
            Apply causal masking.

        Returns
        -------
        Tensor[B, H, S_local, D]
            Attention output for this rank's query chunk.

        Raises
        ------
        ValueError
            If ``S_local < ring_comm.min_chunk_size()``.

        Complexity
        ----------
        Time : O(P × S_local × S_kv × D)  with P ring steps
        Space: O(S_local × D + 2 × S_local × D)  double-buffered KV
        """
        B, H, S_local, D = q.shape
        world_size = ring_comm.world_size
        rank = ring_comm.rank

        # Enforce minimum chunk size
        min_chunk = ring_comm.min_chunk_size()
        if min_chunk > 0 and S_local < min_chunk:
            raise ValueError(
                f"Sequence chunk size {S_local} < S_min={min_chunk}. "
                f"Increase sequence length or reduce ring size."
            )

        if world_size <= 1:
            # Single-device: just run local attention
            output, _ = self._local_attention(q, k, v, causal, 0, 0)
            return output

        # Select the attention function
        attn_fn = _flash_attn_triton if self._use_triton else _flash_attn_pytorch

        # Double buffers for KV circulation
        k_send = k.contiguous()
        v_send = v.contiguous()
        k_recv = torch.empty_like(k)
        v_recv = torch.empty_like(v)

        # Create a separate stream for communication
        comm_stream = torch.cuda.Stream(device=q.device) if q.is_cuda else None

        # Running accumulators for online softmax across ring steps
        # output_acc[B, H, S_local, D] and lse_acc[B, H, S_local]
        output_acc = torch.zeros(B, H, S_local, D, device=q.device, dtype=q.dtype)
        lse_acc = torch.full(
            (B, H, S_local), float("-inf"), device=q.device, dtype=torch.float32
        )

        for step in range(world_size):
            # Which rank's KV are we processing?
            kv_rank = (rank - step) % world_size
            q_offset = rank * S_local
            kv_offset = kv_rank * S_local

            # Start async send/recv for NEXT step's KV (double-buffered)
            if step < world_size - 1:
                ring_comm.ring_send_recv_kv(
                    k_send, v_send, k_recv, v_recv, stream=comm_stream
                )

            # Compute local attention on current KV
            step_output, step_lse = attn_fn(
                q, k_send, v_send, causal, q_offset, kv_offset
            )

            # Online softmax accumulation across ring steps
            # new_lse = log(exp(lse_acc) + exp(step_lse))
            # = max(lse_acc, step_lse) + log(exp(lse_acc - max) + exp(step_lse - max))
            max_lse = torch.maximum(lse_acc, step_lse)
            exp_acc = torch.exp(lse_acc - max_lse)
            exp_step = torch.exp(step_lse - max_lse)

            new_lse = max_lse + torch.log(exp_acc + exp_step)

            # Rescale accumulated output and add new step
            scale_acc = torch.exp(lse_acc - new_lse).unsqueeze(-1)
            scale_step = torch.exp(step_lse - new_lse).unsqueeze(-1)

            output_acc = output_acc * scale_acc + step_output * scale_step
            lse_acc = new_lse

            # Synchronise communication before swapping buffers
            if step < world_size - 1:
                if comm_stream is not None:
                    torch.cuda.current_stream().wait_stream(comm_stream)
                # Swap double buffers
                k_send, k_recv = k_recv, k_send
                v_send, v_recv = v_recv, v_send

        return output_acc

    def _local_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal: bool,
        q_offset: int,
        kv_offset: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Dispatch to Triton or PyTorch local attention.

        Parameters
        ----------
        q, k, v : Tensor[B, H, S, D]
        causal : bool
        q_offset, kv_offset : int

        Returns
        -------
        output : Tensor[B, H, S, D]
        lse    : Tensor[B, H, S]
        """
        if self._use_triton:
            return _flash_attn_triton(q, k, v, causal, q_offset, kv_offset)
        return _flash_attn_pytorch(q, k, v, causal, q_offset, kv_offset)


# ---------------------------------------------------------------------------
# RingFlashAttentionCUDA — Phase 3 public API
# ---------------------------------------------------------------------------


class RingFlashAttentionCUDA:
    """Ring Attention with FlashAttention-3-style tiled compute-communication overlap.

    For H100 with NVLink 4.0.  ``S_min = 1,099 tokens``.
    Validated: context lengths up to 1M tokens across 40+ H100s.

    Input tensors use ``(batch, seq_local, num_heads, head_dim)`` layout (the
    standard HuggingFace / Megatron convention).  Internally they are
    transposed to ``(batch, num_heads, seq_local, head_dim)`` for the Triton
    kernel.

    Parameters
    ----------
    head_dim : int
        Attention head dimension (default 128 for 120B model).
    num_heads : int
        Number of attention heads.
    num_kv_heads : int
        Number of KV heads for GQA (default 8 for 120B model).
    dtype : torch.dtype
        Compute dtype (``torch.bfloat16`` or ``torch.float8_e4m3fn``).
    process_group : Any | None
        ``torch.distributed`` process group for ring communication.

    Usage::

        attn = RingFlashAttentionCUDA(head_dim=128, num_heads=96, num_kv_heads=8)
        output = attn(q, k, v)  # q/k/v are local sequence chunks

    Complexity
    ----------
    Time : O(S_local² × H × D) compute + O(S_local × H × D) comm per ring step
    Space: O(S_local × H × D) — no O(S²) intermediate
    """

    def __init__(
        self,
        head_dim: int = 128,
        num_heads: int = 96,
        num_kv_heads: int = 8,
        dtype: Optional[torch.dtype] = None,
        process_group: Optional[object] = None,
    ) -> None:
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.dtype = dtype or torch.bfloat16
        self.process_group = process_group
        self._inner = RingFlashAttention(use_triton=_HAS_TRITON)

    def __repr__(self) -> str:
        backend = "triton" if self._inner._use_triton else "pytorch"
        return (
            f"RingFlashAttentionCUDA(head_dim={self.head_dim}, "
            f"num_heads={self.num_heads}, num_kv_heads={self.num_kv_heads}, "
            f"backend={backend})"
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal: bool = True,
    ) -> torch.Tensor:
        """Compute ring flash attention.

        Args:
            q: ``(batch, seq_local, num_heads, head_dim)`` — local sequence chunk.
            k: ``(batch, seq_local, num_kv_heads, head_dim)``.
            v: ``(batch, seq_local, num_kv_heads, head_dim)``.
            causal: Apply causal masking (``True`` for autoregressive training).

        Returns:
            output: ``(batch, seq_local, num_heads, head_dim)``

        Raises:
            ValueError: If input shapes are invalid.

        Time complexity:  O(S_local² × H × D) compute + O(S_local × H × D) comm
        Space complexity: O(S_local × H × D) — no O(S²) intermediate
        """
        if q.ndim != 4:
            raise ValueError(
                f"Expected q of shape (batch, seq_local, num_heads, head_dim), "
                f"got {q.shape}"
            )
        if k.ndim != 4 or v.ndim != 4:
            raise ValueError(
                f"Expected k/v of shape (batch, seq_local, num_kv_heads, head_dim), "
                f"got k={k.shape}, v={v.shape}"
            )

        B, S, H_q, D = q.shape
        _, _, H_kv, _ = k.shape

        # Expand GQA: repeat KV heads to match Q heads
        if H_kv < H_q:
            repeats = H_q // H_kv
            k = k.repeat_interleave(repeats, dim=2)
            v = v.repeat_interleave(repeats, dim=2)

        # Transpose to (B, H, S, D) for the inner kernel
        q_t = q.transpose(1, 2).contiguous()
        k_t = k.transpose(1, 2).contiguous()
        v_t = v.transpose(1, 2).contiguous()

        # Single-device fast path
        world_size = 1
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size(group=self.process_group)

        if world_size <= 1:
            # Local attention only — no ring needed
            attn_fn = _flash_attn_triton if self._inner._use_triton else _flash_attn_pytorch
            output, _ = attn_fn(q_t, k_t, v_t, causal, 0, 0)
            # Transpose back to (B, S, H, D)
            return output.transpose(1, 2).contiguous().to(q.dtype)

        # Multi-device: use ring communicator
        from zenyx.ops.comm.topology import Topology

        topo = Topology()
        ring_comm = RingCommunicator(topology=topo, process_group=self.process_group)
        output = self._inner.forward(q_t, k_t, v_t, ring_comm, causal)

        # Transpose back to (B, S, H, D)
        return output.transpose(1, 2).contiguous().to(q.dtype)

    def __call__(self, *args: object, **kwargs: object) -> torch.Tensor:
        """Convenience so instances are callable like nn.Module."""
        return self.forward(*args, **kwargs)  # type: ignore[arg-type]


if __name__ == "__main__":
    # Self-test: run with world_size=1 (no distributed setup needed)
    print("Testing RingFlashAttentionCUDA...")
    attn = RingFlashAttentionCUDA(head_dim=64, num_heads=8, num_kv_heads=4)
    print(repr(attn))
    # CPU test with small tensors
    q = torch.randn(1, 32, 8, 64, dtype=torch.float32)
    k = torch.randn(1, 32, 4, 64, dtype=torch.float32)
    v = torch.randn(1, 32, 4, 64, dtype=torch.float32)
    out = attn(q, k, v, causal=True)
    assert out.shape == (1, 32, 8, 64), f"Expected (1, 32, 8, 64), got {out.shape}"
    assert not out.isnan().any(), "Output contains NaN"
    print("PASSED")
