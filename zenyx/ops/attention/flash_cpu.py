"""CPU fallback attention using ``torch.nn.functional.scaled_dot_product_attention``.

Memory-efficient chunked processing avoids materialising the full attention
matrix for long sequences.  Falls back to a manual implementation for
PyTorch < 2.0.

Complexity
----------
- Time : O(S² × D)  — standard attention
- Space: O(C × S × D)  chunked — where C = chunk size (default 1024)
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["flash_attention_cpu"]

logger = logging.getLogger(__name__)

# Check for PyTorch 2.0+ SDPA
_HAS_SDPA = hasattr(F, "scaled_dot_product_attention")


class FlashAttentionCPU(nn.Module):
    """CPU fallback attention with chunked memory-efficient processing.

    For single-device CPU inference/training where ring attention is not
    applicable.  Uses ``torch.nn.functional.scaled_dot_product_attention``
    (PyTorch 2.0+) when available, otherwise a manual chunked implementation.

    Parameters
    ----------
    chunk_size : int
        Number of query tokens to process per chunk.  Larger chunks use
        more memory but have less overhead.  Default: 1024.

    Complexity
    ----------
    Time : O(S² × D)
    Space: O(chunk_size × S × D)  — never materialises full S×S matrix
    """

    def __init__(self, chunk_size: int = 1024) -> None:
        super().__init__()
        self._chunk_size = chunk_size

    def __repr__(self) -> str:
        backend = "sdpa" if _HAS_SDPA else "manual"
        return f"FlashAttentionCPU(chunk_size={self._chunk_size}, backend={backend})"

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal: bool = True,
    ) -> torch.Tensor:
        """Compute scaled dot-product attention on CPU.

        Internal use. For the public API use :func:`flash_attention_cpu`.

        Parameters
        ----------
        q : Tensor[B, H, S_q, D]
            Queries.
        k : Tensor[B, H, S_kv, D]
            Keys.
        v : Tensor[B, H, S_kv, D]
            Values.
        causal : bool
            Apply causal (lower-triangular) mask.

        Returns
        -------
        Tensor[B, H, S_q, D]
            Attention output.

        Complexity
        ----------
        Time : O(S_q × S_kv × D)
        Space: O(min(S_q, chunk_size) × S_kv)  per chunk
        """
        S_q = q.size(2)

        # If sequence fits in one chunk, use non-chunked path
        if S_q <= self._chunk_size:
            return self._attention_full(q, k, v, causal)

        # Chunked processing
        return self._attention_chunked(q, k, v, causal)

    def _attention_full(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal: bool,
    ) -> torch.Tensor:
        """Non-chunked attention for short sequences.

        Complexity
        ----------
        Time : O(S_q × S_kv × D)
        Space: O(S_q × S_kv)
        """
        if _HAS_SDPA:
            return F.scaled_dot_product_attention(
                q, k, v, is_causal=causal, dropout_p=0.0
            )
        return self._manual_attention(q, k, v, causal)

    def _attention_chunked(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal: bool,
    ) -> torch.Tensor:
        """Chunked attention for long sequences.

        Processes ``chunk_size`` query tokens at a time to limit the peak
        memory of the attention matrix.

        Parameters
        ----------
        q : Tensor[B, H, S_q, D]
        k : Tensor[B, H, S_kv, D]
        v : Tensor[B, H, S_kv, D]
        causal : bool

        Returns
        -------
        Tensor[B, H, S_q, D]

        Complexity
        ----------
        Time : O(S_q × S_kv × D)
        Space: O(chunk_size × S_kv)  per chunk
        """
        B, H, S_q, D = q.shape
        output = torch.empty_like(q)

        for start in range(0, S_q, self._chunk_size):
            end = min(start + self._chunk_size, S_q)
            q_chunk = q[:, :, start:end, :]

            if causal:
                # For causal: only attend to KV positions <= query position
                # We need KV up to position `end` (inclusive)
                kv_end = min(end, k.size(2))
                k_chunk = k[:, :, :kv_end, :]
                v_chunk = v[:, :, :kv_end, :]

                if _HAS_SDPA and start == 0:
                    # First chunk: standard causal
                    chunk_out = F.scaled_dot_product_attention(
                        q_chunk, k_chunk, v_chunk, is_causal=True, dropout_p=0.0
                    )
                else:
                    # Non-first chunk or no SDPA: manual with custom mask
                    chunk_out = self._manual_attention_with_offset(
                        q_chunk, k_chunk, v_chunk, q_start=start
                    )
            else:
                if _HAS_SDPA:
                    chunk_out = F.scaled_dot_product_attention(
                        q_chunk, k, v, is_causal=False, dropout_p=0.0
                    )
                else:
                    chunk_out = self._manual_attention(q_chunk, k, v, causal=False)

            output[:, :, start:end, :] = chunk_out

        return output

    @staticmethod
    def _manual_attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal: bool,
    ) -> torch.Tensor:
        """Manual scaled dot-product attention (PyTorch < 2.0 fallback).

        Complexity
        ----------
        Time : O(S_q × S_kv × D)
        Space: O(S_q × S_kv)
        """
        scale = 1.0 / math.sqrt(q.size(-1))
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, H, S_q, S_kv]

        if causal:
            S_q, S_kv = attn.size(-2), attn.size(-1)
            # Standard lower-triangular causal mask
            row_idx = torch.arange(S_q, device=q.device).unsqueeze(1)
            col_idx = torch.arange(S_kv, device=q.device).unsqueeze(0)
            # For rectangular attention: shift so last query attends to last KV
            offset = S_kv - S_q
            mask = row_idx + offset >= col_idx
            attn = attn.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn = torch.softmax(attn, dim=-1)
        return torch.matmul(attn, v)

    @staticmethod
    def _manual_attention_with_offset(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_start: int,
    ) -> torch.Tensor:
        """Manual attention with global position offset for causal chunks.

        Parameters
        ----------
        q : Tensor[B, H, C, D]
            Query chunk.
        k : Tensor[B, H, S_kv, D]
            Keys (up to relevant positions).
        v : Tensor[B, H, S_kv, D]
            Values (up to relevant positions).
        q_start : int
            Global start position of this query chunk.

        Returns
        -------
        Tensor[B, H, C, D]

        Complexity
        ----------
        Time : O(C × S_kv × D)
        Space: O(C × S_kv)
        """
        scale = 1.0 / math.sqrt(q.size(-1))
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale

        S_q_chunk = attn.size(-2)
        S_kv = attn.size(-1)

        # Causal mask with global offsets
        q_pos = torch.arange(q_start, q_start + S_q_chunk, device=q.device)
        kv_pos = torch.arange(S_kv, device=q.device)
        mask = q_pos.unsqueeze(1) >= kv_pos.unsqueeze(0)
        attn = attn.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn = torch.softmax(attn, dim=-1)
        return torch.matmul(attn, v)


# ---------------------------------------------------------------------------
# flash_attention_cpu — Phase 3 convenience function
# ---------------------------------------------------------------------------


def flash_attention_cpu(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = True,
    chunk_size: int = 512,
) -> torch.Tensor:
    """Memory-efficient chunked attention for CPU.

    Canonical public interface: expects tensors in ``(batch, seq, heads, head_dim)``.

    Processes attention in chunks of ``chunk_size`` to avoid materialising
    the full ``(seq, seq)`` attention matrix.  ``O(seq × chunk_size)`` memory
    instead of ``O(seq²)``.

    No ring communication — single device only.
    Suitable for: development, Apple M3 Ultra inference, CPU-only environments.

    NOT suitable for: training on context > 32K (too slow).

    Parameters
    ----------
    q : Tensor
        ``(batch, seq, heads, head_dim)`` — queries.
    k : Tensor
        ``(batch, seq, heads, head_dim)`` — keys.
    v : Tensor
        ``(batch, seq, heads, head_dim)`` — values.
    causal : bool
        Apply causal masking (default ``True``).
    chunk_size : int
        Number of query tokens per chunk (default 512).

    Returns
    -------
    Tensor
        ``(batch, seq, heads, head_dim)`` — attention output.

    Time complexity:  O(S² × D) total
    Space complexity: O(chunk_size × S × D) — never materialises full S×S
    """
    if q.ndim != 4:
        raise ValueError(
            f"Expected q of shape (batch, seq, heads, head_dim), got {q.shape}"
        )

    # Transpose from (B, S, H, D) to (B, H, S, D) for the FlashAttentionCPU module
    q_t = q.transpose(1, 2).contiguous()
    k_t = k.transpose(1, 2).contiguous()
    v_t = v.transpose(1, 2).contiguous()

    module = FlashAttentionCPU(chunk_size=chunk_size)
    output = module.forward(q_t, k_t, v_t, causal=causal)

    # Transpose back to (B, S, H, D)
    return output.transpose(1, 2).contiguous()


if __name__ == "__main__":
    # Self-test: run with world_size=1 (no distributed setup needed)
    print("Testing flash_attention_cpu...")
    q = torch.randn(1, 64, 4, 32)
    k = torch.randn(1, 64, 4, 32)
    v = torch.randn(1, 64, 4, 32)
    out = flash_attention_cpu(q, k, v, causal=True, chunk_size=16)
    # FIX: Avoid assert for runtime validation in self-test.
    if out.shape != (1, 64, 4, 32):
        raise RuntimeError(f"Expected (1, 64, 4, 32), got {out.shape}")
    if out.isnan().any():
        raise RuntimeError("Output contains NaN")
    print("PASSED")
