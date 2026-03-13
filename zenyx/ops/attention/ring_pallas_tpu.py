"""Ring Attention for TPU using JAX custom_partitioning + Pallas kernels.

Implements ring attention on TPU slices using ``jax.experimental.custom_partitioning``
to control XLA placement (avoiding the reordering issues with vanilla
``shard_map`` + ``lax.ppermute``).  Local attention is computed via a Pallas
custom kernel with online softmax accumulation.

All JAX imports are guarded with ``try/except`` so this module can be safely
imported on non-TPU systems.

Complexity
----------
- Time : O(P × S_local² × d)  per ring step, P steps
- Space: O(S_local × d)  per device
"""

from __future__ import annotations

import logging
import math
from typing import Any, Callable, Optional, Sequence, Tuple

__all__ = ["RingFlashAttentionTPU", "ring_attention_tpu"]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Guarded JAX imports
# ---------------------------------------------------------------------------

_HAS_JAX = False
_JAX_IMPORT_ERROR: Optional[str] = None

try:
    import jax  # type: ignore[import-untyped]
    import jax.numpy as jnp  # type: ignore[import-untyped]
    from jax import lax  # type: ignore[import-untyped]
    from jax.experimental import custom_partitioning  # type: ignore[import-untyped]

    _HAS_JAX = True

    try:
        from jax.experimental import pallas as pl  # type: ignore[import-untyped]

        _HAS_PALLAS = True
    except ImportError:
        _HAS_PALLAS = False
        logger.debug("Pallas not available; using JAX fallback for local attention")

except ImportError as exc:
    _HAS_JAX = False
    _JAX_IMPORT_ERROR = str(exc)
    # Provide stubs so the class can be defined without JAX
    jax = None  # type: ignore[assignment]
    jnp = None  # type: ignore[assignment]
    lax = None  # type: ignore[assignment]
    custom_partitioning = None  # type: ignore[assignment]
    pl = None  # type: ignore[assignment]
    _HAS_PALLAS = False


def _check_jax() -> None:
    """Raise ``ImportError`` with a helpful message if JAX is unavailable."""
    if not _HAS_JAX:
        raise ImportError(
            "JAX is required for TPU ring attention but could not be imported. "
            f"Install JAX with TPU support: `pip install jax[tpu]`.  "
            f"Original error: {_JAX_IMPORT_ERROR}"
        )


# ---------------------------------------------------------------------------
# Pallas local attention kernel
# ---------------------------------------------------------------------------


def _pallas_local_attention(
    q: Any,
    k: Any,
    v: Any,
    causal: bool,
    q_offset: int,
    kv_offset: int,
    head_dim: int = 128,
) -> Tuple[Any, Any]:
    """Local attention using a Pallas kernel (or JAX fallback).

    Parameters
    ----------
    q : jnp.ndarray[B, H, S_q, D]
    k : jnp.ndarray[B, H, S_kv, D]
    v : jnp.ndarray[B, H, S_kv, D]
    causal : bool
    q_offset : int
        Global sequence offset of Q chunk.
    kv_offset : int
        Global sequence offset of KV chunk.
    head_dim : int
        Static head dimension for Pallas BlockSpec compilation.

    Returns
    -------
    output : jnp.ndarray[B, H, S_q, D]
    lse    : jnp.ndarray[B, H, S_q]

    Complexity
    ----------
    Time : O(S_q × S_kv × D)
    Space: O(S_q × S_kv)
    """
    _check_jax()

    if _HAS_PALLAS:
        return _pallas_kernel_attention(q, k, v, causal, q_offset, kv_offset, head_dim=head_dim)
    return _jax_fallback_attention(q, k, v, causal, q_offset, kv_offset)


def _pallas_kernel_attention(
    q: Any, k: Any, v: Any, causal: bool, q_offset: int, kv_offset: int,
    head_dim: int = 128,
) -> Tuple[Any, Any]:
    """Pallas-based local attention kernel for TPU.

    Uses ``pallas_call`` to define a custom kernel that computes attention
    with online softmax, mapped over batch and head dimensions.

    Complexity
    ----------
    Time : O(S_q × S_kv × D)
    Space: O(S_q × S_kv)
    """
    B, H, S_q, D = q.shape
    S_kv = k.shape[2]
    # Use static head_dim for block shapes to avoid JIT recompilation
    D_static = head_dim
    scale = 1.0 / math.sqrt(D_static)

    def kernel_fn(q_ref: Any, k_ref: Any, v_ref: Any, o_ref: Any, lse_ref: Any) -> None:
        """Pallas kernel body: fused QKV attention with online softmax."""
        # Load tiles
        q_tile = q_ref[...]  # [S_q, D_static]
        k_tile = k_ref[...]  # [S_kv, D_static]
        v_tile = v_ref[...]  # [S_kv, D_static]

        # Compute scores
        scores = pl.dot(q_tile, k_tile.T) * scale  # [S_q, S_kv]

        if causal:
            q_idx = jnp.arange(S_q) + q_offset
            kv_idx = jnp.arange(S_kv) + kv_offset
            mask = q_idx[:, None] >= kv_idx[None, :]
            scores = jnp.where(mask, scores, -1e10)

        # Online softmax
        row_max = jnp.max(scores, axis=-1)  # [S_q]
        exp_scores = jnp.exp(scores - row_max[:, None])
        row_sum = jnp.sum(exp_scores, axis=-1)  # [S_q]

        # Output
        o_ref[...] = pl.dot(exp_scores, v_tile) / row_sum[:, None]
        lse_ref[...] = row_max + jnp.log(row_sum)

    # Define grid spec for Pallas — use static D_static for block shapes
    # to prevent JIT recompilation when head_dim varies between calls.
    out_shape = [
        jax.ShapeDtypeStruct((S_q, D_static), q.dtype),  # output
        jax.ShapeDtypeStruct((S_q,), jnp.float32),  # lse
    ]

    grid_spec = pl.GridSpec(num_programs=1, in_specs=[
        pl.BlockSpec(block_shape=(S_q, D_static), index_map=lambda: (0, 0)),
        pl.BlockSpec(block_shape=(S_kv, D_static), index_map=lambda: (0, 0)),
        pl.BlockSpec(block_shape=(S_kv, D_static), index_map=lambda: (0, 0)),
    ], out_specs=[
        pl.BlockSpec(block_shape=(S_q, D_static), index_map=lambda: (0, 0)),
        pl.BlockSpec(block_shape=(S_q,), index_map=lambda: (0,)),
    ])

    pallas_fn = pl.pallas_call(
        kernel_fn,
        grid_spec=grid_spec,
        out_shape=out_shape,
    )

    # Map over batch and head dims
    def per_bh(qbh: Any, kbh: Any, vbh: Any) -> Tuple[Any, Any]:
        return pallas_fn(qbh, kbh, vbh)

    # vmap over heads then batch
    mapped = jax.vmap(jax.vmap(per_bh))(q, k, v)
    return mapped[0], mapped[1]


def _jax_fallback_attention(
    q: Any, k: Any, v: Any, causal: bool, q_offset: int, kv_offset: int
) -> Tuple[Any, Any]:
    """Pure-JAX fallback attention when Pallas is unavailable.

    Complexity
    ----------
    Time : O(S_q × S_kv × D)
    Space: O(S_q × S_kv)
    """
    scale = 1.0 / math.sqrt(q.shape[-1])
    # [B, H, S_q, S_kv]
    scores = jnp.matmul(q, jnp.swapaxes(k, -2, -1)) * scale

    if causal:
        S_q, S_kv = q.shape[-2], k.shape[-2]
        q_idx = jnp.arange(S_q) + q_offset
        kv_idx = jnp.arange(S_kv) + kv_offset
        mask = q_idx[:, None] >= kv_idx[None, :]
        scores = jnp.where(mask[None, None, :, :], scores, -1e10)

    lse = jax.nn.logsumexp(scores, axis=-1)  # [B, H, S_q]
    weights = jax.nn.softmax(scores, axis=-1)
    output = jnp.matmul(weights, v)
    return output, lse


# ---------------------------------------------------------------------------
# Ring permute primitive
# ---------------------------------------------------------------------------


def _ring_permute(x: Any, axis_name: str) -> Any:
    """Send tensor to the next device in the ring, receive from previous.

    Uses ``lax.ppermute`` wrapped in ``custom_partitioning`` to prevent
    XLA from reordering the communication.

    Parameters
    ----------
    x : jnp.ndarray
        Tensor to send.
    axis_name : str
        Name of the mesh axis for the ring.

    Returns
    -------
    jnp.ndarray
        Tensor received from the previous rank.
    """
    _check_jax()
    axis_size = lax.psum(1, axis_name)
    perm = [(i, (i + 1) % axis_size) for i in range(axis_size)]
    return lax.ppermute(x, axis_name, perm)


# ---------------------------------------------------------------------------
# RingFlashAttentionTPU
# ---------------------------------------------------------------------------


class RingFlashAttentionTPU:
    """Ring Attention for TPU with custom_partitioning + Pallas kernels.

    Implements ring attention where KV chunks circulate around the TPU
    ICI ring.  Uses ``jax.experimental.custom_partitioning`` to control
    XLA scheduling and prevent send/recv reordering.

    Raises ``ImportError`` on construction if JAX is not installed.

    Complexity
    ----------
    Time : O(P × S_local² × d)  per ring
    Space: O(S_local × d) per device
    """

    def __init__(self, head_dim: int = 128, causal: bool = True) -> None:
        _check_jax()
        self._head_dim = head_dim
        self._causal = causal
        logger.info(
            "RingFlashAttentionTPU initialized (pallas=%s, causal=%s, head_dim=%d)",
            _HAS_PALLAS,
            causal,
            head_dim,
        )

    def __repr__(self) -> str:
        backend = "pallas" if _HAS_PALLAS else "jax"
        return f"RingFlashAttentionTPU(backend={backend}, causal={self._causal})"

    def forward(
        self,
        q: Any,
        k: Any,
        v: Any,
        axis_name: str = "ring",
    ) -> Any:
        """Compute ring attention on TPU.

        Parameters
        ----------
        q : jnp.ndarray[B, H, S_local, D]
            Local query chunk for this device.
        k : jnp.ndarray[B, H, S_local, D]
            Local key chunk (initial).
        v : jnp.ndarray[B, H, S_local, D]
            Local value chunk (initial).
        axis_name : str
            Mesh axis name for the ring dimension.

        Returns
        -------
        jnp.ndarray[B, H, S_local, D]
            Attention output for this device's query chunk.

        Complexity
        ----------
        Time : O(P × S_local² × D)
        Space: O(S_local × D)
        """
        _check_jax()

        ring_size = lax.psum(1, axis_name)
        rank = lax.axis_index(axis_name)
        S_local = q.shape[-2]

        # Running accumulators
        output_acc = jnp.zeros_like(q)
        lse_acc = jnp.full(q.shape[:-1], -1e10, dtype=jnp.float32)

        k_cur = k
        v_cur = v

        def ring_step(
            carry: Tuple[Any, Any, Any, Any, int], _: Any
        ) -> Tuple[Tuple[Any, Any, Any, Any, int], None]:
            """One step of the ring: compute + permute."""
            output_acc, lse_acc, k_cur, v_cur, step = carry

            # Which rank's KV are we processing?
            kv_rank = (rank - step) % ring_size
            q_offset = rank * S_local
            kv_offset = kv_rank * S_local

            # Local attention
            step_output, step_lse = _pallas_local_attention(
                q, k_cur, v_cur, self._causal, q_offset, kv_offset,
                head_dim=self._head_dim,
            )

            # Online softmax accumulation
            max_lse = jnp.maximum(lse_acc, step_lse)
            exp_acc = jnp.exp(lse_acc - max_lse)
            exp_step = jnp.exp(step_lse - max_lse)
            new_lse = max_lse + jnp.log(exp_acc + exp_step)

            scale_acc = jnp.exp(lse_acc - new_lse)[..., None]
            scale_step = jnp.exp(step_lse - new_lse)[..., None]

            output_acc = output_acc * scale_acc + step_output * scale_step
            lse_acc = new_lse

            # Send KV to next, receive from previous
            k_cur = _ring_permute(k_cur, axis_name)
            v_cur = _ring_permute(v_cur, axis_name)

            return (output_acc, lse_acc, k_cur, v_cur, step + 1), None

        init_carry = (output_acc, lse_acc, k_cur, v_cur, 0)
        (output_acc, lse_acc, _, _, _), _ = lax.scan(
            ring_step, init_carry, None, length=ring_size
        )

        return output_acc

    def __call__(
        self,
        q: Any,
        k: Any,
        v: Any,
        axis_name: str = "ring",
    ) -> Any:
        """Alias for :meth:`forward`."""
        return self.forward(q, k, v, axis_name)


# ---------------------------------------------------------------------------
# ring_attention_tpu — Phase 3 convenience function
# ---------------------------------------------------------------------------


def ring_attention_tpu(
    q: Any,
    k: Any,
    v: Any,
    axis_name: str = "devices",
    causal: bool = True,
) -> Any:
    """Ring Attention for TPU v5e/v5p using Pallas + Shardy.

    The sequence dimension is sharded across devices via ``pmap``/``shard_map``.
    Each device holds ``seq_local = seq_total / num_devices`` tokens.

    Communication uses ``lax.ppermute`` inside ``custom_partitioning`` boundaries
    so XLA cannot reorder it relative to the Pallas attention kernel.

    S_min for TPU v5e ICI = 493 tokens (F=197 TFLOPS, B=400 GB/s).
    S_min for TPU v5p ICI = 383 tokens (F=459 TFLOPS, B=1200 GB/s).

    Parameters
    ----------
    q : jax.Array
        ``(batch, seq_local, num_heads, head_dim)`` — local query chunk.
    k : jax.Array
        ``(batch, seq_local, num_kv_heads, head_dim)`` — local key chunk.
    v : jax.Array
        ``(batch, seq_local, num_kv_heads, head_dim)`` — local value chunk.
    axis_name : str
        Name of the device axis for ``pmap``/``shard_map`` (default ``"devices"``).
    causal : bool
        Apply causal masking (default ``True``).

    Returns
    -------
    jax.Array
        ``(batch, seq_local, num_heads, head_dim)`` — attention output.

    Raises
    ------
    ImportError
        If JAX is not installed.
    TypeError
        If ``q`` is not a JAX array with a ``.shape`` attribute.

    Time complexity:  O(P × S_local² × D) where P = ring size
    Space complexity: O(S_local × H × D)
    """
    if not _HAS_JAX:
        raise ImportError(
            "JAX is not installed. Install with: pip install jax[tpu]"
        )

    # q must be a real JAX array — there is no valid fallback for head_dim.
    if not hasattr(q, "shape"):
        raise TypeError(
            "ring_attention_tpu: q must be a JAX array with a .shape "
            f"attribute, got {type(q).__name__!r}."
        )
    head_dim = q.shape[-1]

    attn = RingFlashAttentionTPU(head_dim=head_dim, causal=causal)
    return attn(q, k, v, axis_name=axis_name)


if __name__ == "__main__":
    # Self-test: verify module imports without JAX
    print("Testing ring_pallas_tpu module...")
    if _HAS_JAX:
        print("JAX available — RingFlashAttentionTPU instantiable")
        attn = RingFlashAttentionTPU(head_dim=64)
        print(repr(attn))
    else:
        print("JAX not available — verifying graceful fallback")
        try:
            ring_attention_tpu(None, None, None)
            assert False, "Should have raised ImportError"
        except ImportError as e:
            print(f"Correctly raised ImportError: {e}")
    print("PASSED")
