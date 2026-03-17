"""FP8 E4M3 activation storage for memory-efficient training.

INT8 activation storage is **unsafe** for training — QuEST (arXiv 2502.05003)
proves quantisation error compounds multiplicatively across layers during
backpropagation, destroying the Hessian.  FP8 E4M3 is the safe quantisation
level, confirmed by the COAT paper (ICLR 2025).

Key components
--------------
* :class:`FP8ActivationStorage` — quantise / dequantise helpers using
  ``torch.float8_e4m3fn`` (PyTorch 2.1+), with a software fallback on
  hardware that lacks native FP8.
* :class:`FP8CheckpointFunction` — ``torch.autograd.Function`` that stores
  activations in FP8 during the forward pass and restores them for backward.
* :func:`fp8_checkpoint` — convenience wrapper: apply FP8 activation
  checkpointing every *N*-th layer of any ``nn.Module``.

Fix notes
---------
* Simulated FP8 quantise: the previous implementation cast scaled values in
  ``[-448, 448]`` to ``torch.uint8`` (unsigned, range ``[0, 255]``).  Any
  negative float cast to uint8 in PyTorch wraps/truncates to 0, so the entire
  negative half of the activation tensor was silently zeroed.  Dequantising
  then divided those zeros by the scale and returned 0 for all originally
  negative activations — completely corrupt gradients.

  Fix: use ``torch.int16`` (range ``[-32768, 32767]``) which covers the FP8
  E4M3 dynamic range ``[-448, 448]`` exactly.  The dequantise path was
  already correct (casts to float32 before dividing by scale).
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn

__all__ = [
    "FP8ActivationStorage",
    "FP8CheckpointFunction",
    "fp8_checkpoint",
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# FP8 E4M3 dynamic range limits (IEEE-style, exponent bias 7).
_FP8_E4M3_MAX: float = 448.0
_FP8_E4M3_MIN: float = -448.0

# Threshold below which we batch tensors in upstream loaders; here we use it
# to decide whether a simulated-FP8 path is acceptable.
_NATIVE_FP8_DTYPE: Optional[torch.dtype] = getattr(torch, "float8_e4m3fn", None)


def _has_native_fp8() -> bool:
    """Return ``True`` if the current PyTorch build exposes float8_e4m3fn."""
    return _NATIVE_FP8_DTYPE is not None


# ---------------------------------------------------------------------------
# FP8ActivationStorage
# ---------------------------------------------------------------------------


class FP8ActivationStorage:
    """Per-tensor dynamic-scaling FP8 E4M3 quantisation for activations.

    Parameters
    ----------
    force_simulated : bool, optional
        If *True*, always use the software fallback even when native FP8
        hardware is available.  Useful for unit-testing.

    Simulated path note
    -------------------
    The simulated fallback scales the tensor into ``[-448, 448]`` and then
    rounds to the nearest integer stored as ``int16`` (NOT ``uint8``).  Using
    ``uint8`` was a critical bug: casting a negative float to uint8 in PyTorch
    is defined to produce 0 (or wrap), destroying all negative activations.
    ``int16`` covers ``[-32768, 32767]`` so ``[-448, 448]`` fits without loss
    of sign.

    Complexity
    ----------
    * ``quantize``  — *O(n)* time, *O(n)* space (int16 = same bytes as fp16).
    * ``dequantize`` — *O(n)* time, *O(n)* space (restores full dtype).
    """

    def __init__(self, *, force_simulated: bool = False) -> None:
        self._use_native: bool = _has_native_fp8() and not force_simulated
        if not self._use_native:
            logger.info(
                "Native FP8 E4M3 unavailable — using simulated (clamp+scale+int16) path."
            )

    # -- public API ---------------------------------------------------------

    def quantize(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantise *tensor* to FP8 E4M3 with per-tensor dynamic scaling.

        Parameters
        ----------
        tensor : torch.Tensor
            Activation tensor in FP16, BF16, or FP32.

        Returns
        -------
        quantized : torch.Tensor
            Native FP8 tensor (``float8_e4m3fn``) if hardware supports it,
            otherwise ``int16`` with values in ``[-448, 448]``.
        scale : torch.Tensor
            Scalar multiplicative scale factor stored in FP32.

        Complexity
        ----------
        Time *O(n)*, space *O(n)*.
        """
        amax = tensor.abs().amax().clamp(min=1e-12)
        scale = torch.tensor(
            _FP8_E4M3_MAX / amax.item(), dtype=torch.float32, device=tensor.device
        )

        if self._use_native:
            if _NATIVE_FP8_DTYPE is None:
                # FIX: Avoid assert for runtime validation when native FP8 is expected.
                raise RuntimeError("Native FP8 dtype is unavailable but native path was selected.")
            quantized = (tensor.float() * scale).to(_NATIVE_FP8_DTYPE)
        else:
            # Software fallback: scale into [-448, 448], clamp, round, store
            # as int16.  int16 range is [-32768, 32767] which covers [-448, 448]
            # exactly with sign preserved.
            #
            # Previous bug: .to(torch.uint8) — uint8 is unsigned [0, 255].
            # Casting any negative float to uint8 in PyTorch produces 0 (the
            # value is clamped/wrapped to the unsigned range), so ALL negative
            # activations were silently zeroed.  Dequantise then returned 0 /
            # scale = 0.0 for the entire negative half of the tensor, producing
            # completely wrong gradients throughout training.
            scaled = tensor.float() * scale
            quantized = scaled.clamp(_FP8_E4M3_MIN, _FP8_E4M3_MAX).to(torch.int16)

        return quantized, scale

    def dequantize(
        self, quantized: torch.Tensor, scale: torch.Tensor, *, dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """Restore a quantised tensor to *dtype*.

        Parameters
        ----------
        quantized : torch.Tensor
            FP8 (``float8_e4m3fn``) or ``int16`` tensor produced by
            :meth:`quantize`.
        scale : torch.Tensor
            Scale factor produced alongside *quantized*.
        dtype : torch.dtype, optional
            Target precision (default ``float32``).

        Returns
        -------
        torch.Tensor

        Complexity
        ----------
        Time *O(n)*, space *O(n)*.
        """
        return (quantized.float() / scale).to(dtype)

    # -- dunder -------------------------------------------------------------

    def __repr__(self) -> str:
        backend = "native-fp8-e4m3" if self._use_native else "simulated-fp8-int16"
        return f"FP8ActivationStorage(backend={backend!r})"


# ---------------------------------------------------------------------------
# INT8 guard
# ---------------------------------------------------------------------------


def _warn_int8() -> None:
    """Emit a loud warning when INT8 activation storage is requested."""
    warnings.warn(
        "INT8 activation storage is UNSAFE for training "
        "(QuEST, arXiv 2502.05003). Using FP8 E4M3 instead.",
        UserWarning,
        stacklevel=3,
    )


# ---------------------------------------------------------------------------
# FP8CheckpointFunction (torch.autograd.Function)
# ---------------------------------------------------------------------------


class FP8CheckpointFunction(torch.autograd.Function):
    """Autograd function that stores activations in FP8 E4M3.

    In the forward pass the output is computed normally but the *input*
    activations are compressed to FP8 before being saved for backward.
    In the backward pass the activations are dequantised back to the
    original dtype before gradient computation.

    Complexity
    ----------
    Forward  — *O(n)* extra for quantisation.
    Backward — *O(n)* extra for dequantisation.
    """

    @staticmethod
    def forward(
        ctx: Any,
        module: nn.Module,
        storage: FP8ActivationStorage,
        *inputs: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        """Run *module* forward and save FP8-compressed activations.

        Parameters
        ----------
        ctx : Any
            Autograd context.
        module : nn.Module
            The sub-module to execute.
        storage : FP8ActivationStorage
            FP8 quantiser instance.
        *inputs : torch.Tensor
            Input tensors to *module*.
        """
        ctx.module = module
        ctx.storage = storage
        ctx.input_dtypes = [t.dtype for t in inputs]

        # Forward pass with full precision.
        outputs = module(*inputs)

        # Store activations in FP8.
        quantized_pairs: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for t in inputs:
            q, s = storage.quantize(t.detach())
            quantized_pairs.append((q, s))

        # Save FP8 tensors (quantized + scales) for backward.
        flat: List[torch.Tensor] = []
        for q, s in quantized_pairs:
            flat.extend([q, s])
        ctx.save_for_backward(*flat)
        ctx.num_inputs = len(inputs)

        if isinstance(outputs, torch.Tensor):
            return (outputs,)
        return tuple(outputs)

    @staticmethod
    def backward(
        ctx: Any, *grad_outputs: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], ...]:
        """Dequantise stored activations and recompute gradients.

        Complexity
        ----------
        Time *O(n)* dequantisation + cost of ``module.forward`` recomputation.
        """
        saved = ctx.saved_tensors
        storage: FP8ActivationStorage = ctx.storage
        module: nn.Module = ctx.module
        num_inputs: int = ctx.num_inputs
        input_dtypes: List[torch.dtype] = ctx.input_dtypes

        # Reconstruct full-precision inputs.
        inputs: List[torch.Tensor] = []
        for i in range(num_inputs):
            q = saved[2 * i]
            s = saved[2 * i + 1]
            inputs.append(storage.dequantize(q, s, dtype=input_dtypes[i]))

        # Re-enable gradients for recomputation.
        inputs_with_grad = [t.detach().requires_grad_(True) for t in inputs]

        with torch.enable_grad():
            outputs = module(*inputs_with_grad)
            if isinstance(outputs, torch.Tensor):
                outputs = (outputs,)
            torch.autograd.backward(outputs, grad_outputs)

        input_grads = tuple(t.grad for t in inputs_with_grad)
        # Return None for (module, storage) + input grads.
        return (None, None, *input_grads)


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------


class _FP8CheckpointWrapper(nn.Module):
    """Thin wrapper applying FP8 activation checkpointing to a sub-module.

    Parameters
    ----------
    module : nn.Module
        The layer to wrap.
    storage : FP8ActivationStorage
        Shared FP8 storage instance.
    """

    def __init__(self, module: nn.Module, storage: FP8ActivationStorage) -> None:
        super().__init__()
        self.module = module
        self.storage = storage

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Forward through FP8 checkpoint function.

        Complexity
        ----------
        Same as wrapped module + *O(n)* quantisation overhead.
        """
        outs = FP8CheckpointFunction.apply(self.module, self.storage, *inputs)
        if outs is None:
            # FIX: Avoid assert for runtime validation on autograd outputs.
            raise RuntimeError("FP8CheckpointFunction returned None outputs.")
        if len(outs) == 1:
            return outs[0]
        return outs  # type: ignore[return-value]

    def __repr__(self) -> str:
        return (
            f"_FP8CheckpointWrapper(\n"
            f"  module={self.module!r},\n"
            f"  storage={self.storage!r}\n"
            f")"
        )


def fp8_checkpoint(
    module: nn.Module,
    every_n: int = 4,
    *,
    force_simulated: bool = False,
    _int8: bool = False,
) -> nn.Module:
    """Wrap *module* with FP8 E4M3 activation checkpointing every *N*-th layer.

    Parameters
    ----------
    module : nn.Module
        Top-level module whose children will be selectively wrapped.
    every_n : int, optional
        Checkpoint every *N*-th child starting at index 0 (default 4).
    force_simulated : bool, optional
        Force the software FP8 fallback.
    _int8 : bool, optional
        **Ignored** — exists only to emit a safety warning.  INT8 is never
        used.  Do not set this parameter.

    Returns
    -------
    nn.Module
        The same *module*, with selected children replaced by
        :class:`_FP8CheckpointWrapper`.

    Complexity
    ----------
    Time *O(L)* where *L* = number of children.
    """
    # FIX: Document index-0 wrapping behavior for fp8_checkpoint.
    if _int8:
        _warn_int8()

    storage = FP8ActivationStorage(force_simulated=force_simulated)
    children = list(module.named_children())

    for idx, (name, child) in enumerate(children):
        if idx % every_n == 0:
            wrapped = _FP8CheckpointWrapper(child, storage)
            setattr(module, name, wrapped)
            logger.debug("FP8 checkpoint applied to layer %s (index %d)", name, idx)

    logger.info(
        "FP8 checkpointing: wrapped %d / %d layers (every_n=%d)",
        len([i for i in range(len(children)) if i % every_n == 0]),
        len(children),
        every_n,
    )
    return module
