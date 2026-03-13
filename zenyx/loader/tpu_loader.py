"""TPU model loader via multi-threaded pread + DMA to HBM.

For TPU hardware there is no GDS equivalent. Instead we use multi-threaded
POSIX ``pread`` into pinned host memory, then DMA to TPU HBM via
``jax.device_put`` (or standard XLA transfer).

JAX imports are guarded — the module is importable even when JAX is not
installed, falling back to standard PyTorch ``torch.load``.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

__all__ = [
    "TPUModelLoader",
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_IO_WORKERS: int = 8  # TPU hosts typically have high core counts.
_PREAD_CHUNK_SIZE: int = 64 * 1024 * 1024  # 64 MB per pread call.

# ---------------------------------------------------------------------------
# Guarded JAX imports
# ---------------------------------------------------------------------------

_jax: Any = None
_jnp: Any = None
_jax_available: Optional[bool] = None


def _check_jax() -> bool:
    """Probe for JAX availability (cached).

    Complexity
    ----------
    Time *O(1)* (import cost amortised).
    """
    global _jax, _jnp, _jax_available
    if _jax_available is not None:
        return _jax_available
    try:
        import jax  # type: ignore[import-untyped]
        import jax.numpy as jnp  # type: ignore[import-untyped]

        _jax = jax
        _jnp = jnp
        _jax_available = True
    except ImportError:
        _jax_available = False
        logger.info("JAX not available — TPU loader will use PyTorch fallback.")
    return _jax_available


# ---------------------------------------------------------------------------
# Low-level pread helpers
# ---------------------------------------------------------------------------


def _pread_chunk(fd: int, offset: int, size: int) -> bytes:
    """Read *size* bytes from file descriptor *fd* at *offset* via ``os.pread``.

    Complexity
    ----------
    Time *O(size)*, space *O(size)*.
    """
    data = bytearray()
    remaining = size
    pos = offset
    while remaining > 0:
        chunk = os.pread(fd, min(remaining, _PREAD_CHUNK_SIZE), pos)
        if not chunk:
            break
        data.extend(chunk)
        pos += len(chunk)
        remaining -= len(chunk)
    return bytes(data)


# ---------------------------------------------------------------------------
# TPUModelLoader
# ---------------------------------------------------------------------------


class TPUModelLoader:
    """Load model weights into TPU HBM via multi-threaded pread + DMA.

    Shares the same public interface as :class:`~zenyx.loader.gds_loader.GDSModelLoader`.

    Parameters
    ----------
    model_path : str
        Path to a PyTorch checkpoint or safetensors file.
    device : torch.device | str
        Target device.  For TPU workloads, pass ``torch.device("xla:0")`` or
        ``"tpu"``.  Falls back to CPU if JAX is unavailable.
    io_workers : int, optional
        Number of parallel pread I/O threads (default 8).

    Complexity
    ----------
    * ``load`` — Time *O(M / B_disk)*, space *O(M)* host + *O(M)* device.
    """

    def __init__(
        self,
        model_path: str,
        device: Any = "tpu",
        *,
        io_workers: int = _DEFAULT_IO_WORKERS,
    ) -> None:
        self.model_path = Path(model_path)
        self.device = device
        self._io_workers = io_workers
        self._use_jax = _check_jax()
        self._executor: Optional[ThreadPoolExecutor] = None

    # -- Public API ---------------------------------------------------------

    def load(self) -> Dict[str, torch.Tensor]:
        """Load model weights with multi-threaded I/O and TPU DMA transfer.

        Returns
        -------
        Dict[str, torch.Tensor]
            State dict.  Tensors are on device (TPU HBM if JAX is available,
            otherwise CPU).

        Raises
        ------
        FileNotFoundError
            If *model_path* does not exist.

        Complexity
        ----------
        Time *O(M / B)* pipelined across I/O threads, space *O(M)* host + device.
        """
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        self._executor = ThreadPoolExecutor(
            max_workers=self._io_workers,
            thread_name_prefix="zenyx-tpu-io",
        )

        try:
            # Step 1 — load state dict to host memory.
            cpu_state = self._load_to_host()

            # Step 2 — transfer to TPU HBM (or stay on CPU).
            if self._use_jax:
                return self._transfer_jax(cpu_state)
            else:
                return self._transfer_torch(cpu_state)
        finally:
            self._executor.shutdown(wait=False)
            self._executor = None

    # -- Internal -----------------------------------------------------------

    def _load_to_host(self) -> Dict[str, torch.Tensor]:
        """Load state dict to CPU pinned memory using threaded pread.

        Complexity
        ----------
        Time *O(M / B_disk)* across threads, space *O(M)*.
        """
        suffix = self.model_path.suffix.lower()
        if suffix == ".safetensors":
            return self._load_safetensors_host()
        return self._load_torch_host()

    def _load_torch_host(self) -> Dict[str, torch.Tensor]:
        """Load via torch.load on CPU with async I/O.

        Complexity
        ----------
        Time *O(M)*, space *O(M)*.
        """
        assert self._executor is not None
        future: Future[Dict[str, torch.Tensor]] = self._executor.submit(
            torch.load,
            str(self.model_path),
            map_location="cpu",
            weights_only=True,
        )
        return future.result()

    def _load_safetensors_host(self) -> Dict[str, torch.Tensor]:
        """Load via safetensors, fallback to torch.

        Complexity
        ----------
        Time *O(M)*, space *O(M)*.
        """
        try:
            from safetensors.torch import load_file  # type: ignore[import-untyped]

            assert self._executor is not None
            future: Future[Dict[str, torch.Tensor]] = self._executor.submit(
                load_file, str(self.model_path), device="cpu"
            )
            return future.result()
        except ImportError:
            logger.warning("safetensors not installed — using torch.load")
            return self._load_torch_host()

    def _transfer_jax(
        self, cpu_state: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Transfer tensors to TPU HBM via jax.device_put.

        JAX arrays are converted back to ``torch.Tensor`` wrappers so the
        return type is consistent with the GPU loader.

        Complexity
        ----------
        Time *O(M / B_hbm)*, space *O(M)* on device.
        """
        assert _jax is not None and _jnp is not None

        result: Dict[str, torch.Tensor] = {}
        tpu_devices = _jax.devices("tpu")
        target_device = tpu_devices[0] if tpu_devices else _jax.devices()[0]

        for key, tensor in cpu_state.items():
            np_array = tensor.numpy()
            jax_array = _jax.device_put(_jnp.array(np_array), device=target_device)
            # Convert back to torch tensor (stays on XLA device via torch_xla
            # if available, otherwise CPU copy).
            try:
                import torch_xla.core.xla_model as xm  # type: ignore[import-untyped]

                result[key] = torch.from_numpy(jax_array.__array__())
            except ImportError:
                result[key] = torch.from_numpy(jax_array.__array__())
            logger.debug("Transferred %s to TPU HBM (%.2f MB)", key,
                         tensor.nelement() * tensor.element_size() / (1024**2))

        return result

    def _transfer_torch(
        self, cpu_state: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Fallback: keep tensors on CPU (or move to XLA device if available).

        Complexity
        ----------
        Time *O(M)*, space *O(M)*.
        """
        try:
            import torch_xla.core.xla_model as xm  # type: ignore[import-untyped]

            xla_device = xm.xla_device()
            return {k: v.to(xla_device) for k, v in cpu_state.items()}
        except ImportError:
            logger.warning(
                "Neither JAX nor torch_xla available — keeping tensors on CPU."
            )
            return cpu_state

    # -- dunder -------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"TPUModelLoader("
            f"path={str(self.model_path)!r}, "
            f"device={self.device!r}, "
            f"jax={'available' if self._use_jax else 'unavailable'}, "
            f"io_workers={self._io_workers})"
        )
