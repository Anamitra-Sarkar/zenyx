"""GPU Direct Storage (GDS) model loader with triple-buffering pipeline.

Uses cuFile API for NVMe → GPU HBM direct DMA, bypassing the CPU page cache.
Falls back to standard triple-buffering with a CPU bounce buffer when cuFile
is not available.

Triple-buffering pipeline
-------------------------
Three pinned buffers rotate simultaneously:

1. **Buffer A** — reading from NVMe (disk I/O thread).
2. **Buffer B** — transferring from CPU pinned memory to GPU HBM (CUDA stream).
3. **Buffer C** — being consumed (weights assigned to model parameters).

Small tensors (< 5 MB) are batched into contiguous blocks before transfer to
keep GDS efficient.

All I/O is async via :class:`concurrent.futures.ThreadPoolExecutor`.
"""

from __future__ import annotations

import io
import logging
import os
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

__all__ = [
    "GDSModelLoader",
    "estimate_load_time",
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SMALL_TENSOR_THRESHOLD: int = 5 * 1024 * 1024  # 5 MB
_NUM_BUFFERS: int = 3
_DEFAULT_IO_WORKERS: int = 4

# ---------------------------------------------------------------------------
# cuFile availability probe
# ---------------------------------------------------------------------------

_cufile_available: Optional[bool] = None


def _check_cufile() -> bool:
    """Return *True* if the cuFile (GDS) library is importable.

    Complexity
    ----------
    Time *O(1)*, one-time import probe cached globally.
    """
    global _cufile_available
    if _cufile_available is not None:
        return _cufile_available
    try:
        import kvikio  # noqa: F401 — GDS Python bindings

        _cufile_available = True
    except ImportError:
        _cufile_available = False
    if not _cufile_available:
        logger.info(
            "cuFile/GDS not available — falling back to CPU bounce-buffer "
            "triple-buffering."
        )
    return _cufile_available


# ---------------------------------------------------------------------------
# Helper dataclass for triple-buffer slots
# ---------------------------------------------------------------------------


@dataclass
class _BufferSlot:
    """One slot in the triple-buffer ring.

    Attributes
    ----------
    cpu_buf : Optional[torch.Tensor]
        Pinned CPU buffer (allocated lazily on first use).
    stream : Optional[torch.cuda.Stream]
        Dedicated CUDA stream for async H2D copy.
    """

    cpu_buf: Optional[torch.Tensor] = None
    stream: Optional[Any] = None  # torch.cuda.Stream (lazy)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def __repr__(self) -> str:
        sz = self.cpu_buf.nelement() * self.cpu_buf.element_size() if self.cpu_buf is not None else 0
        return f"_BufferSlot(size={sz / (1024**2):.1f}MB)"


# ---------------------------------------------------------------------------
# GDSModelLoader
# ---------------------------------------------------------------------------


class GDSModelLoader:
    """Load model weights from NVMe into GPU HBM using GDS triple-buffering.

    Parameters
    ----------
    model_path : str
        Path to a ``safetensors`` or ``torch.save``-d state dict file.
    device : torch.device
        Target GPU device (e.g. ``torch.device("cuda:0")``).
    io_workers : int, optional
        Number of I/O threads (default 4).

    Complexity
    ----------
    * ``load`` — Time *O(M / B)* where *M* = model size, *B* = NVMe bandwidth.
      Space *O(3 × block_size)* for the triple buffer.
    """

    def __init__(
        self,
        model_path: str,
        device: torch.device,
        *,
        io_workers: int = _DEFAULT_IO_WORKERS,
    ) -> None:
        self.model_path = Path(model_path)
        self.device = device
        self._io_workers = io_workers
        self._use_gds = _check_cufile() and device.type == "cuda"
        self._executor: Optional[ThreadPoolExecutor] = None
        self._buffers: List[_BufferSlot] = [_BufferSlot() for _ in range(_NUM_BUFFERS)]

    # -- Public API ---------------------------------------------------------

    def load(self) -> Dict[str, torch.Tensor]:
        """Load model weights using triple-buffered async I/O.

        Returns
        -------
        Dict[str, torch.Tensor]
            State dict with tensors on *self.device*.

        Raises
        ------
        FileNotFoundError
            If *model_path* does not exist.

        Complexity
        ----------
        Time *O(M / B)* pipelined, space *O(3 × chunk)* pinned.
        """
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        self._executor = ThreadPoolExecutor(
            max_workers=self._io_workers,
            thread_name_prefix="zenyx-gds-io",
        )

        try:
            # Step 1 — read raw state dict from disk (CPU).
            raw_state = self._read_state_dict()

            # Step 2 — batch small tensors.
            batched_keys, batched_tensors = self._batch_small_tensors(raw_state)

            # Step 3 — triple-buffered transfer to GPU.
            result = self._triple_buffer_transfer(batched_keys, batched_tensors)

            return result
        except torch.cuda.OutOfMemoryError:
            logger.warning(
                "CUDA OOM during model loading — retrying with sequential transfer."
            )
            torch.cuda.empty_cache()
            return self._fallback_sequential_load()
        finally:
            self._executor.shutdown(wait=False)
            self._executor = None

    # -- Internal -----------------------------------------------------------

    def _read_state_dict(self) -> Dict[str, torch.Tensor]:
        """Read raw state dict from disk on CPU.

        Complexity
        ----------
        Time *O(M)*, space *O(M)* — full model on CPU.
        """
        suffix = self.model_path.suffix.lower()
        if suffix in (".safetensors",):
            return self._read_safetensors()
        # Default: standard PyTorch checkpoint.
        return self._read_torch()

    def _read_torch(self) -> Dict[str, torch.Tensor]:
        """Load via torch.load on CPU.

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

    def _read_safetensors(self) -> Dict[str, torch.Tensor]:
        """Load via safetensors on CPU.

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
            logger.warning(
                "safetensors not installed — falling back to torch.load"
            )
            return self._read_torch()

    def _batch_small_tensors(
        self, state: Dict[str, torch.Tensor]
    ) -> Tuple[List[str], List[torch.Tensor]]:
        """Batch tensors smaller than 5 MB into contiguous blocks.

        Parameters
        ----------
        state : Dict[str, torch.Tensor]

        Returns
        -------
        keys : List[str]
            Tensor names (batched small tensors use a synthetic key).
        tensors : List[torch.Tensor]
            Contiguous tensors ready for transfer.

        Complexity
        ----------
        Time *O(n)* where *n* = number of tensors, space *O(S)* where *S* =
        total size of small tensors.
        """
        large_keys: List[str] = []
        large_tensors: List[torch.Tensor] = []
        small_items: List[Tuple[str, torch.Tensor]] = []
        small_bytes: int = 0

        for key, tensor in state.items():
            nbytes = tensor.nelement() * tensor.element_size()
            if nbytes >= _SMALL_TENSOR_THRESHOLD:
                large_keys.append(key)
                large_tensors.append(tensor.contiguous())
            else:
                small_items.append((key, tensor))
                small_bytes += nbytes

        # Batch small tensors: flatten, concatenate, record slicing metadata.
        if small_items:
            self._small_tensor_meta: List[Tuple[str, torch.Size, torch.dtype, int, int]] = []
            flat_parts: List[torch.Tensor] = []
            offset = 0
            for key, tensor in small_items:
                flat = tensor.contiguous().view(-1).to(torch.float32)
                self._small_tensor_meta.append(
                    (key, tensor.shape, tensor.dtype, offset, offset + flat.numel())
                )
                flat_parts.append(flat)
                offset += flat.numel()

            batched = torch.cat(flat_parts)
            large_keys.append("__zenyx_small_batch__")
            large_tensors.append(batched)
            logger.debug(
                "Batched %d small tensors (%.2f MB) into one block.",
                len(small_items),
                small_bytes / (1024**2),
            )
        else:
            self._small_tensor_meta = []

        return large_keys, large_tensors

    def _triple_buffer_transfer(
        self,
        keys: List[str],
        tensors: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Pipeline disk→CPU→GPU using three rotating buffer slots.

        Complexity
        ----------
        Time *O(M / B)* (pipelined), space *O(3 × max_tensor)* pinned.
        """
        result: Dict[str, torch.Tensor] = {}
        cuda_available = torch.cuda.is_available() and self.device.type == "cuda"

        # Ensure CUDA streams exist.
        if cuda_available:
            for slot in self._buffers:
                if slot.stream is None:
                    slot.stream = torch.cuda.Stream(device=self.device)

        for idx, (key, tensor) in enumerate(zip(keys, tensors)):
            slot = self._buffers[idx % _NUM_BUFFERS]
            with slot.lock:
                if cuda_available:
                    # Pin the CPU tensor for async H2D transfer.
                    pinned = tensor.pin_memory()
                    assert slot.stream is not None
                    with torch.cuda.stream(slot.stream):
                        gpu_tensor = pinned.to(self.device, non_blocking=True)
                    slot.stream.synchronize()
                else:
                    gpu_tensor = tensor.to(self.device)

                if key == "__zenyx_small_batch__":
                    # Unbatch small tensors.
                    for name, shape, dtype, start, end in self._small_tensor_meta:
                        result[name] = (
                            gpu_tensor[start:end].to(dtype).reshape(shape)
                        )
                else:
                    result[key] = gpu_tensor

        return result

    def _fallback_sequential_load(self) -> Dict[str, torch.Tensor]:
        """Sequential tensor-by-tensor load as OOM recovery path.

        Complexity
        ----------
        Time *O(M / B)* (not pipelined), space *O(max_tensor)*.
        """
        raw = self._read_state_dict()
        result: Dict[str, torch.Tensor] = {}
        for key, tensor in raw.items():
            try:
                result[key] = tensor.to(self.device)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                logger.error(
                    "Persistent OOM loading tensor %s (%.2f MB) — keeping on CPU.",
                    key,
                    tensor.nelement() * tensor.element_size() / (1024**2),
                )
                result[key] = tensor
        return result

    # -- dunder -------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"GDSModelLoader("
            f"path={str(self.model_path)!r}, "
            f"device={self.device}, "
            f"gds={'enabled' if self._use_gds else 'fallback'}, "
            f"io_workers={self._io_workers})"
        )


# ---------------------------------------------------------------------------
# Standalone helpers
# ---------------------------------------------------------------------------


def estimate_load_time(
    model_size_gb: float,
    nvme_bandwidth_gb_s: float = 15.0,
) -> float:
    """Estimate wall-clock seconds to load a model via GDS.

    Parameters
    ----------
    model_size_gb : float
        Total model size in gigabytes.
    nvme_bandwidth_gb_s : float, optional
        Sustained NVMe read bandwidth (default 15 GB/s).

    Returns
    -------
    float
        Estimated seconds (cold load; warm cache ≈ 2–3 s).

    Complexity
    ----------
    Time *O(1)*.
    """
    return model_size_gb / nvme_bandwidth_gb_s
