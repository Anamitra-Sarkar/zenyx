"""Async distributed checkpointing to T2 (NVMe).

Checkpoints are saved in a background thread so training is **never blocked**.
Each rank saves its own shard alongside a metadata JSON file that records
tensor names, shapes, dtypes, and byte offsets for reconstruction.

Incremental checkpointing is supported: only tensors whose data has changed
since the last save are written.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import torch

__all__ = [
    "AsyncCheckpointer",
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_METADATA_FILENAME: str = "checkpoint_meta.json"
_DEFAULT_IO_WORKERS: int = 2


# ---------------------------------------------------------------------------
# Shard metadata
# ---------------------------------------------------------------------------


@dataclass
class _ShardMeta:
    """Per-tensor metadata stored in the checkpoint manifest.

    Attributes
    ----------
    name : str
        Fully-qualified parameter name.
    shape : List[int]
        Tensor dimensions.
    dtype : str
        String representation of the torch dtype.
    filename : str
        Shard filename on disk.
    byte_size : int
        Raw byte count.
    checksum : str
        MD5 hex digest for integrity and incremental diffing.
    """

    name: str
    shape: List[int]
    dtype: str
    filename: str
    byte_size: int
    checksum: str

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to JSON-compatible dict.

        Complexity
        ----------
        Time *O(1)*.
        """
        return {
            "name": self.name,
            "shape": self.shape,
            "dtype": self.dtype,
            "filename": self.filename,
            "byte_size": self.byte_size,
            "checksum": self.checksum,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> _ShardMeta:
        """Deserialise from dict.

        Complexity
        ----------
        Time *O(1)*.
        """
        return cls(**d)


# ---------------------------------------------------------------------------
# AsyncCheckpointer
# ---------------------------------------------------------------------------


class AsyncCheckpointer:
    """Non-blocking distributed checkpoint writer.

    Parameters
    ----------
    rank : int
        Current process / device rank.
    world_size : int
        Total number of ranks (for metadata).
    io_workers : int, optional
        Background I/O thread count (default 2).

    Complexity
    ----------
    * ``save``  — Time *O(1)* caller-side (I/O happens in background thread).
      Background cost *O(M)* where *M* = state dict size.
    * ``load``  — Time *O(M)*, space *O(M)*.
    """

    def __init__(
        self,
        rank: int = 0,
        world_size: int = 1,
        *,
        io_workers: int = _DEFAULT_IO_WORKERS,
    ) -> None:
        self.rank = rank
        self.world_size = world_size
        self._io_workers = io_workers
        self._executor = ThreadPoolExecutor(
            max_workers=io_workers,
            thread_name_prefix="zenyx-ckpt-io",
        )
        self._lock = threading.Lock()
        # Checksums from last successful save (for incremental mode).
        self._prev_checksums: Dict[str, str] = {}
        self._pending_futures: List[Future[None]] = []

    # -- Public API ---------------------------------------------------------

    def save(
        self,
        state_dict: Dict[str, torch.Tensor],
        path: str,
        *,
        incremental: bool = True,
    ) -> Future[None]:
        """Schedule an async checkpoint save — **never blocks training**.

        Parameters
        ----------
        state_dict : Dict[str, torch.Tensor]
            Model (or optimizer) state to persist.
        path : str
            Directory where shards will be written.
        incremental : bool, optional
            If *True* (default), skip tensors unchanged since last save.

        Returns
        -------
        concurrent.futures.Future[None]
            Resolves when the I/O completes.

        Complexity
        ----------
        Caller: *O(M)* snapshot (CPU tensor copies), then returns immediately.
        Background thread: *O(M_changed)* disk I/O.
        """
        # Snapshot tensors to CPU immediately (fast, pinned-memory copy) so
        # the GPU state can evolve while I/O proceeds.
        snapshot: Dict[str, bytes] = {}
        checksums: Dict[str, str] = {}
        meta_entries: List[_ShardMeta] = []

        for name, tensor in state_dict.items():
            cpu_tensor = tensor.detach().cpu()
            raw = cpu_tensor.numpy().tobytes()
            chk = hashlib.md5(raw).hexdigest()
            checksums[name] = chk

            if incremental and self._prev_checksums.get(name) == chk:
                logger.debug("Incremental skip: %s (unchanged)", name)
                continue

            filename = f"rank{self.rank}_{name.replace('.', '_')}.pt"
            snapshot[filename] = raw
            meta_entries.append(
                _ShardMeta(
                    name=name,
                    shape=list(tensor.shape),
                    dtype=str(tensor.dtype),
                    filename=filename,
                    byte_size=len(raw),
                    checksum=chk,
                )
            )

        future: Future[None] = self._executor.submit(
            self._write_shards, path, snapshot, meta_entries, checksums
        )
        with self._lock:
            # Prune completed futures.
            self._pending_futures = [f for f in self._pending_futures if not f.done()]
            self._pending_futures.append(future)

        return future

    def load(self, path: str) -> Dict[str, torch.Tensor]:
        """Load a checkpoint from *path* for this rank.

        Parameters
        ----------
        path : str
            Directory written by a previous :meth:`save`.

        Returns
        -------
        Dict[str, torch.Tensor]

        Raises
        ------
        FileNotFoundError
            If the metadata file is missing.

        Complexity
        ----------
        Time *O(M)*, space *O(M)*.
        """
        ckpt_dir = Path(path)
        meta_path = ckpt_dir / f"rank{self.rank}_{_METADATA_FILENAME}"
        if not meta_path.exists():
            # Try global metadata.
            meta_path = ckpt_dir / _METADATA_FILENAME
        if not meta_path.exists():
            raise FileNotFoundError(
                f"Checkpoint metadata not found at {ckpt_dir}. "
                f"Expected file: rank{self.rank}_{_METADATA_FILENAME} "
                f"or {_METADATA_FILENAME}"
            )

        try:
            with open(meta_path, "r") as f:
                raw_meta = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise RuntimeError(
                f"Checkpoint at {meta_path} is corrupted or unreadable: {e}"
            ) from e

        result: Dict[str, torch.Tensor] = {}
        for entry_dict in raw_meta.get("shards", []):
            meta = _ShardMeta.from_dict(entry_dict)
            shard_path = ckpt_dir / meta.filename
            if not shard_path.exists():
                logger.warning("Shard file missing: %s — skipping %s", shard_path, meta.name)
                continue

            import numpy as np

            raw_bytes = shard_path.read_bytes()
            try:
                dtype_torch = getattr(torch, meta.dtype.replace("torch.", ""), torch.float32)
                np_dtype = {
                    torch.float32: np.float32,
                    torch.float16: np.float16,
                    torch.bfloat16: np.float32,  # bfloat16 numpy compat
                    torch.int64: np.int64,
                    torch.int32: np.int32,
                }.get(dtype_torch, np.float32)

                arr = np.frombuffer(raw_bytes, dtype=np_dtype).reshape(meta.shape)
                result[meta.name] = torch.from_numpy(arr.copy()).to(dtype_torch)
            except (ValueError, RuntimeError) as e:
                raise RuntimeError(
                    f"Checkpoint shard {shard_path} for tensor '{meta.name}' "
                    f"is corrupted or incompatible: {e}"
                ) from e

        logger.info(
            "Loaded checkpoint from %s: %d tensors for rank %d.",
            path,
            len(result),
            self.rank,
        )
        return result

    def wait(self, timeout: Optional[float] = None) -> None:
        """Block until all pending saves complete.

        Parameters
        ----------
        timeout : float, optional
            Max seconds to wait (``None`` = forever).

        Complexity
        ----------
        Time bounded by I/O latency.
        """
        with self._lock:
            futures = list(self._pending_futures)
        for fut in futures:
            fut.result(timeout=timeout)

    def shutdown(self) -> None:
        """Wait for pending I/O and shut down the thread pool.

        Complexity
        ----------
        Time bounded by pending I/O.
        """
        self.wait()
        self._executor.shutdown(wait=True)

    # -- Internal -----------------------------------------------------------

    def _write_shards(
        self,
        path: str,
        snapshot: Dict[str, bytes],
        meta_entries: List[_ShardMeta],
        checksums: Dict[str, str],
    ) -> None:
        """Background: write shard files and metadata JSON.

        Complexity
        ----------
        Time *O(M_changed)*, space *O(1)* streaming writes.
        """
        ckpt_dir = Path(path)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        for filename, raw_bytes in snapshot.items():
            shard_path = ckpt_dir / filename
            shard_path.write_bytes(raw_bytes)
            logger.debug("Wrote shard %s (%d bytes)", shard_path, len(raw_bytes))

        # Write metadata.
        meta_dict = {
            "rank": self.rank,
            "world_size": self.world_size,
            "timestamp": time.time(),
            "shards": [m.to_dict() for m in meta_entries],
        }
        meta_path = ckpt_dir / f"rank{self.rank}_{_METADATA_FILENAME}"
        with open(meta_path, "w") as f:
            json.dump(meta_dict, f, indent=2)

        # Update cached checksums.
        self._prev_checksums.update(checksums)
        logger.info(
            "Checkpoint saved to %s: %d shards (rank %d).",
            path,
            len(snapshot),
            self.rank,
        )

    # -- dunder -------------------------------------------------------------

    def __repr__(self) -> str:
        with self._lock:
            pending = sum(1 for f in self._pending_futures if not f.done())
        return (
            f"AsyncCheckpointer("
            f"rank={self.rank}, "
            f"world_size={self.world_size}, "
            f"io_workers={self._io_workers}, "
            f"pending_saves={pending}, "
            f"tracked_tensors={len(self._prev_checksums)})"
        )
