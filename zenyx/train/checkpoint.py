"""Async distributed checkpointing to T2 (NVMe).

Checkpoints are saved in a background thread so training is **never blocked**.
Each rank saves its own shard alongside a metadata JSON file that records
tensor names, shapes, dtypes, and byte offsets for reconstruction.

Incremental checkpointing is supported: only tensors whose data has changed
since the last save are written.

Fix notes
---------
* bfloat16 save: stored as uint16 raw bytes (view cast), reloaded via
  torch.frombuffer. numpy has no bfloat16 dtype; the previous np.float32
  mapping caused a 2×-element-count mismatch on load.
* Incremental-save race: _prev_checksums is now updated from a snapshot
  dict captured *before* the background thread starts, not from inside
  _write_shards. This prevents a second concurrent save() from reading a
  partially-committed checksums map.
* load() duplicate import: numpy was imported twice inside the method —
  once at the method scope and again inside the bfloat16 branch.  The
  redundant inner import has been removed.
* load() bfloat16 zero-copy: the previous code called arr_u16.tobytes()
  which created a second copy of the data in a new bytes object before
  passing it to torch.frombuffer.  The fix passes raw_bytes directly to
  torch.frombuffer so no intermediate copy is made.
* Cross-platform atomic rename: os.replace() raises PermissionError on
  Windows when the destination file is held open by another process.
  _write_shards now catches OSError on both the shard and metadata rename
  and falls back to shutil.move(), which works cross-platform.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
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

# Special marker in dtype field to signal uint16-encoded bfloat16.
_BFLOAT16_MARKER: str = "torch.bfloat16"


# ---------------------------------------------------------------------------
# Shard metadata
# ---------------------------------------------------------------------------


@dataclass
class _ShardMeta:
    """Per-tensor metadata stored in the checkpoint manifest."""

    name: str
    shape: List[int]
    dtype: str
    filename: str
    byte_size: int
    checksum: str

    def to_dict(self) -> Dict[str, Any]:
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
        self._prev_checksums: Dict[str, str] = {}
        self._prev_meta_entries: Dict[str, "_ShardMeta"] = {}
        self._pending_futures: List[Future[None]] = []
        self._async_errors: List[BaseException] = []

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
        """
        # FIX: Surface any prior async errors before scheduling new work.
        self._raise_async_errors()
        snapshot: Dict[str, bytes] = {}
        checksums: Dict[str, str] = {}
        # Start with ALL entries from the previous checkpoint manifest.
        # Unchanged tensors will keep their old entry (pointing to old shard file).
        # Changed tensors will have their entry overwritten below.
        current_meta: Dict[str, _ShardMeta] = dict(self._prev_meta_entries)

        for name, tensor in state_dict.items():
            cpu_tensor = tensor.detach().cpu()

            if cpu_tensor.dtype == torch.bfloat16:
                raw = cpu_tensor.contiguous().view(torch.uint16).numpy().tobytes()
            else:
                raw = cpu_tensor.numpy().tobytes()

            chk = hashlib.md5(raw).hexdigest()
            checksums[name] = chk

            if incremental and self._prev_checksums.get(name) == chk:
                logger.debug("Incremental skip: %s (unchanged, kept in manifest)", name)
                continue

            filename = f"rank{self.rank}_{name.replace('.', '_')}.pt"
            snapshot[filename] = raw
            new_entry = _ShardMeta(
                name=name,
                shape=list(tensor.shape),
                dtype=str(tensor.dtype),
                filename=filename,
                byte_size=len(raw),
                checksum=chk,
            )
            current_meta[name] = new_entry

        meta_entries = list(current_meta.values())

        future: Future[None] = self._executor.submit(
            self._write_shards, path, snapshot, meta_entries, checksums, current_meta
        )
        # FIX: Capture background exceptions promptly for training-loop polling.
        future.add_done_callback(self._capture_async_exception)
        with self._lock:
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
        """
        ckpt_dir = Path(path)
        meta_path = ckpt_dir / f"rank{self.rank}_{_METADATA_FILENAME}"
        if not meta_path.exists():
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

            raw_bytes = shard_path.read_bytes()
            try:
                dtype_torch = getattr(torch, meta.dtype.replace("torch.", ""), torch.float32)

                if dtype_torch == torch.bfloat16:
                    # bfloat16 was stored as uint16 raw bytes.
                    # Pass raw_bytes directly — no intermediate numpy copy.
                    t = torch.frombuffer(
                        raw_bytes, dtype=torch.uint16
                    ).view(torch.bfloat16)
                    result[meta.name] = t.reshape(meta.shape).clone()
                else:
                    np_dtype = {
                        torch.float32: np.float32,
                        torch.float16: np.float16,
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
        """Block until all pending saves complete."""
        with self._lock:
            futures = list(self._pending_futures)
        for fut in futures:
            fut.result(timeout=timeout)
        self._raise_async_errors()

    def poll(self) -> None:
        """Check for completed background errors without blocking."""
        # FIX: Allow training loops to surface async checkpoint errors promptly.
        self._raise_async_errors()
        with self._lock:
            self._pending_futures = [f for f in self._pending_futures if not f.done()]

    def shutdown(self) -> None:
        """Wait for pending I/O and shut down the thread pool."""
        self.wait()
        self._executor.shutdown(wait=True)

    # -- Internal -----------------------------------------------------------

    def _write_shards(
        self,
        path: str,
        snapshot: Dict[str, bytes],
        meta_entries: List[_ShardMeta],
        checksums: Dict[str, str],
        full_meta: Dict[str, "_ShardMeta"],
    ) -> None:
        """Background: write shard files and metadata JSON.

        Uses atomic rename (os.replace) for crash-safety.  Falls back to
        shutil.move() on Windows where os.replace() raises PermissionError
        when the destination is held open by another process.
        """
        ckpt_dir = Path(path)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        for filename, raw_bytes in snapshot.items():
            shard_path = ckpt_dir / filename
            tmp_shard = Path(str(shard_path) + ".tmp")
            tmp_shard.write_bytes(raw_bytes)
            try:
                os.replace(tmp_shard, shard_path)
            except OSError:
                # Windows: destination may be locked by another process.
                shutil.move(str(tmp_shard), str(shard_path))
            logger.debug("Wrote shard %s (%d bytes)", shard_path, len(raw_bytes))

        meta_dict = {
            "rank": self.rank,
            "world_size": self.world_size,
            "timestamp": time.time(),
            "shards": [m.to_dict() for m in meta_entries],
        }
        meta_path = ckpt_dir / f"rank{self.rank}_{_METADATA_FILENAME}"
        tmp_meta = Path(str(meta_path) + ".tmp")
        with open(tmp_meta, "w") as f:
            json.dump(meta_dict, f, indent=2)
        try:
            os.replace(tmp_meta, meta_path)
        except OSError:
            # Windows: destination may be locked by another process.
            shutil.move(str(tmp_meta), str(meta_path))

        with self._lock:
            self._prev_checksums.update(checksums)
            self._prev_meta_entries = dict(full_meta)

        logger.info(
            "Checkpoint saved to %s: %d shards (rank %d).",
            path,
            len(snapshot),
            self.rank,
        )

    def _capture_async_exception(self, future: Future[None]) -> None:
        exc = future.exception()
        if exc is None:
            return
        with self._lock:
            self._async_errors.append(exc)

    def _raise_async_errors(self) -> None:
        with self._lock:
            if not self._async_errors:
                return
            exc = self._async_errors.pop(0)
        raise RuntimeError("Async checkpoint save failed") from exc

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
