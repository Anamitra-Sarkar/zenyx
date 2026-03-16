"""Ring communication primitives for distributed ring attention.

Provides asynchronous send/recv over a ring topology using ``torch.distributed``
P2P operations.  Supports double-buffered KV cache transfers for compute/comm
overlap.

Complexity
----------
Each ``ring_send_recv`` call is O(numel) data transfer, fully async when a
CUDA stream is provided.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Tuple

import torch
import torch.distributed as dist

from zenyx.ops.comm.topology import Topology, LinkType

__all__ = ["RingCommunicator"]

logger = logging.getLogger(__name__)


class RingCommunicator:
    """Manages ring send/recv for distributed ring attention.

    Builds an optimal ring order from the hardware topology and exposes
    async P2P primitives for tensor exchange.

    Parameters
    ----------
    topology : Topology
        Hardware topology descriptor.
    process_group : ProcessGroup | None
        Torch distributed process group.  ``None`` → default group.

    Complexity
    ----------
    Construction: O(D log D) for ring order computation.
    """

    def __init__(
        self,
        topology: Topology,
        process_group: Optional[Any] = None,
    ) -> None:
        self._topology = topology
        self._process_group = process_group

        if dist.is_initialized():
            self._world_size: int = dist.get_world_size(group=process_group)
            self._rank: int = dist.get_rank(group=process_group)
        else:
            self._world_size = 1
            self._rank = 0

        self._ring_order: List[int] = self._compute_ring_order()

        # Positions in the ring
        ring_pos = self._ring_order.index(self._rank) if self._rank in self._ring_order else 0
        self._next_rank: int = self._ring_order[(ring_pos + 1) % self._world_size]
        self._prev_rank: int = self._ring_order[(ring_pos - 1) % self._world_size]

        logger.debug(
            "RingCommunicator: rank=%d, next=%d, prev=%d, ring=%s",
            self._rank,
            self._next_rank,
            self._prev_rank,
            self._ring_order,
        )

        # Async send/recv state (initialised to prevent hasattr checks)
        self._send_req: Optional[Any] = None
        self._recv_req: Optional[Any] = None
        self._recv_buffer: Optional[torch.Tensor] = None
        self._step_counter: int = 0

    def __repr__(self) -> str:
        return (
            f"RingCommunicator(rank={self._rank}, world={self._world_size}, "
            f"next={self._next_rank}, prev={self._prev_rank}, "
            f"interconnect={self._topology.interconnect_type.value})"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def rank(self) -> int:
        """Current process rank."""
        return self._rank

    @property
    def world_size(self) -> int:
        """Total number of ranks in the ring."""
        return self._world_size

    def get_ring_order(self) -> List[int]:
        """Return the ring order (list of ranks).

        Returns
        -------
        list[int]
            Optimal ring traversal order maximising bandwidth.

        Complexity
        ----------
        Time : O(1) — precomputed at construction.
        """
        return list(self._ring_order)

    def min_chunk_size(self) -> int:
        """Return ``S_min``: minimum sequence chunk size for ring attention.

        Returns
        -------
        int
            Minimum tokens per ring chunk.
        """
        return self._topology.s_min

    def ring_send_recv(
        self,
        send_tensor: torch.Tensor,
        recv_buffer: torch.Tensor,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> torch.Tensor:
        """Send a tensor to the next rank and receive from the previous.

        Parameters
        ----------
        send_tensor : Tensor
            Data to send forward in the ring.
        recv_buffer : Tensor
            Pre-allocated buffer for receiving data.
        stream : torch.cuda.Stream | None
            Optional CUDA stream for async execution.

        Returns
        -------
        Tensor
            ``recv_buffer`` filled with data from the previous rank.

        Complexity
        ----------
        Time : O(numel) data transfer (async if stream provided).
        """
        if self._world_size <= 1:
            recv_buffer.copy_(send_tensor)
            return recv_buffer

        if stream is not None:
            with torch.cuda.stream(stream):
                return self._do_p2p_send_recv(send_tensor, recv_buffer)
        return self._do_p2p_send_recv(send_tensor, recv_buffer)

    def ring_send_recv_kv(
        self,
        send_k: torch.Tensor,
        send_v: torch.Tensor,
        recv_k_buffer: torch.Tensor,
        recv_v_buffer: torch.Tensor,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Ring send/recv specialised for KV cache pairs.

        Parameters
        ----------
        send_k, send_v : Tensor
            Key and value tensors to send.
        recv_k_buffer, recv_v_buffer : Tensor
            Pre-allocated receive buffers.
        stream : torch.cuda.Stream | None
            Optional CUDA stream for async execution.

        Returns
        -------
        tuple[Tensor, Tensor]
            ``(recv_k_buffer, recv_v_buffer)`` filled with received data.

        Complexity
        ----------
        Time : O(numel_k + numel_v) data transfer.
        """
        if self._world_size <= 1:
            recv_k_buffer.copy_(send_k)
            recv_v_buffer.copy_(send_v)
            return recv_k_buffer, recv_v_buffer

        if stream is not None:
            with torch.cuda.stream(stream):
                return self._do_p2p_send_recv_kv(
                    send_k, send_v, recv_k_buffer, recv_v_buffer
                )
        return self._do_p2p_send_recv_kv(
            send_k, send_v, recv_k_buffer, recv_v_buffer
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _do_p2p_send_recv(
        self, send_tensor: torch.Tensor, recv_buffer: torch.Tensor
    ) -> torch.Tensor:
        """Execute P2P send/recv via ``torch.distributed.batch_isend_irecv``.

        Complexity
        ----------
        Time : O(numel) transfer + O(1) latency.
        """
        ops: List[dist.P2POp] = [
            dist.P2POp(dist.isend, send_tensor, self._next_rank, group=self._process_group),
            dist.P2POp(dist.irecv, recv_buffer, self._prev_rank, group=self._process_group),
        ]
        reqs = dist.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()
        return recv_buffer

    def _do_p2p_send_recv_kv(
        self,
        send_k: torch.Tensor,
        send_v: torch.Tensor,
        recv_k_buffer: torch.Tensor,
        recv_v_buffer: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Execute batched P2P send/recv for KV pairs.

        Complexity
        ----------
        Time : O(numel_k + numel_v) transfer.
        """
        ops: List[dist.P2POp] = [
            dist.P2POp(dist.isend, send_k, self._next_rank, group=self._process_group),
            dist.P2POp(dist.isend, send_v, self._next_rank, group=self._process_group),
            dist.P2POp(dist.irecv, recv_k_buffer, self._prev_rank, group=self._process_group),
            dist.P2POp(dist.irecv, recv_v_buffer, self._prev_rank, group=self._process_group),
        ]
        reqs = dist.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()
        return recv_k_buffer, recv_v_buffer

    def _compute_ring_order(self) -> List[int]:
        """Compute optimal ring traversal order from topology.

        For NVLink / ICI, orders ranks to maximise bandwidth by following
        high-bandwidth links.  For PCIe or single-device, uses natural order.

        Returns
        -------
        list[int]
            Ring order as a permutation of ranks ``[0, world_size)``.

        Complexity
        ----------
        Time : O(D²) greedy nearest-neighbour on bandwidth graph.
        """
        n = self._world_size
        if n <= 1:
            return list(range(n))

        links = self._topology.links
        if not links:
            return list(range(n))

        # Build bandwidth adjacency: bw[i][j] = max bandwidth i→j
        bw: dict[Tuple[int, int], float] = {}
        for link in links:
            key = (link.src_device, link.dst_device)
            bw[key] = max(bw.get(key, 0.0), link.bandwidth_gb_s)

        # Greedy ring construction starting from rank 0
        visited = {0}
        order = [0]
        current = 0

        for _ in range(n - 1):
            best_next = -1
            best_bw = -1.0
            for candidate in range(n):
                if candidate in visited:
                    continue
                link_bw = bw.get((current, candidate), 0.0)
                if link_bw > best_bw:
                    best_bw = link_bw
                    best_next = candidate
            if best_next == -1:
                # No link info — pick smallest unvisited
                for candidate in range(n):
                    if candidate not in visited:
                        best_next = candidate
                        break
            visited.add(best_next)
            order.append(best_next)
            current = best_next

        return order

    # ------------------------------------------------------------------
    # Non-blocking send/recv API (per Phase 3 spec)
    # ------------------------------------------------------------------

    def send_kv(self, kv: torch.Tensor) -> None:
        """Non-blocking send of KV block to next rank in ring.

        Initiates an ``isend`` to the next rank in the ring.  The caller
        must later call :meth:`wait_send` before reusing the buffer.

        Parameters
        ----------
        kv : Tensor
            KV block to send.

        Raises
        ------
        RuntimeError
            If a previous send has not yet been waited on.

        Time complexity:  O(1) to initiate.
        Space complexity: O(1).
        """
        if self._world_size <= 1:
            return
        if self._send_req is not None:
            raise RuntimeError(
                "send_kv() called while a previous send is still pending. "
                "Call wait_send() first to avoid buffer corruption."
            )
        self._send_req = dist.isend(kv, self._next_rank, group=self._process_group)

    def recv_kv(self, buffer: torch.Tensor) -> None:
        """Non-blocking recv of KV block from previous rank in ring.

        Initiates an ``irecv`` from the previous rank.  The caller must
        later call :meth:`wait_recv` to block until the transfer completes.

        Parameters
        ----------
        buffer : Tensor
            Pre-allocated buffer to receive into.

        Time complexity:  O(1) to initiate.
        Space complexity: O(1).
        """
        if self._world_size <= 1:
            return
        self._recv_buffer = buffer
        self._recv_req = dist.irecv(buffer, self._prev_rank, group=self._process_group)

    def wait_send(self) -> None:
        """Block until the pending send completes.

        Time complexity:  O(numel) — waits for transfer.
        Space complexity: O(1).
        """
        if self._world_size <= 1:
            return
        if self._send_req is not None:
            self._send_req.wait()
            self._send_req = None

    def wait_recv(self) -> torch.Tensor:
        """Block until the pending recv completes.

        Returns
        -------
        Tensor
            The received KV tensor (same as the buffer passed to :meth:`recv_kv`).

        Time complexity:  O(numel) — waits for transfer.
        Space complexity: O(1).
        """
        if self._world_size <= 1:
            return self._recv_buffer if self._recv_buffer is not None else torch.empty(0)
        if self._recv_req is not None:
            self._recv_req.wait()
            self._recv_req = None
        if self._recv_buffer is None:
            raise RuntimeError("wait_recv() called before recv_kv() — no pending receive.")
        return self._recv_buffer

    def step(self) -> None:
        """Advance ring position by one step (increment step counter).

        Time complexity:  O(1).
        Space complexity: O(1).
        """
        self._step_counter += 1

    @property
    def num_steps(self) -> int:
        """Total ring steps = world_size - 1.

        Returns
        -------
        int
            Number of ring communication steps required.
        """
        return max(0, self._world_size - 1)

    @property
    def current_kv_rank(self) -> int:
        """Which rank's KV block we currently hold after ``step`` steps.

        Returns
        -------
        int
            The rank whose KV data this device currently holds.
        """
        steps = self._step_counter
        return (self._rank - steps) % self._world_size


if __name__ == "__main__":
    # Self-test: run with world_size=1 (no distributed setup needed)
    print("Testing RingCommunicator...")
    from zenyx.ops.comm.topology import Topology

    topo = Topology()
    comm = RingCommunicator(topology=topo)
    assert comm.world_size == 1
    assert comm.num_steps == 0
    assert comm.current_kv_rank == 0
    print(repr(comm))
    print("PASSED")
