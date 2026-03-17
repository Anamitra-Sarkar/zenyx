"""Hardware topology detection and ring-attention feasibility analysis.

Detects NVLink, ICI, PCIe, and InfiniBand interconnects, then computes the
minimum sequence chunk size ``S_min`` for ring attention.

Complexity
----------
Detection is O(D²) where D = number of devices (link probing).
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import torch

__all__ = [
    "DeviceType",
    "LinkType",
    "DeviceInfo",
    "Link",
    "Topology",
    "TopologyDetector",
    "TopologyInfo",
    "detect_topology",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class DeviceType(Enum):
    """Accelerator type."""

    GPU = "gpu"
    TPU = "tpu"
    CPU = "cpu"


class LinkType(Enum):
    """Inter-device interconnect type."""

    NVLINK = "nvlink"
    ICI = "ici"
    PCIE = "pcie"
    INFINIBAND = "infiniband"
    LOOPBACK = "loopback"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DeviceInfo:
    """Descriptor for a single compute device.

    Attributes
    ----------
    device_id : int
        Ordinal device index.
    device_type : DeviceType
        Kind of accelerator.
    memory_gb : float
        Total device memory in GiB.
    compute_tflops : float
        Peak FP16/BF16 TFLOPS.
    """

    device_id: int
    device_type: DeviceType
    memory_gb: float
    compute_tflops: float

    def __repr__(self) -> str:
        return (
            f"DeviceInfo(id={self.device_id}, type={self.device_type.value}, "
            f"mem={self.memory_gb:.1f}GB, compute={self.compute_tflops:.1f}TFLOPS)"
        )


@dataclass(frozen=True)
class Link:
    """Descriptor for a unidirectional device-to-device link.

    Attributes
    ----------
    src_device : int
        Source device ordinal.
    dst_device : int
        Destination device ordinal.
    bandwidth_gb_s : float
        Unidirectional bandwidth in GB/s.
    latency_us : float
        One-way latency in microseconds.
    link_type : LinkType
        Interconnect technology.
    """

    src_device: int
    dst_device: int
    bandwidth_gb_s: float
    latency_us: float
    link_type: LinkType

    def __repr__(self) -> str:
        return (
            f"Link({self.src_device}→{self.dst_device}, "
            f"{self.bandwidth_gb_s:.1f}GB/s, "
            f"{self.latency_us:.1f}μs, {self.link_type.value})"
        )


@dataclass
class Topology:
    """Full hardware topology for a set of devices.

    Attributes
    ----------
    devices : list[DeviceInfo]
        All detected devices.
    links : list[Link]
        All detected inter-device links.
    interconnect_type : LinkType
        Dominant interconnect technology.
    ring_bandwidth_gb_s : float
        Effective ring bandwidth in GB/s (bidirectional for NVLink per S_min convention).
    all_reduce_bandwidth_gb_s : float
        Effective AllReduce bandwidth in GB/s.
    s_min : int
        Minimum sequence chunk size for ring attention (tokens).
    """

    devices: List[DeviceInfo] = field(default_factory=list)
    links: List[Link] = field(default_factory=list)
    interconnect_type: LinkType = LinkType.PCIE
    ring_bandwidth_gb_s: float = 0.0
    all_reduce_bandwidth_gb_s: float = 0.0
    s_min: int = 0

    def __repr__(self) -> str:
        return (
            f"Topology(devices={len(self.devices)}, "
            f"interconnect={self.interconnect_type.value}, "
            f"ring_bw={self.ring_bandwidth_gb_s:.1f}GB/s, "
            f"S_min={self.s_min})"
        )


# ---------------------------------------------------------------------------
# Reference bandwidth / compute values
# ---------------------------------------------------------------------------

# NVLink 4.0 (H100 SXM): 450 GB/s per direction per link, 900 GB/s bidirectional
_NVLINK4_BW_GB_S = 450.0
# FIX: Use bidirectional bandwidth for ring-attention S_min calculations.
_NVLINK4_BW_BIDIR_GB_S = _NVLINK4_BW_GB_S * 2.0
_NVLINK4_LATENCY_US = 1.5

# PCIe Gen4 x16: ~25 GB/s per direction
_PCIE4_BW_GB_S = 25.0
_PCIE4_LATENCY_US = 5.0

# PCIe Gen5 x16: ~63 GB/s per direction
_PCIE5_BW_GB_S = 63.0
_PCIE5_LATENCY_US = 4.0

# TPU ICI bandwidths (approximate, per direction)
_ICI_V4_BW_GB_S = 400.0
_ICI_V5E_BW_GB_S = 400.0
_ICI_V5P_BW_GB_S = 1200.0
_ICI_LATENCY_US = 1.0

# Reference TFLOPS (BF16)
_H100_SXM_TFLOPS = 989.5
_H100_PCIE_TFLOPS = 756.0
_A100_SXM_TFLOPS = 312.0
_TPU_V4_TFLOPS = 275.0
_TPU_V5E_TFLOPS = 197.0
_TPU_V5P_TFLOPS = 459.0

# Precomputed S_min reference values
_REFERENCE_S_MIN = {
    "H100_NVLink4": 1_099,  # 989.5 TFLOPS / 900 GB/s × 10³ (bidirectional)
    "TPU_v4_ICI": 687,  # 275 TFLOPS / 400 GB/s × 10³
    "TPU_v5e_ICI": 493,  # 197 TFLOPS / 400 GB/s × 10³
    "TPU_v5p_ICI": 383,  # 459 TFLOPS / 1200 GB/s × 10³
    "PCIe_Gen4": 15_453,  # 386 TFLOPS / 25 GB/s × 10³  (A100-like)
}


# ---------------------------------------------------------------------------
# TopologyDetector
# ---------------------------------------------------------------------------

class TopologyDetector:
    """Detect hardware topology and compute ring-attention parameters.

    Probes available accelerators and their interconnects to build a
    :class:`Topology` descriptor.  Supports CUDA (NVLink / PCIe),
    JAX/TPU (ICI), and CPU fallback.

    Complexity
    ----------
    Time : O(D²)  where D = number of devices (link probing)
    Space: O(D²)  for link storage
    """

    def __repr__(self) -> str:
        return "TopologyDetector()"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self) -> Topology:
        """Detect the full hardware topology.

        Returns
        -------
        Topology
            Populated topology with devices, links, and ``s_min``.

        Complexity
        ----------
        Time : O(D²)
        Space: O(D²)
        """
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            topology = self._detect_cuda()
        else:
            topology = self._try_detect_tpu()
            if topology is None:
                topology = self._detect_cpu()

        self._compute_s_min(topology)
        self._warn_if_pcie(topology)
        return topology

    # ------------------------------------------------------------------
    # CUDA detection
    # ------------------------------------------------------------------

    def _detect_cuda(self) -> Topology:
        """Detect CUDA devices and NVLink/PCIe topology.

        Complexity
        ----------
        Time : O(D²) for pairwise link probing
        Space: O(D + D²)
        """
        device_count = torch.cuda.device_count()
        devices: List[DeviceInfo] = []
        links: List[Link] = []

        has_nvlink = False

        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_mem / (1024**3)

            # Estimate compute TFLOPS from device name
            compute_tflops = self._estimate_cuda_tflops(props.name)

            devices.append(
                DeviceInfo(
                    device_id=i,
                    device_type=DeviceType.GPU,
                    memory_gb=memory_gb,
                    compute_tflops=compute_tflops,
                )
            )

        # Probe NVLink via pynvml if available, else fall back to heuristics
        has_nvlink = self._probe_nvlink(device_count, links)

        if not has_nvlink:
            # Fall back to PCIe links
            self._add_pcie_links(device_count, links)

        interconnect = LinkType.NVLINK if has_nvlink else LinkType.PCIE
        ring_bw = _NVLINK4_BW_BIDIR_GB_S if has_nvlink else _PCIE4_BW_GB_S
        # AllReduce effective BW ≈ 2 × (N-1)/N × unidirectional ring BW
        # FIX: Use per-direction bandwidth for AllReduce even when ring_bw is bidirectional.
        ring_bw_for_ar = _NVLINK4_BW_GB_S if has_nvlink else ring_bw
        n = max(device_count, 1)
        ar_bw = 2.0 * ((n - 1) / n) * ring_bw_for_ar

        return Topology(
            devices=devices,
            links=links,
            interconnect_type=interconnect,
            ring_bandwidth_gb_s=ring_bw,
            all_reduce_bandwidth_gb_s=ar_bw,
        )

    @staticmethod
    def _estimate_cuda_tflops(name: str) -> float:
        """Estimate BF16 TFLOPS from CUDA device name.

        Parameters
        ----------
        name : str
            ``torch.cuda.get_device_properties().name``.

        Returns
        -------
        float
            Estimated peak BF16 TFLOPS.
        """
        name_upper = name.upper()
        if "H100" in name_upper and "SXM" in name_upper:
            return _H100_SXM_TFLOPS
        if "H100" in name_upper:
            return _H100_PCIE_TFLOPS
        if "A100" in name_upper and "SXM" in name_upper:
            return _A100_SXM_TFLOPS
        if "A100" in name_upper:
            return _A100_SXM_TFLOPS * 0.9  # PCIe A100 is ~90% of SXM
        # Conservative default
        return 200.0

    @staticmethod
    def _probe_nvlink(device_count: int, links: List[Link]) -> bool:
        """Attempt NVLink detection via pynvml.

        Parameters
        ----------
        device_count : int
            Number of CUDA devices.
        links : list[Link]
            Mutated in-place — detected NVLink links are appended.

        Returns
        -------
        bool
            ``True`` if at least one NVLink link was found.
        """
        try:
            import pynvml  # type: ignore[import-untyped]

            pynvml.nvmlInit()
        except (ImportError, Exception):
            logger.debug("pynvml not available; skipping NVLink probe")
            return False

        found_nvlink = False
        # FIX: NVML expects a link index, not a device index; map links via PCI IDs.
        pci_to_index: Dict[str, int] = {}
        handles: List[object] = []
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            handles.append(handle)
            try:
                pci_info = pynvml.nvmlDeviceGetPciInfo(handle)
                bus_id = pci_info.busId.decode() if isinstance(pci_info.busId, bytes) else pci_info.busId
                pci_to_index[str(bus_id)] = i
            except Exception:
                continue
        try:
            max_links = 12
            for i, handle_i in enumerate(handles):
                for link_idx in range(max_links):
                    try:
                        if not pynvml.nvmlDeviceGetNvLinkState(handle_i, link_idx):
                            continue
                    except Exception:
                        break
                    try:
                        info = pynvml.nvmlDeviceGetNvLinkRemotePciInfo_v2(handle_i, link_idx)
                    except Exception:
                        continue
                    bus_id = info.busId.decode() if isinstance(info.busId, bytes) else info.busId
                    dst_device = pci_to_index.get(str(bus_id))
                    if dst_device is None or dst_device == i:
                        continue
                    links.append(
                        Link(
                            src_device=i,
                            dst_device=dst_device,
                            bandwidth_gb_s=_NVLINK4_BW_GB_S,
                            latency_us=_NVLINK4_LATENCY_US,
                            link_type=LinkType.NVLINK,
                        )
                    )
                    found_nvlink = True
        finally:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

        return found_nvlink

    @staticmethod
    def _add_pcie_links(device_count: int, links: List[Link]) -> None:
        """Add PCIe Gen4 links between all device pairs.

        Parameters
        ----------
        device_count : int
            Number of CUDA devices.
        links : list[Link]
            Mutated in-place.
        """
        for i in range(device_count):
            for j in range(device_count):
                if i == j:
                    continue
                links.append(
                    Link(
                        src_device=i,
                        dst_device=j,
                        bandwidth_gb_s=_PCIE4_BW_GB_S,
                        latency_us=_PCIE4_LATENCY_US,
                        link_type=LinkType.PCIE,
                    )
                )

    # ------------------------------------------------------------------
    # TPU detection
    # ------------------------------------------------------------------

    def _try_detect_tpu(self) -> Optional[Topology]:
        """Attempt TPU / ICI detection via JAX.

        Returns
        -------
        Topology | None
            Topology if JAX + TPU backend available, else ``None``.

        Complexity
        ----------
        Time : O(D²)
        Space: O(D²)
        """
        try:
            import jax  # type: ignore[import-untyped]
        except ImportError:
            return None

        try:
            tpu_devices = jax.devices("tpu")
        except (RuntimeError, ValueError):
            return None

        if not tpu_devices:
            return None

        devices: List[DeviceInfo] = []
        links: List[Link] = []
        device_count = len(tpu_devices)

        # FIX: Parse TPU generation from device_kind to pick specs.
        kind = str(getattr(tpu_devices[0], "device_kind", "")).lower()
        if "v5p" in kind:
            mem_gb = 32.0
            tflops = _TPU_V5P_TFLOPS
            ici_bw = _ICI_V5P_BW_GB_S
        elif "v5e" in kind:
            mem_gb = 16.0
            tflops = _TPU_V5E_TFLOPS
            ici_bw = _ICI_V5E_BW_GB_S
        elif "v4" in kind:
            mem_gb = 32.0
            tflops = _TPU_V4_TFLOPS
            ici_bw = _ICI_V4_BW_GB_S
        else:
            mem_gb = 16.0
            tflops = _TPU_V5E_TFLOPS
            ici_bw = _ICI_V5E_BW_GB_S

        for i, _dev in enumerate(tpu_devices):
            devices.append(
                DeviceInfo(
                    device_id=i,
                    device_type=DeviceType.TPU,
                    memory_gb=mem_gb,
                    compute_tflops=tflops,
                )
            )

        # ICI mesh: assume fully connected within a slice
        for i in range(device_count):
            for j in range(device_count):
                if i == j:
                    continue
                links.append(
                    Link(
                        src_device=i,
                        dst_device=j,
                        bandwidth_gb_s=ici_bw,
                        latency_us=_ICI_LATENCY_US,
                        link_type=LinkType.ICI,
                    )
                )

        n = max(device_count, 1)
        ar_bw = 2.0 * ((n - 1) / n) * ici_bw

        return Topology(
            devices=devices,
            links=links,
            interconnect_type=LinkType.ICI,
            ring_bandwidth_gb_s=ici_bw,
            all_reduce_bandwidth_gb_s=ar_bw,
        )

    # ------------------------------------------------------------------
    # CPU fallback
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_cpu() -> Topology:
        """Create a single-device CPU topology.

        Returns
        -------
        Topology
            Topology with one CPU device and a loopback link.
        """
        import multiprocessing
        import os

        cores = os.cpu_count() or multiprocessing.cpu_count()
        # Rough estimate: modern x86 core ≈ 0.1 BF16 TFLOPS with AVX-512
        compute = cores * 0.1

        devices = [
            DeviceInfo(
                device_id=0,
                device_type=DeviceType.CPU,
                memory_gb=0.0,  # Not meaningful for CPU
                compute_tflops=compute,
            )
        ]
        links = [
            Link(
                src_device=0,
                dst_device=0,
                bandwidth_gb_s=50.0,  # DDR5 bandwidth estimate
                latency_us=0.1,
                link_type=LinkType.LOOPBACK,
            )
        ]

        return Topology(
            devices=devices,
            links=links,
            interconnect_type=LinkType.LOOPBACK,
            ring_bandwidth_gb_s=50.0,
            all_reduce_bandwidth_gb_s=50.0,
        )

    # ------------------------------------------------------------------
    # S_min computation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_s_min(topology: Topology) -> None:
        """Compute minimum sequence chunk size for ring attention.

        Formula: ``S_min = (F × 10³) / B``
        where F = peak TFLOPS, B = ring bandwidth GB/s.

        Mutates ``topology.s_min`` in-place.

        Parameters
        ----------
        topology : Topology
            Topology to update.
        """
        if not topology.devices or topology.ring_bandwidth_gb_s <= 0:
            topology.s_min = 0
            return

        # Use the average compute across devices
        avg_tflops = sum(d.compute_tflops for d in topology.devices) / len(
            topology.devices
        )
        bw = topology.ring_bandwidth_gb_s

        topology.s_min = int((avg_tflops * 1_000) / bw)

        logger.info(
            "S_min = %d tokens (F=%.1f TFLOPS, B=%.1f GB/s)",
            topology.s_min,
            avg_tflops,
            bw,
        )

    @staticmethod
    def _warn_if_pcie(topology: Topology) -> None:
        """Emit a warning if the topology is PCIe-only.

        Parameters
        ----------
        topology : Topology
            Detected topology.
        """
        if topology.interconnect_type == LinkType.PCIE:
            msg = (
                f"PCIe Gen4 interconnect: ring attention for 1M+ context is "
                f"not feasible (S_min={topology.s_min:,} tokens)"
            )
            warnings.warn(msg, stacklevel=3)
            logger.warning(msg)


# ---------------------------------------------------------------------------
# TopologyInfo — lightweight topology summary per the Phase 3 spec
# ---------------------------------------------------------------------------


@dataclass
class TopologyInfo:
    """Detected hardware topology summary for ring attention.

    Attributes
    ----------
    backend : str
        Communication backend: ``"nccl"``, ``"xla_ici"``, ``"gloo"``, ``"mpi"``.
    interconnect : str
        Interconnect type: ``"nvlink"``, ``"pcie_gen5"``, ``"pcie_gen4"``,
        ``"ici_2d_torus"``, ``"ici_3d_torus"``, ``"infinity_fabric"``.
    ring_bandwidth_gbps : float
        Effective ring bandwidth in GB/s for ring attention.
    world_size : int
        Total number of participating devices.
    local_rank : int
        Rank of this process within the local node.
    global_rank : int
        Global rank of this process.
    num_nodes : int
        Number of distinct nodes.
    gpus_per_node : int
        Number of GPUs (or accelerators) per node.
    s_min_tokens : int
        Minimum ring attention chunk size for this hardware.

    Time complexity:  O(1) for construction.
    Space complexity: O(1).
    """

    backend: str
    interconnect: str
    ring_bandwidth_gbps: float
    world_size: int
    local_rank: int
    global_rank: int
    num_nodes: int
    gpus_per_node: int
    s_min_tokens: int

    def __repr__(self) -> str:
        return (
            f"TopologyInfo(\n"
            f"  backend={self.backend!r},\n"
            f"  interconnect={self.interconnect!r},\n"
            f"  ring_bandwidth_gbps={self.ring_bandwidth_gbps:.1f},\n"
            f"  world_size={self.world_size},\n"
            f"  local_rank={self.local_rank},\n"
            f"  global_rank={self.global_rank},\n"
            f"  num_nodes={self.num_nodes},\n"
            f"  gpus_per_node={self.gpus_per_node},\n"
            f"  s_min_tokens={self.s_min_tokens}\n"
            f")"
        )


def detect_topology() -> TopologyInfo:
    """Auto-detect hardware topology from ``torch.distributed`` and device info.

    Detects the communication backend, interconnect type, ring bandwidth,
    and computes the minimum ring attention chunk size ``s_min_tokens``
    using the formula::

        s_min = ceil((compute_tflops × 1000) / ring_bandwidth_gbps)

    Known hardware configurations with pre-computed values:

    * H100 NVLink 4.0: ``ring_bandwidth_gbps=900``, ``s_min=1099``
    * TPU v5e ICI: ``ring_bandwidth_gbps=400``, ``s_min=493``
    * PCIe Gen4: ``ring_bandwidth_gbps=64``, ``s_min=15454``
    * PCIe Gen5: ``ring_bandwidth_gbps=128``, ``s_min=7726``

    Emits a warning when PCIe Gen4 is detected with large context lengths.

    Returns
    -------
    TopologyInfo
        Populated topology information.

    Time complexity:  O(D²) where D = number of devices.
    Space complexity: O(1) for the returned dataclass.
    """
    import math
    import os

    # Defaults for single-device / non-distributed
    world_size = 1
    global_rank = 0
    local_rank = 0
    backend = "gloo"

    # Check torch.distributed
    dist_available = False
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            dist_available = True
            world_size = dist.get_world_size()
            global_rank = dist.get_rank()
            backend = dist.get_backend()
    except Exception:
        pass

    # Local rank from environment variable (standard convention)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    # Detect hardware via the existing TopologyDetector
    detector = TopologyDetector()
    topo = detector.detect()

    # Determine interconnect string and ring bandwidth
    interconnect: str
    ring_bw: float
    compute_tflops: float

    if topo.devices:
        compute_tflops = sum(d.compute_tflops for d in topo.devices) / len(topo.devices)
    else:
        compute_tflops = 2.0  # CPU fallback

    link_type = topo.interconnect_type
    if link_type == LinkType.NVLINK:
        interconnect = "nvlink"
        ring_bw = _NVLINK4_BW_BIDIR_GB_S
        if backend != "nccl" and dist_available:
            backend = "nccl"
    elif link_type == LinkType.ICI:
        interconnect = "ici_2d_torus"
        ring_bw = topo.ring_bandwidth_gb_s
        backend = "xla_ici"
    elif link_type == LinkType.PCIE:
        # Distinguish Gen4 vs Gen5
        if topo.ring_bandwidth_gb_s >= 50.0:
            interconnect = "pcie_gen5"
            ring_bw = 128.0
        else:
            interconnect = "pcie_gen4"
            ring_bw = 64.0
    else:
        interconnect = "pcie_gen4"
        ring_bw = 64.0

    # Compute s_min
    if ring_bw > 0:
        s_min = int(math.ceil((compute_tflops * 1e3) / ring_bw))
    else:
        s_min = 0

    # Determine gpus_per_node
    gpus_per_node = 1
    if torch.cuda.is_available():
        gpus_per_node = torch.cuda.device_count()
    elif topo.devices:
        gpus_per_node = len(topo.devices)

    num_nodes = max(1, world_size // max(gpus_per_node, 1))

    # PCIe Gen4 warning for large contexts
    if interconnect == "pcie_gen4":
        logger.warning(
            "WARNING: PCIe Gen4 ring attention requires S_min=%d tokens. "
            "For 1M context with 8 devices this requires 125K tokens per "
            "device — memory pressure will be extreme. Consider NVLink or "
            "ICI hardware.",
            s_min,
        )

    return TopologyInfo(
        backend=backend,
        interconnect=interconnect,
        ring_bandwidth_gbps=ring_bw,
        world_size=world_size,
        local_rank=local_rank,
        global_rank=global_rank,
        num_nodes=num_nodes,
        gpus_per_node=gpus_per_node,
        s_min_tokens=s_min,
    )


if __name__ == "__main__":
    # Self-test: run with world_size=1 (no distributed setup needed)
    print("Testing topology detection...")
    info = detect_topology()
    print(info)
    # FIX: Avoid assert for runtime validation in self-test.
    if info.world_size < 1:
        raise RuntimeError(f"Expected world_size >= 1, got {info.world_size}")
    if info.s_min_tokens < 0:
        raise RuntimeError(f"Expected s_min_tokens >= 0, got {info.s_min_tokens}")
    print("PASSED")
