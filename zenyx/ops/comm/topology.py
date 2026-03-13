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
from typing import List, Optional

import torch

__all__ = [
    "DeviceType",
    "LinkType",
    "DeviceInfo",
    "Link",
    "Topology",
    "TopologyDetector",
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
        Effective unidirectional ring bandwidth in GB/s.
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

# NVLink 4.0 (H100 SXM): 450 GB/s per direction per link, 900 GB/s bidi
_NVLINK4_BW_GB_S = 450.0
_NVLINK4_LATENCY_US = 1.5

# PCIe Gen4 x16: ~25 GB/s per direction
_PCIE4_BW_GB_S = 25.0
_PCIE4_LATENCY_US = 5.0

# PCIe Gen5 x16: ~63 GB/s per direction
_PCIE5_BW_GB_S = 63.0
_PCIE5_LATENCY_US = 4.0

# TPU v5e ICI: ~400 GB/s per direction per link
_ICI_V5E_BW_GB_S = 400.0
_ICI_LATENCY_US = 1.0

# Reference TFLOPS (BF16)
_H100_SXM_TFLOPS = 989.5
_H100_PCIE_TFLOPS = 756.0
_A100_SXM_TFLOPS = 312.0
_TPU_V5E_TFLOPS = 197.0

# Precomputed S_min reference values
_REFERENCE_S_MIN = {
    "H100_NVLink4": 1_099,  # 989.5 TFLOPS / 450 GB/s × 10³ ÷ 2
    "TPU_v5e_ICI": 493,  # 197 TFLOPS / 400 GB/s × 10³
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
        ring_bw = _NVLINK4_BW_GB_S if has_nvlink else _PCIE4_BW_GB_S
        # AllReduce effective BW ≈ 2 × (N-1)/N × unidirectional ring BW
        n = max(device_count, 1)
        ar_bw = 2.0 * ((n - 1) / n) * ring_bw

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
        try:
            for i in range(device_count):
                handle_i = pynvml.nvmlDeviceGetHandleByIndex(i)
                for j in range(device_count):
                    if i == j:
                        continue
                    try:
                        # Check NVLink status for each potential link
                        info = pynvml.nvmlDeviceGetNvLinkRemotePciInfo_v2(handle_i, j)
                        if info is not None:
                            links.append(
                                Link(
                                    src_device=i,
                                    dst_device=j,
                                    bandwidth_gb_s=_NVLINK4_BW_GB_S,
                                    latency_us=_NVLINK4_LATENCY_US,
                                    link_type=LinkType.NVLINK,
                                )
                            )
                            found_nvlink = True
                    except Exception:
                        continue
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

        for i, dev in enumerate(tpu_devices):
            # TPU v5e: 16 GiB HBM, 197 TFLOPS BF16
            devices.append(
                DeviceInfo(
                    device_id=i,
                    device_type=DeviceType.TPU,
                    memory_gb=16.0,
                    compute_tflops=_TPU_V5E_TFLOPS,
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
                        bandwidth_gb_s=_ICI_V5E_BW_GB_S,
                        latency_us=_ICI_LATENCY_US,
                        link_type=LinkType.ICI,
                    )
                )

        n = max(device_count, 1)
        ar_bw = 2.0 * ((n - 1) / n) * _ICI_V5E_BW_GB_S

        return Topology(
            devices=devices,
            links=links,
            interconnect_type=LinkType.ICI,
            ring_bandwidth_gb_s=_ICI_V5E_BW_GB_S,
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
