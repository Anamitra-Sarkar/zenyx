"""Zenyx hardware auto-detection.

Probes the host for available accelerators and returns a :class:`HardwareInfo`
dataclass describing capabilities, memory, interconnect bandwidth, and compute
throughput.

Detection order: CUDA → ROCm → XLA/TPU → Metal/MLX → CPU.
"""
from __future__ import annotations

import logging
import os
import platform
import subprocess
from dataclasses import dataclass
from typing import Literal

logger = logging.getLogger("zenyx.core.hal.detector")

BackendType = Literal["cuda", "rocm", "xla", "metal", "cpu"]
InterconnectType = Literal["nvlink", "ici", "pcie", "infiniband", "xgmi", "none"]


# ---------------------------------------------------------------------------
# HardwareInfo
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class HardwareInfo:
    """Snapshot of the detected hardware environment.

    Attributes:
        backend:              Accelerator backend identifier.
        device_count:         Number of usable devices (GPUs / TPU chips / CPU sockets).
        per_device_memory_bytes: Bytes of fast memory per device (HBM, VRAM, …).
        interconnect:         Device-to-device link type.
        bandwidth_t0_t1:      Bytes/sec between T0 (device) and T1 (host pinned).
        bandwidth_t1_t2:      Bytes/sec between T1 (host) and T2 (NVMe).
        compute_tflops:       Peak FP16 TFLOPS per device.
        device_name:          Human-readable device name string.

    Time complexity:  O(1) for construction.
    Space complexity: O(1).
    """

    backend: BackendType
    device_count: int
    per_device_memory_bytes: int
    interconnect: InterconnectType
    bandwidth_t0_t1: float  # bytes/sec
    bandwidth_t1_t2: float  # bytes/sec
    compute_tflops: float   # peak FP16 TFLOPS per device
    device_name: str

    def __repr__(self) -> str:
        mem_gib = self.per_device_memory_bytes / (1024 ** 3)
        bw01_gbs = self.bandwidth_t0_t1 / (1024 ** 3)
        bw12_gbs = self.bandwidth_t1_t2 / (1024 ** 3)
        return (
            f"HardwareInfo(\n"
            f"  backend={self.backend!r},\n"
            f"  device_count={self.device_count},\n"
            f"  per_device_memory={mem_gib:.1f} GiB,\n"
            f"  interconnect={self.interconnect!r},\n"
            f"  bandwidth_t0_t1={bw01_gbs:.1f} GiB/s,\n"
            f"  bandwidth_t1_t2={bw12_gbs:.1f} GiB/s,\n"
            f"  compute_tflops={self.compute_tflops:.1f},\n"
            f"  device_name={self.device_name!r}\n"
            f")"
        )


# ---------------------------------------------------------------------------
# Known GPU specs (peak FP16 TFLOPS, interconnect)
# ---------------------------------------------------------------------------

# Mapping of known GPU name substrings → (tflops, interconnect)
_KNOWN_GPUS: dict[str, tuple[float, InterconnectType]] = {
    "H100": (989.5, "nvlink"),
    "H200": (989.5, "nvlink"),
    "A100": (312.0, "nvlink"),
    "A10G": (125.0, "pcie"),
    "L40":  (181.0, "pcie"),
    "L4":   (121.0, "pcie"),
    "V100": (125.0, "nvlink"),
    "4090": (165.2, "pcie"),
    "4080": (97.5,  "pcie"),
    "3090": (71.0,  "pcie"),
}

_KNOWN_ROCM_GPUS: dict[str, tuple[float, InterconnectType]] = {
    "MI300X": (1307.0, "xgmi"),
    "MI300A": (977.0, "xgmi"),
    "MI250X": (383.0, "xgmi"),
    "MI250": (362.0, "pcie"),
    "MI210": (181.0, "pcie"),
    "RX 7900": (61.0, "pcie"),
}

# TPU generation defaults (approximate, per-chip)
_TPU_V4_TFLOPS = 275.0
_TPU_V4_MEM_GB = 32.0
_TPU_V5E_TFLOPS = 197.0
_TPU_V5E_MEM_GB = 16.0
_TPU_V5P_TFLOPS = 459.0
_TPU_V5P_MEM_GB = 32.0


# ---------------------------------------------------------------------------
# Internal detection helpers
# ---------------------------------------------------------------------------


def _detect_cuda() -> HardwareInfo | None:
    """Attempt CUDA detection via PyTorch.

    Time complexity:  O(D) where D = number of CUDA devices.
    Space complexity: O(1).
    """
    try:
        import torch  # noqa: F811
    except ImportError:
        return None

    if not torch.cuda.is_available():
        return None

    device_count = torch.cuda.device_count()
    if device_count == 0:
        return None

    props = torch.cuda.get_device_properties(0)
    device_name: str = props.name
    per_device_mem: int = props.total_mem

    # Determine if this is ROCm masquerading as CUDA
    is_rocm = hasattr(torch.version, "hip") and torch.version.hip is not None
    if is_rocm:
        return _build_rocm_info(device_count, per_device_mem, device_name)

    # Identify GPU model for TFLOPS / interconnect lookup
    tflops = 50.0  # conservative default
    interconnect: InterconnectType = "pcie"
    for name_fragment, (known_tflops, known_ic) in _KNOWN_GPUS.items():
        if name_fragment in device_name:
            tflops = known_tflops
            interconnect = known_ic
            break

    # Multi-GPU NVLink detection via nvidia-smi
    if device_count > 1 and interconnect == "pcie":
        interconnect = _probe_nvlink()

    # Bandwidth estimates
    # PCIe Gen5 x16 ≈ 64 GB/s, PCIe Gen4 x16 ≈ 32 GB/s
    bw_t0_t1: float
    if interconnect == "nvlink":
        bw_t0_t1 = 64.0 * (1024 ** 3)  # host↔device PCIe Gen5
    else:
        bw_t0_t1 = 32.0 * (1024 ** 3)  # PCIe Gen4 fallback

    # NVMe Gen5 ≈ 14 GB/s, Gen4 ≈ 7 GB/s
    bw_t1_t2 = 14.0 * (1024 ** 3)

    logger.info(
        "CUDA detected: %d × %s (%.1f TFLOPS FP16, %s)",
        device_count,
        device_name,
        tflops,
        interconnect,
    )

    return HardwareInfo(
        backend="cuda",
        device_count=device_count,
        per_device_memory_bytes=per_device_mem,
        interconnect=interconnect,
        bandwidth_t0_t1=bw_t0_t1,
        bandwidth_t1_t2=bw_t1_t2,
        compute_tflops=tflops,
        device_name=device_name,
    )


def _build_rocm_info(
    device_count: int,
    per_device_mem: int,
    device_name: str,
) -> HardwareInfo:
    """Build HardwareInfo for ROCm (AMD) GPUs.

    Time complexity:  O(1).
    Space complexity: O(1).
    """
    logger.warning(
        "ROCm detected: expect 37-45%% MFU vs H100's 45-55%% due to kernel gap"
    )
    tflops = 50.0
    interconnect: InterconnectType = "pcie"
    for key, (known_tflops, known_interconnect) in _KNOWN_ROCM_GPUS.items():
        if key in device_name:
            tflops = known_tflops
            interconnect = known_interconnect
            break

    # FIX: ROCm detection only distinguishes XGMI vs PCIe today (no IB probe).
    bw_t0_t1 = 64.0 * (1024 ** 3) if interconnect == "xgmi" else 32.0 * (1024 ** 3)

    return HardwareInfo(
        backend="rocm",
        device_count=device_count,
        per_device_memory_bytes=per_device_mem,
        interconnect=interconnect,
        bandwidth_t0_t1=bw_t0_t1,
        bandwidth_t1_t2=7.0 * (1024 ** 3),
        compute_tflops=tflops,
        device_name=device_name,
    )


def _probe_nvlink() -> InterconnectType:
    """Probe for NVLink via nvidia-smi topology.

    Time complexity:  O(1) — single subprocess call.
    Space complexity: O(output size).
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "topo", "-m"],
            capture_output=True,
            text=True,
            # FIX: Use a shorter timeout to avoid hanging hardware detection.
            timeout=2,
        )
        if result.returncode == 0 and "NV" in result.stdout:
            return "nvlink"
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return "pcie"


def _detect_xla() -> HardwareInfo | None:
    """Attempt XLA / TPU detection via JAX.

    Time complexity:  O(D) where D = number of TPU chips.
    Space complexity: O(1).
    """
    try:
        import jax  # type: ignore[import-untyped]
    except ImportError:
        return None

    devices = jax.devices()
    tpu_devices = [d for d in devices if d.platform.lower() == "tpu"]
    if not tpu_devices:
        return None

    device_count = len(tpu_devices)
    device_kind = (
        str(tpu_devices[0].device_kind) if hasattr(tpu_devices[0], "device_kind") else "TPU"
    )
    device_name = device_kind

    # FIX: Support TPU v4, v5e, and v5p by parsing the device kind string.
    kind_lower = device_kind.lower()
    if "v5p" in kind_lower:
        per_device_mem = _TPU_V5P_MEM_GB * (1024 ** 3)
        compute_tflops = _TPU_V5P_TFLOPS
    elif "v5e" in kind_lower:
        per_device_mem = _TPU_V5E_MEM_GB * (1024 ** 3)
        compute_tflops = _TPU_V5E_TFLOPS
    elif "v4" in kind_lower:
        per_device_mem = _TPU_V4_MEM_GB * (1024 ** 3)
        compute_tflops = _TPU_V4_TFLOPS
    else:
        per_device_mem = _TPU_V5E_MEM_GB * (1024 ** 3)
        compute_tflops = _TPU_V5E_TFLOPS

    logger.info("XLA/TPU detected: %d × %s", device_count, device_name)

    return HardwareInfo(
        backend="xla",
        device_count=device_count,
        per_device_memory_bytes=per_device_mem,
        interconnect="ici",
        bandwidth_t0_t1=20.0 * (1024 ** 3),   # host↔TPU ≈ 20 GB/s
        bandwidth_t1_t2=7.0 * (1024 ** 3),
        compute_tflops=compute_tflops,
        device_name=device_name,
    )


def _detect_metal() -> HardwareInfo | None:
    """Attempt Apple Metal / MPS detection.

    Time complexity:  O(1).
    Space complexity: O(1).
    """
    try:
        import torch
    except ImportError:
        return None

    if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        return None

    logger.warning(
        "Apple Metal: inference and fine-tuning ONLY — no custom allocator support"
    )

    # Rough defaults for Apple Silicon (M2 Ultra as reference)
    return HardwareInfo(
        backend="metal",
        device_count=1,
        per_device_memory_bytes=64 * (1024 ** 3),  # unified memory varies
        interconnect="none",
        bandwidth_t0_t1=200.0 * (1024 ** 3),  # unified memory bandwidth
        bandwidth_t1_t2=7.0 * (1024 ** 3),
        compute_tflops=27.0,
        device_name="Apple Silicon (MPS)",
    )


def _detect_cpu() -> HardwareInfo:
    """CPU fallback — always succeeds.

    Time complexity:  O(1).
    Space complexity: O(1).
    """
    import os as _os

    cpu_count = _os.cpu_count() or 1
    device_name = platform.processor() or "Unknown CPU"

    # Detect AVX-512 support
    avx512 = _has_avx512()
    if avx512:
        logger.info("CPU with AVX-512 detected: %s", device_name)
    else:
        logger.info("CPU detected (no AVX-512): %s", device_name)

    # Rough estimates: DDR5 ≈ 50 GB/s, NVMe ≈ 7 GB/s
    return HardwareInfo(
        backend="cpu",
        device_count=1,
        per_device_memory_bytes=_get_system_memory(),
        interconnect="none",
        bandwidth_t0_t1=50.0 * (1024 ** 3),  # CPU "T0" = system RAM
        bandwidth_t1_t2=7.0 * (1024 ** 3),
        compute_tflops=2.0 if avx512 else 0.5,
        device_name=device_name,
    )


def _has_avx512() -> bool:
    """Check for AVX-512 instruction set support on x86 Linux.

    Time complexity:  O(1).
    Space complexity: O(1).
    """
    if platform.machine() not in ("x86_64", "AMD64"):
        return False
    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if line.startswith("flags"):
                    return "avx512f" in line
    except OSError:
        pass
    return False


def _get_system_memory() -> int:
    """Return total system RAM in bytes.

    Time complexity:  O(1).
    Space complexity: O(1).
    """
    try:
        import psutil
        return psutil.virtual_memory().total
    except ImportError:
        pass
    try:
        mem_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")  # type: ignore[attr-defined]
        return mem_bytes
    except (AttributeError, ValueError):
        # Fallback: 16 GiB default
        return 16 * (1024 ** 3)


def build_hal_for_hardware(hw_info: HardwareInfo, device_index: int = 0):
    """Instantiate the appropriate HAL backend for *hw_info*.

    This is a **single-device factory**.  To use all devices on a
    multi-GPU / multi-chip host, call this function once per device index
    (``0`` through ``hw_info.device_count - 1``) and manage the resulting
    HAL list yourself.

    Args:
        hw_info:      Hardware descriptor from :func:`detect_hardware`.
        device_index: Which device to build the HAL for (default ``0``).
                      Must be in ``range(hw_info.device_count)``.

    Raises:
        ValueError: If *device_index* is out of range.

    Time complexity: O(1).
    Space complexity: O(1).
    """
    if device_index < 0 or device_index >= hw_info.device_count:
        raise ValueError(
            f"device_index={device_index} is out of range for "
            f"hw_info.device_count={hw_info.device_count}."
        )

    if hw_info.device_count > 1 and device_index == 0:
        logger.warning(
            "build_hal_for_hardware: %d devices detected but building HAL for "
            "device_index=0 only. Call this function once per device index "
            "(0..%d) to utilise all devices.",
            hw_info.device_count,
            hw_info.device_count - 1,
        )

    if hw_info.backend == "cuda":
        from zenyx.core.hal.cuda_hal import CudaHAL
        return CudaHAL(device=device_index)

    if hw_info.backend == "rocm":
        from zenyx.core.hal.rocm_hal import RocmHAL
        return RocmHAL(device_index=device_index)

    if hw_info.backend == "xla":
        from zenyx.core.hal.xla_hal import XlaHAL
        return XlaHAL(device_index=device_index)

    if hw_info.backend == "metal":
        logger.info("Metal backend: using CpuHAL (MPS memory management via PyTorch)")
        from zenyx.core.hal.cpu_hal import CpuHAL
        return CpuHAL(t0_capacity_bytes=hw_info.per_device_memory_bytes)

    from zenyx.core.hal.cpu_hal import CpuHAL
    return CpuHAL(t0_capacity_bytes=hw_info.per_device_memory_bytes)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_hardware() -> HardwareInfo:
    """Auto-detect the best available hardware backend.

    Detection order: CUDA → ROCm (via CUDA shim) → XLA/TPU → Metal/MLX → CPU.

    Returns:
        :class:`HardwareInfo` describing the detected environment.

    Time complexity:  O(D) where D = total number of accelerator devices.
    Space complexity: O(1).
    """
    # CUDA / ROCm (ROCm uses torch.cuda shim, handled inside _detect_cuda)
    info = _detect_cuda()
    if info is not None:
        return info

    # XLA / TPU
    info = _detect_xla()
    if info is not None:
        return info

    # Metal / MPS
    info = _detect_metal()
    if info is not None:
        return info

    # CPU fallback — always works
    return _detect_cpu()
