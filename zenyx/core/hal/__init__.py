"""Zenyx Hardware Abstraction Layer (HAL).

Exports:
    HALBase      — Abstract base class for all backends.
    MemTier      — Three-tier memory enum (T0/T1/T2).
    MemBlock     — Handle to a contiguous memory block.
    ReduceOp     — Supported collective-reduce operations.
    CudaHAL      — CUDA backend implementation.
    RocmHAL      — ROCm/HIP backend implementation.
    XlaHAL       — XLA/TPU backend implementation.
    CpuHAL       — CPU/OpenBLAS backend implementation.
    detect_hardware — Auto-detect available hardware.
    HardwareInfo   — Hardware capabilities dataclass.
"""
from zenyx.core.hal.base import HALBase, MemBlock, MemTier, ReduceOp
from zenyx.core.hal.detector import HardwareInfo, detect_hardware

# Lazy imports for backends with optional dependencies
def __getattr__(name: str):
    if name == "CudaHAL":
        from zenyx.core.hal.cuda_hal import CudaHAL
        return CudaHAL
    elif name == "RocmHAL":
        from zenyx.core.hal.rocm_hal import RocmHAL
        return RocmHAL
    elif name == "XlaHAL":
        from zenyx.core.hal.xla_hal import XlaHAL
        return XlaHAL
    elif name == "CpuHAL":
        from zenyx.core.hal.cpu_hal import CpuHAL
        return CpuHAL
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "HALBase",
    "MemTier",
    "MemBlock",
    "ReduceOp",
    "CudaHAL",
    "RocmHAL",
    "XlaHAL",
    "CpuHAL",
    "detect_hardware",
    "HardwareInfo",
]
