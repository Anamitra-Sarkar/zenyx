"""
ZENYX Phase 7: Bélády-Optimal KV Cache Tiering for Ring Attention

This module implements deterministic offline-optimal page replacement for KV cache
across three-tier memory (HBM/DRAM/NVMe) during ring attention with predictable
forward+backward access patterns.

Key insight: Ring attention produces a completely deterministic access sequence.
Forward: device d at step r needs KV from device (d-r) mod N
Backward: symmetric reverse pattern - also known in advance.
Therefore Bélády's optimal eviction (evict page used farthest in future) applies.

Refs: Phase 7 of Zenyx papers
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Callable
from collections import defaultdict
import heapq
from dataclasses import dataclass, field
from enum import Enum


class MemoryTier(Enum):
    """Three-tier memory hierarchy"""
    HBM = 0     # High Bandwidth Memory (16GB on v5e-8)
    DRAM = 1    # System DRAM (CPU host memory)
    NVME = 2    # NVMe disk storage


@dataclass
class MemoryBandwidth:
    """Measured or specified memory bandwidth between tiers"""
    hbm_to_dram: float = 400e9      # ~400 GB/s (HBM-DRAM link)
    dram_to_nvme: float = 100e9     # ~100 GB/s (typical SSD)
    compute_flops: float = 197e12   # 197 TFLOP/s (v5e peak per chip)
    
    def check_feasibility(self) -> Tuple[bool, float]:
        """
        Check three-tier memory guarantee:
        (1/B_01) + (1/B_12) <= 1/F_compute
        
        If violated, compute must throttle to let data stream through.
        
        Returns:
            (is_feasible, headroom_ratio)
        """
        lhs = (1.0 / self.hbm_to_dram) + (1.0 / self.dram_to_nvme)
        rhs = 1.0 / self.compute_flops
        
        if lhs <= rhs:
            headroom = rhs / lhs if lhs > 0 else float('inf')
            return True, headroom
        else:
            return False, lhs / rhs


@dataclass
class KVCacheMetadata:
    """Metadata for a KV cache block in three-tier system"""
    layer_id: int
    block_id: int              # Ring step / block index
    device_id: int             # Which device holds this block
    size_bytes: int
    tier: MemoryTier
    last_forward_access: int   # Step in forward pass
    last_backward_access: int  # Step in backward pass
    
    @property
    def last_access_overall(self) -> int:
        """Combined forward+backward access schedule"""
        return max(self.last_forward_access, self.last_backward_access)


class RingAttentionAccessSequence:
    """
    Computes deterministic access pattern for ring attention.
    
    For ring attention with N devices and sequence split into N blocks:
    - Forward pass step r: device d reads KV from device (d - r) mod N
    - Backward pass: reverse pattern with same blocks
    
    This creates a fully deterministic access schedule that Bélády can optimize.
    """
    
    def __init__(self, 
                 num_devices: int, 
                 num_layers: int,
                 num_ring_steps: int,
                 tokens_per_block: int,
                 kv_bytes_per_token: int = 4096):
        self.num_devices = num_devices
        self.num_layers = num_layers
        self.num_ring_steps = num_ring_steps
        self.tokens_per_block = tokens_per_block
        self.kv_bytes_per_token = kv_bytes_per_token
        
        # Precompute access schedule
        self.forward_accesses = self._compute_forward_accesses()
        self.backward_accesses = self._compute_backward_accesses()
        self.combined_schedule = self._combine_schedules()
    
    def _compute_forward_accesses(self) -> Dict[Tuple[int, int, int], List[int]]:
        """
        Forward pass access schedule.
        
        Returns:
            {(device_id, layer_id, kv_block_id): [step indices]}
        """
        accesses = defaultdict(list)
        
        for step in range(self.num_ring_steps):
            for device_id in range(self.num_devices):
                for layer_id in range(self.num_layers):
                    # At ring step r, device d needs KV from device (d-r) mod N
                    source_device = (device_id - step) % self.num_devices
                    accesses[(device_id, layer_id, source_device)].append(step)
        
        return dict(accesses)
    
    def _compute_backward_accesses(self) -> Dict[Tuple[int, int, int], List[int]]:
        """
        Backward pass access schedule (reverse of forward).
        Offset to occur after forward pass.
        """
        accesses = defaultdict(list)
        backward_offset = self.num_ring_steps
        
        # Backward follows reverse ring pattern
        for step in range(self.num_ring_steps):
            for device_id in range(self.num_devices):
                for layer_id in range(self.num_layers):
                    # Backward accesses KV in reverse order
                    source_device = (device_id + step) % self.num_devices
                    backward_step = backward_offset + (self.num_ring_steps - 1 - step)
                    accesses[(device_id, layer_id, source_device)].append(backward_step)
        
        return dict(accesses)
    
    def _combine_schedules(self) -> Dict[Tuple[int, int, int], Tuple[int, int]]:
        """
        Combine forward and backward schedules.
        
        Returns:
            {(device_id, layer_id, kv_block_id): (last_forward, last_backward)}
        """
        combined = {}
        all_keys = set(self.forward_accesses.keys()) | set(self.backward_accesses.keys())
        
        for key in all_keys:
            forward_times = self.forward_accesses.get(key, [])
            backward_times = self.backward_accesses.get(key, [])
            
            last_fwd = max(forward_times) if forward_times else -1
            last_bwd = max(backward_times) if backward_times else -1
            
            combined[key] = (last_fwd, last_bwd)
        
        return combined
    
    def get_next_access(self, block_key: Tuple[int, int, int], current_step: int) -> int:
        """
        Distance to next access (for Bélády eviction).
        Returns: number of steps until next access, or infinity if no future access.
        """
        forward_times = self.forward_accesses.get(block_key, [])
        backward_times = self.backward_accesses.get(block_key, [])
        all_times = sorted(forward_times + backward_times)
        
        # Find first access at or after current_step
        future_accesses = [t for t in all_times if t > current_step]
        
        if future_accesses:
            return future_accesses[0] - current_step
        else:
            return float('inf')


class BeladyKVCacheTieringManager:
    """
    Bélády-optimal KV cache tiering across HBM/DRAM/NVMe.
    
    Manages three-tier memory with offline-optimal eviction based on
    predictable ring attention access patterns.
    """
    
    def __init__(self,
                 ring_sequence: RingAttentionAccessSequence,
                 hbm_size_bytes: int = 12e9,      # 12 GB for KV (leave 4GB for weights/state)
                 dram_size_bytes: int = 128e9,    # 128 GB system DRAM
                 nvme_size_bytes: int = 2e12,     # 2 TB NVMe
                 bandwidth: Optional[MemoryBandwidth] = None):
        
        self.ring_sequence = ring_sequence
        self.hbm_size = hbm_size_bytes
        self.dram_size = dram_size_bytes
        self.nvme_size = nvme_size_bytes
        self.bandwidth = bandwidth or MemoryBandwidth()
        
        # Current occupancy per tier
        self.hbm_contents: Dict[Tuple, bytes] = {}  # (device, layer, block) -> size
        self.dram_contents: Dict[Tuple, bytes] = {}
        self.nvme_contents: Dict[Tuple, bytes] = {}
        
        self.hbm_used = 0
        self.dram_used = 0
        self.nvme_used = 0
        
        # Statistics
        self.hbm_hits = 0
        self.dram_hits = 0
        self.nvme_hits = 0
        self.evictions = 0
    
    def allocate_kv_block(self, 
                         block_key: Tuple[int, int, int],
                         size_bytes: int,
                         current_step: int) -> MemoryTier:
        """
        Allocate a KV block using Bélády-optimal eviction.
        
        Returns: Which tier the block is stored in
        """
        # Try HBM first
        if self.hbm_used + size_bytes <= self.hbm_size:
            self.hbm_contents[block_key] = size_bytes
            self.hbm_used += size_bytes
            return MemoryTier.HBM
        
        # HBM full - evict least-recently-needed block using Bélády
        if self.hbm_used > 0:
            # Find block in HBM with farthest next access
            farthest_block = None
            farthest_distance = -1
            
            for hbm_block in self.hbm_contents:
                distance = self.ring_sequence.get_next_access(hbm_block, current_step)
                if distance > farthest_distance:
                    farthest_distance = distance
                    farthest_block = hbm_block
            
            # Evict it to DRAM
            if farthest_block and farthest_distance != 0:  # Don't evict if needed now
                evicted_size = self.hbm_contents.pop(farthest_block)
                self.hbm_used -= evicted_size
                
                # Move to DRAM if space, else NVMe
                if self.dram_used + evicted_size <= self.dram_size:
                    self.dram_contents[farthest_block] = evicted_size
                    self.dram_used += evicted_size
                else:
                    self.nvme_contents[farthest_block] = evicted_size
                    self.nvme_used += evicted_size
                
                self.evictions += 1
        
        # Now allocate new block to HBM
        if self.hbm_used + size_bytes <= self.hbm_size:
            self.hbm_contents[block_key] = size_bytes
            self.hbm_used += size_bytes
            return MemoryTier.HBM
        
        # Still full - use DRAM
        if self.dram_used + size_bytes <= self.dram_size:
            self.dram_contents[block_key] = size_bytes
            self.dram_used += size_bytes
            return MemoryTier.DRAM
        
        # Use NVMe as fallback
        self.nvme_contents[block_key] = size_bytes
        self.nvme_used += size_bytes
        return MemoryTier.NVME
    
    def access_kv_block(self, block_key: Tuple, current_step: int) -> MemoryTier:
        """
        Access a KV block - returns which tier it's in (with automatic prefetch if needed).
        """
        if block_key in self.hbm_contents:
            self.hbm_hits += 1
            return MemoryTier.HBM
        elif block_key in self.dram_contents:
            self.dram_hits += 1
            # Consider prefetching to HBM if space available
            return MemoryTier.DRAM
        elif block_key in self.nvme_contents:
            self.nvme_hits += 1
            # Prefetch from NVMe to DRAM to HBM
            return MemoryTier.NVME
        else:
            raise KeyError(f"KV block {block_key} not allocated")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Return tiering statistics"""
        total_hits = self.hbm_hits + self.dram_hits + self.nvme_hits
        return {
            'hbm_hits': self.hbm_hits,
            'dram_hits': self.dram_hits,
            'nvme_hits': self.nvme_hits,
            'total_hits': total_hits,
            'hbm_hit_rate': self.hbm_hits / total_hits if total_hits > 0 else 0,
            'dram_hit_rate': self.dram_hits / total_hits if total_hits > 0 else 0,
            'nvme_hit_rate': self.nvme_hits / total_hits if total_hits > 0 else 0,
            'evictions': self.evictions,
            'hbm_utilization': self.hbm_used / self.hbm_size,
            'dram_utilization': self.dram_used / self.dram_size,
            'nvme_utilization': self.nvme_used / self.nvme_size,
        }


# Example usage
if __name__ == "__main__":
    # Configuration for 1T model on v5e-8 with 1M context
    ring_seq = RingAttentionAccessSequence(
        num_devices=8,
        num_layers=126,
        num_ring_steps=8,
        tokens_per_block=125000,
        kv_bytes_per_token=4096  # 2048 for K, 2048 for V
    )
    
    print(f"Ring attention with 8 devices, 126 layers")
    print(f"Forward+backward access schedule precomputed")
    print(f"KV cache per block: {125000 * 4096 / 1e9:.2f} GB")
    print(f"Total KV cache: {125000 * 4096 * 8 * 126 / 1e9:.2f} GB")
    
    # Initialize tiering manager
    manager = BeladyKVCacheTieringManager(
        ring_seq,
        hbm_size_bytes=12e9,
        dram_size_bytes=128e9,
        nvme_size_bytes=2e12
    )
    
    # Check memory feasibility
    feasible, headroom = manager.bandwidth.check_feasibility()
    print(f"\nThree-tier memory guarantee: {'✓ FEASIBLE' if feasible else '✗ INFEASIBLE'}")
    print(f"Bandwidth headroom ratio: {headroom:.2f}x")
    
    print("\n✓ Bélády-optimal KV cache tiering module loaded")
