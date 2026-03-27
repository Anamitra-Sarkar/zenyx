"""
ZENYX Phase 9: Dynamic Ring Degree Curriculum

Implements gradual context length growth during training using curriculum learning.
Instead of starting with full 1M context, gradually increase via dynamic ring degree:

Ring degree progression:
  1 ring step (1 device only) -> 8K context
  2 ring steps (2 devices)     -> 16K context  
  4 ring steps (4 devices)     -> 32K context
  8 ring steps (8 devices)     -> 128K context (or up to 1M with more devices)

Benefits:
- Warmup training at small context (faster, more stable)
- Gradual difficulty increase (curriculum learning)
- Avoid early training instability with huge context
- Schedule: exponential growth (doubling) or mixed linear+exponential

Constraints:
- JAX requires static shapes at compile time
- Changing ring degree = changing sequence partitioning = requires re-JIT
- Can use dynamic_slice + collective_permute but still needs recompile
- Solution: Accept recompilation cost during curriculum transitions

Refs: Phase 9 of Zenyx papers, Apple VSL (ICML 2024), BigBird
"""

from typing import List, Tuple, Optional, Callable, Dict, Any
from dataclasses import dataclass
from enum import Enum
import math


class CurriculumScheduleType(Enum):
    """Type of curriculum schedule"""
    LINEAR = "linear"              # Linear growth: 8K -> 16K -> 32K -> ...
    EXPONENTIAL = "exponential"    # Exponential (doubling): 8K -> 16K -> 32K -> ...
    CYCLIC = "cyclic"              # Cycle short/long: 8K -> 16K -> 8K -> 32K -> ...
    MIXED = "mixed"                # Mix of linear and exponential phases


@dataclass
class RingDegreeCurriculumConfig:
    """Configuration for dynamic ring degree curriculum"""
    
    # Initial and final ring degrees (1 = 1 device, 2 = 2 devices, etc.)
    initial_ring_degree: int = 1      # Start: 8K context (1 device only)
    final_ring_degree: int = 8        # End: 128K context (all 8 devices)
    
    # Context lengths per ring degree
    # ring_degree: 1   2     4     8      16      32
    # context:     8K  16K   32K   128K   256K    1M (hypothetical)
    tokens_per_device: int = 8000     # Base tokens per ring step
    
    # Curriculum schedule
    schedule_type: CurriculumScheduleType = CurriculumScheduleType.EXPONENTIAL
    total_training_steps: int = 50000
    
    # For exponential: steps per ring degree
    # For linear: linear interpolation of ring degree
    steps_per_degree_increase: Optional[int] = None  # Auto-calculate if None
    
    # Reshard cost
    reshard_communication_cost_sec: float = 1.0  # ~1-3 seconds for full reshard
    
    @property
    def steps_between_reshards(self) -> int:
        """How many training steps between ring degree increases"""
        if self.steps_per_degree_increase:
            return self.steps_per_degree_increase
        
        # Auto-calculate: divide training into phases
        num_increases = math.ceil(math.log2(self.final_ring_degree / self.initial_ring_degree))
        return max(1000, self.total_training_steps // (num_increases + 1))


class RingDegreeScheduler:
    """
    Manages curriculum learning through dynamic ring degree changes.
    
    Schedules when to increase ring degree (more devices active),
    tracking current context length and predicting next reshard point.
    """
    
    def __init__(self, config: RingDegreeCurriculumConfig):
        self.config = config
        self.schedule = self._build_schedule()
        self.current_idx = 0
    
    def _build_schedule(self) -> List[Tuple[int, int, int]]:
        """
        Build curriculum schedule.
        
        Returns:
            List of (step, ring_degree, context_tokens)
        """
        schedule = []
        
        if self.config.schedule_type == CurriculumScheduleType.EXPONENTIAL:
            # Doubling schedule: 1 -> 2 -> 4 -> 8
            ring_degree = self.config.initial_ring_degree
            step = 0
            
            while ring_degree <= self.config.final_ring_degree:
                context_tokens = ring_degree * self.config.tokens_per_device
                schedule.append((step, ring_degree, context_tokens))
                
                # Move to next phase
                step += self.config.steps_between_reshards
                ring_degree *= 2
        
        elif self.config.schedule_type == CurriculumScheduleType.LINEAR:
            # Linear interpolation
            steps_between = self.config.steps_between_reshards
            ring_degree = self.config.initial_ring_degree
            step = 0
            
            while ring_degree <= self.config.final_ring_degree:
                context_tokens = ring_degree * self.config.tokens_per_device
                schedule.append((step, ring_degree, context_tokens))
                ring_degree += 1
                step += steps_between
        
        elif self.config.schedule_type == CurriculumScheduleType.CYCLIC:
            # Cycle between short and long contexts
            ring_degrees = [1, 2, 1, 4, 2, 8, 4]  # Custom pattern
            steps_between = self.config.steps_between_reshards
            
            for i, rd in enumerate(ring_degrees):
                if rd > self.config.final_ring_degree:
                    break
                context_tokens = rd * self.config.tokens_per_device
                step = i * steps_between
                schedule.append((step, rd, context_tokens))
        
        elif self.config.schedule_type == CurriculumScheduleType.MIXED:
            # Mix: linear for first half, exponential for second half
            mid_step = self.config.total_training_steps // 2
            half_steps = self.config.steps_between_reshards
            
            # Linear phase: 1 -> 2 -> 4
            for i in range(3):
                ring_degree = self.config.initial_ring_degree + i
                context_tokens = ring_degree * self.config.tokens_per_device
                step = i * half_steps
                schedule.append((step, ring_degree, context_tokens))
                if ring_degree >= 4:
                    break
            
            # Exponential phase: 4 -> 8 -> 16
            ring_degree = 4
            while ring_degree <= self.config.final_ring_degree:
                context_tokens = ring_degree * self.config.tokens_per_device
                step = mid_step + (math.log2(ring_degree / 4) * half_steps)
                schedule.append((step, ring_degree, context_tokens))
                ring_degree *= 2
        
        return sorted(schedule, key=lambda x: x[0])
    
    def get_ring_degree_at_step(self, step: int) -> Tuple[int, int]:
        """
        Get current ring degree and context length at training step.
        
        Args:
            step: Training step number
        
        Returns:
            (ring_degree, context_tokens)
        """
        # Binary search in schedule
        for i in range(len(self.schedule) - 1, -1, -1):
            if step >= self.schedule[i][0]:
                return self.schedule[i][1], self.schedule[i][2]
        
        # Before first schedule point
        return self.config.initial_ring_degree, self.schedule[0][2]
    
    def get_next_reshard_step(self, current_step: int) -> Optional[int]:
        """Get next step where ring degree will increase"""
        for schedule_step, _, _ in self.schedule:
            if schedule_step > current_step:
                return schedule_step
        return None
    
    def should_reshard(self, current_step: int, prev_step: int) -> bool:
        """Check if we've crossed a reshard boundary"""
        if prev_step < 0:
            return False
        
        curr_rd, _ = self.get_ring_degree_at_step(current_step)
        prev_rd, _ = self.get_ring_degree_at_step(prev_step)
        return curr_rd != prev_rd
    
    def get_schedule_summary(self) -> str:
        """Return human-readable schedule summary"""
        lines = ["\nRing Degree Curriculum Schedule:"]
        lines.append("=" * 60)
        lines.append(f"Schedule type: {self.config.schedule_type.value}")
        lines.append(f"Initial degree: {self.config.initial_ring_degree} ({self.config.initial_ring_degree * self.config.tokens_per_device:,} tokens)")
        lines.append(f"Final degree: {self.config.final_ring_degree} ({self.config.final_ring_degree * self.config.tokens_per_device:,} tokens)")
        lines.append(f"Total training steps: {self.config.total_training_steps:,}")
        lines.append(f"Steps between increases: {self.config.steps_between_reshards:,}\n")
        
        lines.append("Phase breakdown:")
        for i, (step, ring_degree, context_tokens) in enumerate(self.schedule):
            if i < len(self.schedule) - 1:
                duration = self.schedule[i + 1][0] - step
            else:
                duration = self.config.total_training_steps - step
            
            lines.append(f"  Step {step:6d} - {step+duration:6d}: "
                        f"Ring degree {ring_degree} "
                        f"({context_tokens:,} tokens, {duration:,} steps)")
        
        lines.append("=" * 60)
        return "\n".join(lines)


class ReshardingCostAnalyzer:
    """
    Analyzes communication cost of ring degree changes.
    
    When ring degree changes from R to 2R, each device's token count halves.
    All-to-all communication moves tokens to new partitions.
    """
    
    def __init__(self, num_devices: int = 8, ici_bandwidth_gbs: float = 400.0):
        self.num_devices = num_devices
        self.ici_bandwidth_gbs = ici_bandwidth_gbs  # ICI bandwidth in GB/s
    
    def compute_reshard_cost(self,
                            from_ring_degree: int,
                            to_ring_degree: int,
                            tokens_per_device: int,
                            bytes_per_token: int = 4096,
                            num_layers: int = 126) -> Dict[str, float]:
        """
        Compute reshard cost when changing ring degree.
        
        Example: 1->2 (halving tokens per device)
        Total tokens in system = constant (1M)
        Data to move = all activation data across all layers
        
        Args:
            from_ring_degree: Current ring degree
            to_ring_degree: Target ring degree
            tokens_per_device: Tokens per device per ring step
            bytes_per_token: Size of activation data per token
            num_layers: Number of layers (all need to move data)
        
        Returns:
            Dict with cost metrics
        """
        # Total tokens in system
        total_tokens = from_ring_degree * tokens_per_device
        
        # Data that moves: all activations across all layers
        # When resharding from R to 2R, devices need to redistribute tokens
        # Each layer's activations must be moved and repartitioned
        
        if to_ring_degree > from_ring_degree:
            # Increasing ring degree - more devices active
            tokens_moved = total_tokens * num_layers
        else:
            # Decreasing ring degree - fewer devices active
            tokens_moved = total_tokens * num_layers
        
        data_moved_gb = (tokens_moved * bytes_per_token) / 1e9
        
        # Time to transfer at ICI bandwidth
        transfer_time_sec = data_moved_gb / self.ici_bandwidth_gbs
        
        # Synchronization overhead for collective operation
        sync_time_sec = 0.1  # ~100ms for all-to-all sync
        
        total_time_sec = transfer_time_sec + sync_time_sec
        
        return {
            'total_tokens': total_tokens,
            'data_moved_gb': data_moved_gb,
            'transfer_time_sec': transfer_time_sec,
            'sync_time_sec': sync_time_sec,
            'total_time_sec': total_time_sec,
            'throughput_gbs': data_moved_gb / total_time_sec if total_time_sec > 0 else 0,
        }
    
    def get_cost_summary(self,
                        from_ring_degree: int,
                        to_ring_degree: int,
                        tokens_per_device: int,
                        num_layers: int = 126) -> str:
        """Get human-readable reshard cost summary"""
        cost = self.compute_reshard_cost(from_ring_degree, to_ring_degree,
                                        tokens_per_device, num_layers=num_layers)
        
        lines = [f"\nReshard cost (degree {from_ring_degree} -> {to_ring_degree}):"]
        lines.append(f"  Data moved: {cost['data_moved_gb']:.1f} GB")
        lines.append(f"  Transfer time: {cost['transfer_time_sec']:.2f} sec")
        lines.append(f"  Sync time: {cost['sync_time_sec']:.2f} sec")
        lines.append(f"  Total overhead: {cost['total_time_sec']:.2f} sec")
        lines.append(f"  Effective throughput: {cost['throughput_gbs']:.1f} GB/s")
        
        return "\n".join(lines)


# Example usage
if __name__ == "__main__":
    print("=" * 80)
    print("ZENYX Phase 9: Dynamic Ring Degree Curriculum")
    print("=" * 80)
    
    # Create exponential curriculum: 1 -> 2 -> 4 -> 8
    config = RingDegreeCurriculumConfig(
        initial_ring_degree=1,
        final_ring_degree=8,
        tokens_per_device=8000,
        schedule_type=CurriculumScheduleType.EXPONENTIAL,
        total_training_steps=50000,
        steps_per_degree_increase=10000,
    )
    
    scheduler = RingDegreeScheduler(config)
    print(scheduler.get_schedule_summary())
    
    # Test schedule at various steps
    print("\nSchedule sampling:")
    for step in [0, 5000, 10000, 20000, 30000, 40000, 50000]:
        ring_degree, context = scheduler.get_ring_degree_at_step(step)
        next_reshard = scheduler.get_next_reshard_step(step)
        print(f"  Step {step:5d}: Ring degree {ring_degree}, "
              f"Context {context:,} tokens, "
              f"Next reshard at step {next_reshard}")
    
    # Analyze reshard costs
    print("\n" + "=" * 80)
    print("Reshard Communication Costs:")
    print("=" * 80)
    
    analyzer = ReshardingCostAnalyzer()
    reshard_costs = [
        (1, 2),
        (2, 4),
        (4, 8),
    ]
    
    for from_rd, to_rd in reshard_costs:
        cost_summary = analyzer.get_cost_summary(from_rd, to_rd, 8000)
        print(cost_summary)
    
    print("\n" + "=" * 80)
    print("✓ Dynamic ring degree curriculum module ready")
    print("=" * 80)
