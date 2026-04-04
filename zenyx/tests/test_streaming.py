import torch
import torch.nn as nn

from zenyx.streaming import (
    BandwidthAwareScheduler,
    BandwidthSample,
    BeladyApproxCache,
    StreamingExecutionEngine,
    quantize_to_fp8_storage,
    dequantize_for_bf16_compute,
)


def test_belady_eviction_prefers_farthest_next_use():
    cache = BeladyApproxCache(capacity=2)
    cache.update_future("a", 1)
    cache.update_future("b", 100)
    cache.update_future("c", 2)
    cache.touch("a")
    cache.touch("b")
    evicted = cache.touch("c")
    assert evicted == ["b"]


def test_bandwidth_scheduler_throttle_decision():
    sched = BandwidthAwareScheduler(max_fetch_to_compute_ratio=1.0)
    assert sched.should_throttle(BandwidthSample(fetch_time_ms=2.0, compute_time_ms=1.0))
    assert not sched.should_throttle(BandwidthSample(fetch_time_ms=0.5, compute_time_ms=1.0))


def test_precision_roundtrip_shape_dtype():
    x = torch.randn(4, 4)
    q, scale = quantize_to_fp8_storage(x)
    y = dequantize_for_bf16_compute(q, scale)
    assert y.shape == x.shape
    assert y.dtype == torch.bfloat16


def test_streaming_engine_cpu_smoke():
    model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))
    engine = StreamingExecutionEngine(model, device=torch.device("cpu"))
    x = torch.randn(2, 8)
    y = torch.randn(2, 4)
    loss = engine.run_forward_backward(x, y, nn.MSELoss())
    assert loss >= 0.0
