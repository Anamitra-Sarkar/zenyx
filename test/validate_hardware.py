#!/usr/bin/env python3
"""
Zenyx Hardware Validation Report
Comprehensive testing of Zenyx capabilities across different hardware

Tests:
  1. CPU: Verify chunked attention up to 8K context
  2. GPU: Template for 2xT4 with 128K context (requires CUDA)
  3. TPU: Template for v5e-8 with 1M context and 1T params (requires torch_xla)
"""

import sys
import subprocess
from pathlib import Path

def run_test(test_file, test_name):
    """Run a test file and return success status"""
    print(f"\n{'='*80}")
    print(f"Running: {test_name}")
    print(f"Script: {test_file}")
    print(f"{'='*80}")
    
    try:
        result = subprocess.run(
            [sys.executable, str(test_file)],
            cwd=Path(__file__).parent,
            timeout=1200,  # 20 minutes
            capture_output=False,
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"❌ TIMEOUT: {test_name} exceeded 20 minutes")
        return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False


def main():
    """Run all available tests"""
    print("\n" + "="*80)
    print("ZENYX HARDWARE VALIDATION SUITE")
    print("="*80)
    
    app_dir = Path(__file__).parent
    results = {}
    
    # 1. CPU Test (always available)
    print("\n[1/3] CPU TRAINING TEST")
    cpu_test = app_dir / "test_cpu_8k_context.py"
    if cpu_test.exists():
        results["CPU (8K context)"] = run_test(cpu_test, "CPU Training with 8K context")
    else:
        print(f"❌ Test file not found: {cpu_test}")
        results["CPU (8K context)"] = False
    
    # 2. GPU Test (conditional)
    print("\n[2/3] GPU TRAINING TEST (if CUDA available)")
    gpu_test = app_dir / "test_gpu_128k_context.py"
    if gpu_test.exists():
        import torch
        if torch.cuda.is_available():
            results["GPU (128K context, 2xT4)"] = run_test(gpu_test, "GPU Training with 128K context")
        else:
            print("⏭️  SKIPPED: CUDA not available (GPU test requires NVIDIA GPU)")
            results["GPU (128K context, 2xT4)"] = None
    else:
        print(f"❌ Test file not found: {gpu_test}")
        results["GPU (128K context, 2xT4)"] = False
    
    # 3. TPU Test (conditional)
    print("\n[3/3] TPU TRAINING TEST (if torch_xla available)")
    tpu_test = app_dir / "test_tpu_1m_context.py"
    if tpu_test.exists():
        try:
            import torch_xla
            results["TPU (1M context, v5e-8)"] = run_test(tpu_test, "TPU Training with 1M context")
        except ImportError:
            print("⏭️  SKIPPED: torch_xla not installed (TPU test requires torch_xla)")
            results["TPU (1M context, v5e-8)"] = None
    else:
        print(f"❌ Test file not found: {tpu_test}")
        results["TPU (1M context, v5e-8)"] = False
    
    # Summary report
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    
    passed = 0
    skipped = 0
    failed = 0
    
    for test_name, result in results.items():
        if result is True:
            status = "✅ PASS"
            passed += 1
        elif result is False:
            status = "❌ FAIL"
            failed += 1
        else:
            status = "⏭️  SKIPPED"
            skipped += 1
        print(f"  {status}: {test_name}")
    
    print(f"\n{'='*80}")
    print(f"Results: {passed} passed, {failed} failed, {skipped} skipped")
    print(f"{'='*80}\n")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
