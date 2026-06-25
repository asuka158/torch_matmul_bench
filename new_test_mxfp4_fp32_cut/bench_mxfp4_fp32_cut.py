"""
CUTLASS MXFP4 x MXFP4 -> fp32 dense GEMM (D = A @ B, no C) via torch._scaled_mm_v2.

Kernel call path (correct, taken from python/mxfp4/bench_mxfp4_fp32.py + bench_common.prep):
  MXFP4 operands are built once per shape with torchao MXTensor (E2M1 data + UE8M0 1x32
  block-scales, scales pre-swizzled); the timed call is a bare torch._scaled_mm_v2 with
  ScalingType.BlockWise1x32. In torch's scaled ops MXFP4 always dispatches to CUTLASS
  (cuBLASLt has no MXFP4), so format == backend.

Measurement: identical to DeepGEMM/tests/dg_test/bench_fp4_run30.py (run_suite_run30):
  * t = bench_kineto(run, "cutlass", num_tests=30, flush_l2=True)  -> CUPTI pure-kernel time
  * sm_mhz / power_w = NVML MEDIAN over the window wrapping that single bench_kineto call
  * tflops = 2*m*n*k / t ;  gbps = count_bytes(A_fp4,A_scale,B_fp4,B_scale,D) / t
  * CSV columns: m,n,k,us,tflops,gbps,sm_mhz,power_w,backend
  * 50 shapes from /root/workspace/gb_gemm_bench/shape.csv (row M,K,N -> m,n,k)

Run:
  cd python/new_test_mxfp4_fp32_cut
  LD_LIBRARY_PATH=/opt/hpcx/ucx/lib \
  PYTHONPATH=/root/workspace/gb_gemm_bench/python:/root/workspace/gb_gemm_bench/DeepGEMM \
  python bench_mxfp4_fp32_cut.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # python/
from bench_common import run_suite_run30  # noqa: E402

_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'result',
                    'mxfp4_fp32_dense_CUTLASS_run30_50shape.csv')

if __name__ == '__main__':
    run_suite_run30('mxfp4', torch.float32, _OUT, 'cutlass_mxfp4_fp32_run30')
