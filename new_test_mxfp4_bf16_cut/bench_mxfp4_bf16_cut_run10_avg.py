"""
CUTLASS MXFP4 x MXFP4 -> bf16 dense GEMM (D = A @ B) -- run10 + window-AVG telemetry variant.

bf16 counterpart of new_test_mxfp4_fp32_cut/bench_mxfp4_fp32_cut_run10_avg.py: identical kernel
call + data generation (bare torch._scaled_mm_v2, MXFP4 -> CUTLASS), only the output dtype changes
(torch.bfloat16 instead of torch.float32). Measurement knobs unchanged:
  * num_tests = 10
  * sm_mhz / power_w = AVERAGE of NVML samples in the bench_kineto window
bench_kineto(flush_l2=True) CUPTI pure-kernel time, tflops=2mnk/t, gbps=count_bytes/t,
50 shapes from shape.csv, same 9-column CSV schema.

Run:
  cd torch_matmul_bench/new_test_mxfp4_bf16_cut
  LD_LIBRARY_PATH=/opt/hpcx/ucx/lib \
  PYTHONPATH=/root/workspace/gb_gemm_bench/python:/root/workspace/gb_gemm_bench/DeepGEMM \
  python bench_mxfp4_bf16_cut_run10_avg.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # torch_matmul_bench/
from bench_common import run_suite_run30  # noqa: E402

_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'result',
                    'mxfp4_bf16_dense_CUTLASS_run10_avg_50shape.csv')

if __name__ == '__main__':
    run_suite_run30('mxfp4', torch.bfloat16, _OUT, 'cutlass_mxfp4_bf16_run10_avg',
                    num_tests=10, agg='avg')
