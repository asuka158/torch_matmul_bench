"""
CUTLASS MXFP4 x MXFP4 -> fp32 dense GEMM (D = A @ B) -- run30 + window-AVG telemetry variant.

Same kernel call + data generation as bench_mxfp4_fp32_cut.py; only the telemetry aggregation differs:
  * num_tests = 30  (same as the original run30)
  * sm_mhz / power_w = AVERAGE of NVML samples in the bench_kineto window (original used median)
Everything else identical: bench_kineto(flush_l2=True) CUPTI pure-kernel time, tflops=2mnk/t,
gbps=count_bytes/t, 50 shapes from shape.csv, same 9-column CSV schema.

Run:
  cd python/new_test_mxfp4_fp32_cut
  LD_LIBRARY_PATH=/opt/hpcx/ucx/lib \
  PYTHONPATH=/root/workspace/gb_gemm_bench/python:/root/workspace/gb_gemm_bench/DeepGEMM \
  python bench_mxfp4_fp32_cut_run30_avg.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # python/
from bench_common import run_suite_run30  # noqa: E402

_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'result',
                    'mxfp4_fp32_dense_CUTLASS_run30_avg_50shape.csv')

if __name__ == '__main__':
    run_suite_run30('mxfp4', torch.float32, _OUT, 'cutlass_mxfp4_fp32_run30_avg',
                    num_tests=30, agg='avg')
