"""CUTLASS MXFP4 x MXFP4 -> bf16 dense GEMM (D = A @ B) via torch._scaled_mm_v2."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # python/
from bench_common import run_suite  # noqa: E402

_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'result', 'mxfp4_bf16.csv')

if __name__ == '__main__':
    run_suite('mxfp4', torch.bfloat16, _OUT, 'cutlass_mxfp4_bf16')
