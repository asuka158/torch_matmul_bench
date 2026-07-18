"""Support probe for the mxfp8 DENSE kernels.

  CUTLASS : cutlass_dense_mx JIT (2 compiled configs: 2SM 256x256x128, 1SM 128x128x128)
  cuBLASLt: torch._scaled_mm_v2 (BlockWise1x32 + SWIZZLE_32_4_4) -> nvjet

Checks vs fp32 reference on randn data (expected mxfp8 quant relerr ~3.77e-2) and
cutlass-vs-cublas bitwise comparison on the same quantized buffers.

Run: /opt/uv/bin/python probe_dense.py [M N K]
"""
import sys

import torch
from torch._C import _ScalingType as ScalingType, _SwizzleType as SwizzleType

from quant_utils import ceil_div, quant_mxfp8, to_blocked

M, N, K = (int(x) for x in sys.argv[1:4]) if len(sys.argv) > 3 else (513, 768, 5120)
torch.manual_seed(0)

a_hp = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
b_hp = torch.randn(N, K, dtype=torch.bfloat16, device='cuda')
ref = a_hp.float() @ b_hp.float().t()

a_q, a_s = quant_mxfp8(a_hp)
b_q, b_s = quant_mxfp8(b_hp)
sa = to_blocked(a_s)
sb = to_blocked(b_s)
alpha = torch.ones(1, dtype=torch.float32, device='cuda')

cols = ceil_div(K // 32, 4) * 4
rec = [ScalingType.BlockWise1x32.value]
swz = [SwizzleType.SWIZZLE_32_4_4.value]
run_lt = lambda: torch._scaled_mm_v2(
    a_q, b_q.t(), [sa.view(-1, cols)], rec, swz, [sb.view(-1, cols)], rec, swz,
    None, torch.bfloat16, (), False)

import cutlass_dense_mx as cdm  # noqa: E402
print('building cutlass_dense_mx (first build takes minutes)...', flush=True)
cdm.load_ext(verbose=False)
print('built.', flush=True)

outs = {}
for tag, run in (('cutlass 2SM', lambda: cdm.mxfp8_mm_2sm(a_q, b_q, sa, sb, alpha)),
                 ('cutlass 1SM', lambda: cdm.mxfp8_mm_1sm(a_q, b_q, sa, sb, alpha)),
                 ('cublas nvjet', run_lt)):
    out = run()
    torch.cuda.synchronize()
    err = ((out.float() - ref).norm() / ref.norm()).item()
    outs[tag] = out
    print(f'{tag:12}: relerr={err:.4e}')

print(f"bitwise 2SM==1SM   : {torch.equal(outs['cutlass 2SM'], outs['cutlass 1SM'])}")
print(f"bitwise 2SM==cublas: {torch.equal(outs['cutlass 2SM'], outs['cublas nvjet'])}")
d = (outs['cutlass 2SM'].float() - outs['cublas nvjet'].float()).abs()
print(f'maxdiff cutlass-vs-cublas: {d.max().item():.6f}')
