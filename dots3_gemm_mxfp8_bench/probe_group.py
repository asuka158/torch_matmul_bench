"""Support probe for the mxfp8 GROUP kernels with REALISTIC routing shapes.

The nvfp4-bench probe only used m_e = 256 (multiple of 128), which cannot
distinguish the two plausible per-group A-scale layouts of
torch._scaled_grouped_mm_v2 (cutlass_mslk backend):
  (a) concatenated per-group to_blocked() segments, each 128-row padded
      (the cutlass_fp4_group_mm blockscale_offsets convention), vs
  (b) scale pointer advanced by offs[g-1] rows of the UNPADDED scale matrix.
This probe uses uneven m_e (incl. 0 and < 32) to find the correct one.

Run: /opt/uv/bin/python probe_group.py [m_e list, comma separated]
"""
import sys

import torch
from torch._C import _ScalingType as ScalingType, _SwizzleType as SwizzleType

from quant_utils import ceil_div, quant_mxfp8, to_blocked

N, K = 768, 5120
M_LIST = ([int(x) for x in sys.argv[1].split(',')] if len(sys.argv) > 1
          else [3, 0, 17, 129, 64, 1, 250, 33])
E = len(M_LIST)
M_TOT = sum(M_LIST)
torch.manual_seed(0)

a_hp = torch.randn(M_TOT, K, dtype=torch.bfloat16, device='cuda')
w_hp = torch.randn(E, N, K, dtype=torch.bfloat16, device='cuda')

offs = torch.tensor(M_LIST, dtype=torch.int32, device='cuda').cumsum(0).to(torch.int32)
bounds = [0] + offs.tolist()

ref = torch.empty(M_TOT, N, dtype=torch.float32, device='cuda')
for e in range(E):
    lo, hi = bounds[e], bounds[e + 1]
    if hi > lo:
        ref[lo:hi] = a_hp[lo:hi].float() @ w_hp[e].float().t()

a_q, a_s = quant_mxfp8(a_hp)          # [M_TOT, K] e4m3, [M_TOT, K/32] e8m0 unswizzled
wq_list, ws_list = [], []
for e in range(E):
    q, s = quant_mxfp8(w_hp[e])
    wq_list.append(q)
    ws_list.append(to_blocked(s))
b_q = torch.stack(wq_list)                       # [E, N, K]
sb = torch.stack(ws_list).view(E, -1)            # [E, round_up(N,128)*round_up(K/32,4)]
mat_b = b_q.transpose(-2, -1)                    # [E, K, N] (col-major per expert)
rec = [ScalingType.BlockWise1x32.value]
swz = [SwizzleType.SWIZZLE_32_4_4.value]


def try_layout(tag, sa):
    run = lambda: torch._scaled_grouped_mm_v2(
        a_q, mat_b, [sa], rec, swz, [sb], rec, swz,
        offs, None, torch.bfloat16, (), False)
    try:
        out = run()
        torch.cuda.synchronize()
        err = ((out.float() - ref).norm() / ref.norm()).item()
        per_g = []
        for e in range(E):
            lo, hi = bounds[e], bounds[e + 1]
            if hi > lo:
                d = ref[lo:hi]
                per_g.append(((out[lo:hi].float() - d).norm() / d.norm()).item())
        print(f'{tag}: relerr={err:.4e}  per-group max={max(per_g):.4e}')
    except Exception as ex:
        print(f'{tag}: FAIL {type(ex).__name__}: {str(ex)[:300]}')


# (a) per-group 128-row-padded swizzled segments
sa_padded = torch.cat([
    to_blocked(a_s[bounds[e]:bounds[e + 1]]).view(torch.uint8) if M_LIST[e] > 0
    else torch.empty(0, dtype=torch.uint8, device='cuda')
    for e in range(E)
]).view(torch.float8_e8m0fnu)
print(f'E={E} m_e={M_LIST} sum={M_TOT}; sa_padded numel={sa_padded.numel()} '
      f'(= sum round_up(m_e,128)*{ceil_div(K // 32, 4) * 4})')
cols = ceil_div(K // 32, 4) * 4
try_layout('(a) padded per-group segments', sa_padded.view(-1, cols))

# (b) whole-matrix swizzle, pointer advance by raw row offset
sa_whole = to_blocked(a_s)
print(f'sa_whole numel={sa_whole.numel()}')
try_layout('(b) whole-matrix to_blocked  ', sa_whole.view(-1, cols))

# (c) cublasLt grouped mxfp8 ext, fed the SAME buffers/layout as (a)
import cublaslt_group_gemm_mx as cmx  # noqa: E402

expert_offsets = torch.tensor(bounds[:-1], dtype=torch.int32, device='cuda')
bs_rows = [0]
for m in M_LIST:
    bs_rows.append(bs_rows[-1] + ceil_div(m, 128) * 128)
blockscale_offsets = torch.tensor(bs_rows[:-1], dtype=torch.int32, device='cuda')
problem_sizes = torch.zeros(E, 3, dtype=torch.int32, device='cuda')
problem_sizes[:, 0] = torch.tensor(M_LIST, dtype=torch.int32, device='cuda')
params = {'expert_offsets': expert_offsets, 'blockscale_offsets': blockscale_offsets,
          'problem_sizes': problem_sizes}
run_lt = lambda: cmx.cublaslt_mxfp8_group_mm(
    a_q, b_q, sa_padded, torch.stack(ws_list).view(E, -1), torch.bfloat16, 'cuda', params)
out_lt = run_lt()
torch.cuda.synchronize()
err = ((out_lt.float() - ref).norm() / ref.norm()).item()
out_ct = torch._scaled_grouped_mm_v2(
    a_q, mat_b, [sa_padded.view(-1, cols)], rec, swz, [sb], rec, swz,
    offs, None, torch.bfloat16, (), False)
print(f'(c) cublasLt ext              : relerr={err:.4e}  '
      f'bitwise==torch-grouped: {torch.equal(out_lt, out_ct)}  '
      f'maxdiff={(out_lt.float() - out_ct.float()).abs().max().item():.5f}')
