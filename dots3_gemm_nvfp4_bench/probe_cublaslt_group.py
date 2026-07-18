"""Verify the cuBLASLt NVFP4 grouped GEMM python entry (cublaslt_group_mm.py).

Builds one grouped problem (E=4, 256 tokens/expert, N=4096, K=7168) with
per-tensor-global weight scale shared by ALL experts (=> equal alphas, the
cublasLt single-scalar-alpha semantics both backends are aligned to), then:
  1. runs sgl_kernel.cutlass_fp4_group_mm and cublaslt_fp4_group_mm on the SAME
     quantized buffers,
  2. compares the two outputs bitwise + both against the fp32 reference,
  3. prints every CUDA kernel each call launches (for bench kernel_substr).

Run: /opt/uv/bin/python probe_cublaslt_group.py
"""
import torch
from sgl_kernel import (
    cutlass_fp4_group_mm, prepare_moe_input, scaled_fp4_experts_quant,
    scaled_fp4_quant, shuffle_rows,
)

from quant_utils import FLOAT8_E4M3_MAX, FLOAT4_E2M1_MAX
from cublaslt_group_mm import cublaslt_fp4_group_mm

E, M_PER_E, N, K = 4, 256, 4096, 7168
M_TOT = E * M_PER_E
torch.manual_seed(0)
dev = 'cuda'

a_hp = torch.randn(M_TOT, K, dtype=torch.bfloat16, device=dev)
w_hp = torch.randn(E, N, K, dtype=torch.bfloat16, device=dev)

ref = torch.empty(M_TOT, N, dtype=torch.float32, device=dev)
for e in range(E):
    sl = slice(e * M_PER_E, (e + 1) * M_PER_E)
    ref[sl] = a_hp[sl].float() @ w_hp[e].float().t()

# --- routing / offsets (topk=1, tokens already grouped by expert) -------------------
topk_ids = (torch.arange(M_TOT, device=dev, dtype=torch.int32) // M_PER_E).view(M_TOT, 1)
expert_offsets = torch.empty(E + 1, dtype=torch.int32, device=dev)
blockscale_offsets = torch.empty(E + 1, dtype=torch.int32, device=dev)
problem_sizes1 = torch.empty(E, 3, dtype=torch.int32, device=dev)
problem_sizes2 = torch.empty(E, 3, dtype=torch.int32, device=dev)
a_map = torch.empty(M_TOT, dtype=torch.int32, device=dev)
c_map = torch.empty(M_TOT, dtype=torch.int32, device=dev)
prepare_moe_input(topk_ids, expert_offsets, problem_sizes1, problem_sizes2,
                  a_map, c_map, E, N // 2, K, blockscale_offsets)

# --- quantization: ONE global scale for a, ONE for all expert weights ----------------
a_gscale = ((FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / a_hp.abs().amax().float()).repeat(E)
rep_a_fp4, rep_a_blockscale = scaled_fp4_experts_quant(
    a_hp, a_gscale, expert_offsets, blockscale_offsets, 1, expert_map=a_map)

w_gscale = (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / w_hp.abs().amax().float()  # shared by all experts
w_fp4_l, w_bs_l = [], []
for e in range(E):
    qd, sf = scaled_fp4_quant(w_hp[e], w_gscale)
    w_fp4_l.append(qd); w_bs_l.append(sf)
w_fp4 = torch.stack(w_fp4_l).contiguous()          # [E, N, K/2]
w_blockscale = torch.stack(w_bs_l).contiguous()    # [E, N, K/16] (N mult of 128)
alphas = (1.0 / (a_gscale * w_gscale)).float()     # [E], all equal by construction

params = {
    'ab_strides': torch.full((E,), K, dtype=torch.int64, device=dev),
    'c_strides': torch.full((E,), N, dtype=torch.int64, device=dev),
    'problem_sizes': problem_sizes1,
    'expert_offsets': expert_offsets[:-1],
    'blockscale_offsets': blockscale_offsets[:-1],
}

run_ct = lambda: cutlass_fp4_group_mm(
    rep_a_fp4, w_fp4, rep_a_blockscale, w_blockscale, alphas, torch.bfloat16, dev, params)
run_lt = lambda: cublaslt_fp4_group_mm(
    rep_a_fp4, w_fp4, rep_a_blockscale, w_blockscale, alphas, torch.bfloat16, dev, params)


def relerr(out):
    o = shuffle_rows(out, c_map, (M_TOT, N)).float()
    return ((o - ref).norm() / ref.norm()).item()


def kernel_names(fn):
    fn(); torch.cuda.synchronize()  # warmup (JIT plan/heuristic happens here)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as p:
        fn(); torch.cuda.synchronize()
    return [e.name for e in p.events() if e.device_type == torch.profiler.DeviceType.CUDA]


out_ct = run_ct()
out_lt = run_lt()
print(f'cutlass  relerr vs fp32 : {relerr(out_ct):.4e}')
print(f'cublaslt relerr vs fp32 : {relerr(out_lt):.4e}')
print(f'bitwise identical       : {torch.equal(out_ct, out_lt)}')
mism = (out_ct != out_lt).sum().item()
print(f'mismatching elements    : {mism}/{out_ct.numel()}'
      + ('' if mism == 0 else f'  max_abs_diff={(out_ct.float()-out_lt.float()).abs().max().item():.6f}'))

print('\n[cutlass call] device activities:')
for n in kernel_names(run_ct):
    print('  ', n[:130])
print('\n[cublaslt call] device activities:')
for n in kernel_names(run_lt):
    print('  ', n[:130])
