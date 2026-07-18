"""Support probe for the GROUP (grouped/MoE) kernels.

  5. nvfp4*nvfp4=bf16 group, CUTLASS : sgl_kernel.cutlass_fp4_group_mm (sglang cutlass_moe_fp4 path)
  6. nvfp4*nvfp4=bf16 group, cuBLASLt: torch._scaled_grouped_mm_v2 attempt (expect unsupported)
  7. mxfp8*mxfp8=bf16 group, CUTLASS?: torch._scaled_grouped_mm_v2 BlockWise1x32 (check kernel name)
  8. mxfp8*mxfp8=bf16 group, cuBLASLt: (C++ sample LtMxfp8gemmGroupedSimple, probed separately)

Setup: E=4 experts, 256 tokens per expert (multiple of 128 so swizzled scale
offsets are trivially aligned), K=7168, N=4096, weight [E, N, K].

Run: /opt/uv/bin/python probe_group.py [test]
"""
import sys
import torch
from torch._C import _ScalingType as ScalingType, _SwizzleType as SwizzleType

from quant_utils import quant_mxfp8, to_blocked, FLOAT8_E4M3_MAX, FLOAT4_E2M1_MAX

E, M_PER_E, N, K = 4, 256, 4096, 7168
M_TOT = E * M_PER_E
torch.manual_seed(0)

a_hp = torch.randn(M_TOT, K, dtype=torch.bfloat16, device='cuda')       # tokens already grouped by expert
w_hp = torch.randn(E, N, K, dtype=torch.bfloat16, device='cuda')         # per-expert weight [N, K]

ref = torch.empty(M_TOT, N, dtype=torch.float32, device='cuda')
for e in range(E):
    sl = slice(e * M_PER_E, (e + 1) * M_PER_E)
    ref[sl] = a_hp[sl].float() @ w_hp[e].float().t()


def relerr(out):
    return ((out.float() - ref).norm() / ref.norm()).item()


def gpu_kernel_names(fn):
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    return sorted({e.name for e in prof.events() if e.device_type == torch.profiler.DeviceType.CUDA and 'memcpy' not in e.name.lower() and 'memset' not in e.name.lower()})


def report(tag, fn):
    print(f'--- {tag} ---')
    try:
        err, names = fn()
        print(f'  PASS  relerr={err:.4e}')
        for n in names:
            print(f'  kernel: {n[:140]}')
    except Exception as e:
        print(f'  FAIL  {type(e).__name__}: {str(e)[:400]}')
    print()


# 5. CUTLASS nvfp4 group (sgl_kernel) --------------------------------------------------
def t_cutlass_nvfp4_group():
    from sgl_kernel import (
        cutlass_fp4_group_mm, prepare_moe_input, scaled_fp4_experts_quant,
        scaled_fp4_quant, shuffle_rows,
    )
    device = a_hp.device
    # topk=1 routing: token i -> expert i // M_PER_E (already grouped, permutation = identity)
    topk_ids = (torch.arange(M_TOT, device=device, dtype=torch.int32) // M_PER_E).view(M_TOT, 1)

    expert_offsets = torch.empty(E + 1, dtype=torch.int32, device=device)
    blockscale_offsets = torch.empty(E + 1, dtype=torch.int32, device=device)
    problem_sizes1 = torch.empty(E, 3, dtype=torch.int32, device=device)
    problem_sizes2 = torch.empty(E, 3, dtype=torch.int32, device=device)
    a_map = torch.empty(M_TOT, dtype=torch.int32, device=device)
    c_map = torch.empty(M_TOT, dtype=torch.int32, device=device)
    # prepare_moe_input's (n, k) convention: n = intermediate size (w1 is [E, 2n, K]) -> gemm1 N = 2n
    prepare_moe_input(topk_ids, expert_offsets, problem_sizes1, problem_sizes2,
                      a_map, c_map, E, N // 2, K, blockscale_offsets)

    a_gscale = ((FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / a_hp.abs().amax().float()).repeat(E)
    rep_a_fp4, rep_a_blockscale = scaled_fp4_experts_quant(
        a_hp, a_gscale, expert_offsets, blockscale_offsets, 1, expert_map=a_map)

    w_fp4_list, w_bs_list, w_gs_list = [], [], []
    for e in range(E):
        gs = (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / w_hp[e].abs().amax().float()
        qd, sf = scaled_fp4_quant(w_hp[e], gs)
        w_fp4_list.append(qd); w_bs_list.append(sf); w_gs_list.append(gs)
    w_fp4 = torch.stack(w_fp4_list)              # [E, N, K/2] uint8
    w_blockscale = torch.stack(w_bs_list)        # [E, N(pad128), K/16] e4m3 swizzled
    alphas = (1.0 / (a_gscale * torch.stack(w_gs_list))).float()

    params = {
        'ab_strides': torch.full((E,), K, dtype=torch.int64, device=device),
        'c_strides': torch.full((E,), N, dtype=torch.int64, device=device),
        'problem_sizes': problem_sizes1,
        'expert_offsets': expert_offsets[:-1],
        'blockscale_offsets': blockscale_offsets[:-1],
    }
    run = lambda: cutlass_fp4_group_mm(
        rep_a_fp4, w_fp4, rep_a_blockscale, w_blockscale, alphas,
        torch.bfloat16, device, params)
    out = shuffle_rows(run(), c_map, (M_TOT, N))  # un-permute rows back to token order
    return relerr(out), gpu_kernel_names(run)


# 7. torch grouped mxfp8 (backend TBD from kernel name) --------------------------------
def t_torch_mxfp8_group():
    device = a_hp.device
    a_q, a_s = quant_mxfp8(a_hp)                              # [M_TOT, K] e4m3, [M_TOT, K/32] e8m0
    # per-group swizzle of A scales (each group is 256 rows = 2*128, no padding needed)
    sa = torch.cat([to_blocked(a_s[e * M_PER_E:(e + 1) * M_PER_E]) for e in range(E)]).view(M_TOT, K // 32)
    wq_list, ws_list = [], []
    for e in range(E):
        q, s = quant_mxfp8(w_hp[e])
        wq_list.append(q); ws_list.append(to_blocked(s))
    b_q = torch.stack(wq_list)                                 # [E, N, K]
    sb = torch.stack(ws_list).view(E, -1)                      # [E, N*K/32] (swizzled bytes per expert)
    mat_b = b_q.transpose(-2, -1)                              # [E, K, N]
    offs = torch.arange(1, E + 1, device=device, dtype=torch.int32) * M_PER_E
    rec = [ScalingType.BlockWise1x32.value]
    swz = [SwizzleType.SWIZZLE_32_4_4.value]

    run = lambda: torch._scaled_grouped_mm_v2(
        a_q, mat_b, [sa], rec, swz, [sb], rec, swz,
        offs, None, torch.bfloat16, (), False)
    out = run()
    return relerr(out), gpu_kernel_names(run)


# 6. torch grouped nvfp4 attempt (expect unsupported -> cublas group nvfp4 needs C++) ---
def t_torch_nvfp4_group():
    from sgl_kernel import scaled_fp4_quant
    device = a_hp.device
    gs_a = (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / a_hp.abs().amax().float()
    a_fp4, a_sf = scaled_fp4_quant(a_hp, gs_a)   # swizzled already, per 128-row tiles
    wq_list, ws_list, gs_list = [], [], []
    for e in range(E):
        gs = (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / w_hp[e].abs().amax().float()
        q, sf = scaled_fp4_quant(w_hp[e], gs)
        wq_list.append(q); ws_list.append(sf); gs_list.append(gs)
    mat_a = a_fp4.view(torch.float4_e2m1fn_x2)
    mat_b = torch.stack(wq_list).view(torch.float4_e2m1fn_x2).transpose(-2, -1)  # [E, K/2, N]
    sb = torch.stack(ws_list)
    offs = torch.arange(1, E + 1, device=device, dtype=torch.int32) * M_PER_E
    rec = [ScalingType.BlockWise1x16.value]
    swz = [SwizzleType.SWIZZLE_32_4_4.value]

    run = lambda: torch._scaled_grouped_mm_v2(
        mat_a, mat_b, [a_sf], rec, swz, [sb], rec, swz,
        offs, None, torch.bfloat16, (), False)
    out = run().float() / (gs_a * torch.stack(gs_list).mean())  # rough: probe only
    return relerr(out), gpu_kernel_names(run)


if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    tests = {
        'cutlass_nvfp4_group': t_cutlass_nvfp4_group,
        'torch_mxfp8_group': t_torch_mxfp8_group,
        'torch_nvfp4_group': t_torch_nvfp4_group,
    }
    for name, fn in tests.items():
        if which in ('all', name):
            report(name, fn)
