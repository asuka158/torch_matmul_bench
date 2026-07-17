"""Support probe for the 4 DENSE kernels (no perf, just "does it run + is it numerically sane
+ which backend kernel actually executed").

  1. nvfp4*nvfp4=bf16 dense, CUTLASS : sgl_kernel.cutlass_scaled_fp4_mm
  2. nvfp4*nvfp4=bf16 dense, cuBLASLt: torch._scaled_mm_v2 (BlockWise1x16, SWIZZLE_32_4_4)
  3. mxfp8*mxfp8=bf16 dense, CUTLASS : (no sgl_kernel op; see probe output)
  4. mxfp8*mxfp8=bf16 dense, cuBLASLt: torch._scaled_mm_v2 (BlockWise1x32, SWIZZLE_32_4_4)

Run: /opt/uv/bin/python probe_dense.py
"""
import sys
import torch
from torch._C import _ScalingType as ScalingType, _SwizzleType as SwizzleType

from quant_utils import quant_mxfp8, nvfp4_quant, to_blocked

M, N, K = 512, 4096, 7168
torch.manual_seed(0)


def relerr(out, ref):
    return ((out.float() - ref).norm() / ref.norm()).item()


def gpu_kernel_names(fn):
    """Run fn under the profiler, return CUDA kernel names."""
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
        print(f'  FAIL  {type(e).__name__}: {e}')
    print()


a_hp = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
b_hp = torch.randn(N, K, dtype=torch.bfloat16, device='cuda')  # weight [N, K]
ref = (a_hp.float() @ b_hp.float().t())


# 1. CUTLASS nvfp4 dense --------------------------------------------------------------
def t_cutlass_nvfp4():
    from sgl_kernel import cutlass_scaled_fp4_mm
    a_fp4, a_sf, gs_a = nvfp4_quant(a_hp)
    b_fp4, b_sf, gs_b = nvfp4_quant(b_hp)
    alpha = (1.0 / (gs_a * gs_b)).float()
    out = cutlass_scaled_fp4_mm(a_fp4, b_fp4, a_sf, b_sf, alpha, torch.bfloat16)
    names = gpu_kernel_names(lambda: cutlass_scaled_fp4_mm(a_fp4, b_fp4, a_sf, b_sf, alpha, torch.bfloat16))
    return relerr(out, ref), names


# 2. cuBLASLt nvfp4 dense (torch._scaled_mm_v2) ---------------------------------------
def t_cublas_nvfp4():
    a_fp4, a_sf, gs_a = nvfp4_quant(a_hp)   # [M, K/2] uint8, swizzled e4m3 sf
    b_fp4, b_sf, gs_b = nvfp4_quant(b_hp)   # [N, K/2]
    mat_a = a_fp4.view(torch.float4_e2m1fn_x2)
    mat_b = b_fp4.view(torch.float4_e2m1fn_x2).t()  # [K/2, N] col-major view
    rec = [ScalingType.BlockWise1x16.value]
    swz = [SwizzleType.SWIZZLE_32_4_4.value]

    def run():
        return torch._scaled_mm_v2(
            mat_a, mat_b,
            [a_sf], rec, swz,
            [b_sf], rec, swz,
            None, torch.bfloat16, (), False,
        )
    out = run().float() / (gs_a * gs_b)  # global scales are not part of the v2 call
    return relerr(out, ref), gpu_kernel_names(run)


# 4. cuBLASLt mxfp8 dense (torch._scaled_mm_v2) ---------------------------------------
def t_cublas_mxfp8():
    a_q, a_s = quant_mxfp8(a_hp)
    b_q, b_s = quant_mxfp8(b_hp)
    sa = to_blocked(a_s)
    sb = to_blocked(b_s)
    mat_b = b_q.t()  # [K, N] col-major
    rec = [ScalingType.BlockWise1x32.value]
    swz = [SwizzleType.SWIZZLE_32_4_4.value]

    def run():
        return torch._scaled_mm_v2(
            a_q, mat_b,
            [sa], rec, swz,
            [sb], rec, swz,
            None, torch.bfloat16, (), False,
        )
    out = run()
    return relerr(out, ref), gpu_kernel_names(run)


# 3. CUTLASS mxfp8 dense: enumerate what sgl_kernel offers ----------------------------
def t_cutlass_mxfp8():
    import sgl_kernel
    cand = [n for n in dir(sgl_kernel) if 'mx' in n.lower()]
    raise RuntimeError(f'no mxfp8 (1x32 e8m0) gemm in sgl_kernel {sgl_kernel.__version__}; mx-named symbols: {cand}')


if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    tests = {
        'cutlass_nvfp4': t_cutlass_nvfp4,
        'cublas_nvfp4': t_cublas_nvfp4,
        'cutlass_mxfp8': t_cutlass_mxfp8,
        'cublas_mxfp8': t_cublas_mxfp8,
    }
    for name, fn in tests.items():
        if which in ('all', name):
            report(name, fn)
