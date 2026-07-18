"""cublasLt grouped-matmul python wrapper for the MXFP8 MoE group gemms.

Clone of the nvfp4 wrapper (dots3_gemm_nvfp4_bench/cublaslt_group_gemm) adapted
to mxfp8: e4m3 data (1 byte/elem, no fp4 packing), e8m0 scales per 32 elems
along K in the swizzled 128x4 (32x4x4) layout, alpha fixed to 1.0 (mxfp8 has no
per-tensor global scale).

The extension is a single host-API .cpp JIT-compiled with torch.utils.cpp_extension
(~30 s once per node, cached in ~/.cache/torch_extensions). Needs CUDA_HOME.

  * One plan per weight tensor: created lazily on first call; the cublasLt
    heuristic runs once there; per the benchmark convention the plan always runs
    the heuristic's algo[0].
  * Every call rewrites only the CONTENTS of the device-side x/out/x-scale
    pointer arrays and marr (per-expert m) -- no host sync, no replan.
  * A-scale layout contract (same as cutlass_fp4_group_mm / torch grouped mm):
    concatenated per-expert swizzled segments, each 128-row padded;
    params['blockscale_offsets'][e] = cumsum of round_up(m_e, 128).
"""

import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_WORKSPACE_BYTES = 64 << 20

_ext = None
_plans = {}  # (weight data_ptr, E, N, K) -> _Plan
_workspace = None


def _find_cublaslt_libdir():
    # Prefer the cublasLt torch already loaded (nvidia pip wheel), fall back to toolkit.
    import glob
    torch_dir = os.path.dirname(torch.__file__)
    site = os.path.dirname(torch_dir)
    candidates = []
    candidates += glob.glob(os.path.join(site, "nvidia", "cublas", "lib"))
    candidates += glob.glob(os.path.join(torch_dir, "lib"))
    candidates.append("/usr/local/cuda/lib64")
    for c in candidates:
        if glob.glob(os.path.join(c, "libcublasLt.so*")):
            return c
    raise RuntimeError("libcublasLt not found")


def load_ext(verbose=False):
    global _ext
    if _ext is not None:
        return _ext
    from torch.utils.cpp_extension import load

    os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
    cuda_home = os.environ["CUDA_HOME"]
    libdir = _find_cublaslt_libdir()
    _ext = load(
        name="dots3_cublaslt_group_gemm_mx",
        sources=[os.path.join(_HERE, "csrc", "cublaslt_group_gemm_mx.cpp")],
        extra_cflags=["-O3", "-std=c++20"],
        extra_include_paths=[os.path.join(cuda_home, "include")],
        extra_ldflags=[
            f"-L{libdir}",
            "-lcublasLt",
            f"-Wl,-rpath,{libdir}",
            "-lcudart",
            f"-L{os.path.join(cuda_home, 'lib64')}",
        ],
        verbose=verbose,
    )
    print(
        f"[cublaslt-group-gemm-mx] extension loaded, cublasLt version "
        f"{_ext.cublaslt_version()}",
        file=sys.stderr,
        flush=True,
    )
    return _ext


class _Plan:
    __slots__ = ("pid", "E", "N", "K", "sc_cols", "w_ptrs", "wsc_ptrs", "x_ptrs",
                 "o_ptrs", "xsc_ptrs", "marr", "karr", "nwarr")


def _round_up(x, m):
    return (x + m - 1) // m * m


def _get_plan(ext, w_fp8, w_blockscale, sum_m, device):
    E, N, K = w_fp8.shape
    key = (w_fp8.data_ptr(), E, N, K)
    plan = _plans.get(key)
    if plan is not None:
        return plan
    assert w_fp8.is_contiguous() and w_blockscale.is_contiguous()
    sc_cols = _round_up(K // 32, 4)
    per_w_sc = _round_up(N, 128) * sc_cols
    assert w_blockscale.numel() == E * per_w_sc, (w_blockscale.shape, E, per_w_sc)

    global _workspace
    if _workspace is None:
        _workspace = torch.empty(_WORKSPACE_BYTES, dtype=torch.uint8, device=device)

    p = _Plan()
    p.E, p.N, p.K, p.sc_cols = E, N, K, sc_cols
    idx = torch.arange(E, dtype=torch.int64)
    p.w_ptrs = (w_fp8.data_ptr() + idx * (N * K)).to(device)
    p.wsc_ptrs = (w_blockscale.data_ptr() + idx * per_w_sc).to(device)
    p.x_ptrs = torch.empty(E, dtype=torch.int64, device=device)
    p.o_ptrs = torch.empty(E, dtype=torch.int64, device=device)
    p.xsc_ptrs = torch.empty(E, dtype=torch.int64, device=device)
    p.marr = torch.empty(E, dtype=torch.int32, device=device)
    p.karr = torch.full((E,), K, dtype=torch.int32, device=device)
    p.nwarr = torch.full((E,), N, dtype=torch.int32, device=device)
    p.pid = ext.create_plan(
        p.w_ptrs, p.x_ptrs, p.o_ptrs, p.wsc_ptrs, p.xsc_ptrs,
        p.karr, p.nwarr, p.marr,
        N, K, max(1, sum_m // E), 1.0, _workspace,
    )
    _plans[key] = plan = p
    print(
        f"[cublaslt-group-gemm-mx] plan {p.pid} created: E={E} N={N} K={K} "
        f"algos={ext.num_algos(p.pid)}",
        file=sys.stderr,
        flush=True,
    )
    return plan


def cublaslt_mxfp8_group_mm(a_fp8, b_fp8, a_blockscale, b_blockscale,
                            out_dtype, device, params):
    """a_fp8 [sum_m, K] e4m3 grouped by expert; b_fp8 [E, N, K] e4m3;
    a_blockscale: flat/2D e8m0, concatenated per-expert 128-row-padded swizzled
    segments; b_blockscale [E, round_up(N,128)*round_up(K/32,4)] e8m0 swizzled.
    params: expert_offsets [E] (row offsets), blockscale_offsets [E] (128-padded
    scale-row offsets), problem_sizes [E, 3] with column 0 = m_e."""
    assert out_dtype == torch.bfloat16, "cublasLt mxfp8 grouped path emits bf16"
    ext = load_ext()
    plan = _get_plan(ext, b_fp8, b_blockscale, a_fp8.shape[0], device)
    E, N, K = plan.E, plan.N, plan.K

    out = torch.empty((a_fp8.shape[0], N), device=device, dtype=out_dtype)
    eo = params["expert_offsets"].to(torch.int64)      # [E] rows into a_fp8/out
    bo = params["blockscale_offsets"].to(torch.int64)  # [E] 128-padded scale rows
    plan.x_ptrs.copy_(a_fp8.data_ptr() + eo * K)             # e4m3 rows of K B
    plan.o_ptrs.copy_(out.data_ptr() + eo * (N * 2))         # bf16 rows of N*2 B
    plan.xsc_ptrs.copy_(a_blockscale.data_ptr() + bo * plan.sc_cols)  # e8m0
    plan.marr.copy_(params["problem_sizes"][:, 0])           # per-expert m
    ext.run_plan(plan.pid)
    return out
