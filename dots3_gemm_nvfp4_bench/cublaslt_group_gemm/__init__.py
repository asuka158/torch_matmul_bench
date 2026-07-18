"""cublasLt grouped-matmul backend for the NVFP4 MoE group gemms (experimental).

Drop-in alternative to sgl_kernel.cutlass_fp4_group_mm with the same call
signature, selected per gemm via env vars (read in cutlass_moe.py):

  SGLANG_NVFP4_GEMM1_BACKEND=cutlass|cublaslt   (gate_up, default cutlass)
  SGLANG_NVFP4_GEMM2_BACKEND=cutlass|cublaslt   (down,    default cutlass)

The extension is a single host-API .cpp JIT-compiled with torch.utils.cpp_extension
(~30 s once per node, cached in ~/.cache/torch_extensions; concurrent TP ranks are
serialized by torch's FileBaton). Needs CUDA_HOME (same as the fp4 JIT kernels).

How it works (design + verification in backend_test/README.md):
  * One plan per (layer, gemm): created lazily on the layer's FIRST forward
    (warmup, before CUDA graph capture). The cublasLt heuristic runs once there.
    Weight/weight-scale pointer arrays are filled at creation (weights are static).
  * Every call rewrites only the CONTENTS of the device-side x/out/x-scale pointer
    arrays and marr (per-expert m) from expert_offsets/problem_sizes — no host
    sync, no replan; verified exec-time-read, so run_plan is CUDA-graph safe.
  * alpha must be a SINGLE scalar per call (cublasLt grouped limitation) -> the
    per-expert alphas must all be equal, i.e. a per-layer/per-tensor-global
    checkpoint (bf16_cast_nvfp4_moe.py --global_scope per_layer) is required.
  * Outputs are bitwise-identical to the cutlass kernel on the same quantized
    buffers, so switching backends must not change generated tokens.
"""

import glob
import logging
import os
import sys

import torch

logger = logging.getLogger(__name__)

_HERE = os.path.dirname(os.path.abspath(__file__))
_WORKSPACE_BYTES = 64 << 20

_ext = None
_plans = {}  # (weight data_ptr, E, N, K) -> _Plan
_workspace = None

# debug: after every eager cublasLt call, also run the cutlass kernel on the same
# buffers and compare bitwise (skipped during graph capture; syncs — diagnosis only)
_CROSSCHECK = os.environ.get("SGLANG_CUBLASLT_CROSSCHECK", "0") == "1"
_cc_stats = {"n": 0, "bad": 0}


def _find_cublaslt_libdir():
    # Prefer the cublasLt torch already loaded (nvidia pip wheel), fall back to toolkit.
    candidates = []
    torch_dir = os.path.dirname(torch.__file__)
    site = os.path.dirname(torch_dir)
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
        name="sglang_cublaslt_group_gemm",
        sources=[os.path.join(_HERE, "csrc", "cublaslt_group_gemm.cpp")],
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
    # stderr on purpose: sglang eval scripts often run log_level=error, and this
    # line is the dispatch evidence that the cublasLt backend is actually active
    print(
        f"[cublaslt-group-gemm] extension loaded, cublasLt version "
        f"{_ext.cublaslt_version()}",
        file=sys.stderr,
        flush=True,
    )
    return _ext


class _Plan:
    __slots__ = ("pid", "E", "N", "K", "w_ptrs", "wsc_ptrs", "x_ptrs", "o_ptrs",
                 "xsc_ptrs", "marr", "karr", "nwarr")


def _get_plan(ext, w_fp4, w_blockscale, alphas, sum_m, device):
    E, N, half_k = w_fp4.shape
    K = half_k * 2
    key = (w_fp4.data_ptr(), E, N, K)
    plan = _plans.get(key)
    if plan is not None:
        return plan
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "[cublaslt-group-gemm] plan creation attempted during CUDA graph "
            "capture; the layer must run once eagerly (warmup) first"
        )
    assert w_fp4.is_contiguous() and w_blockscale.is_contiguous()
    assert w_blockscale.shape == (E, N, K // 16), w_blockscale.shape

    global _workspace
    if _workspace is None:
        _workspace = torch.empty(_WORKSPACE_BYTES, dtype=torch.uint8, device=device)

    a0 = alphas.flatten()[0]
    if not bool((alphas == a0).all().item()):
        raise ValueError(
            "[cublaslt-group-gemm] per-expert alphas differ within this layer; "
            "cublasLt grouped matmul supports only ONE scalar alpha per call. "
            "Use a per-layer-global checkpoint "
            "(bf16_cast_nvfp4_moe.py --global_scope per_layer)."
        )
    alpha = float(a0.item())

    p = _Plan()
    p.E, p.N, p.K = E, N, K
    idx = torch.arange(E, dtype=torch.int64)
    p.w_ptrs = (w_fp4.data_ptr() + idx * (N * (K // 2))).to(device)
    p.wsc_ptrs = (w_blockscale.data_ptr() + idx * (N * (K // 16))).to(device)
    p.x_ptrs = torch.empty(E, dtype=torch.int64, device=device)
    p.o_ptrs = torch.empty(E, dtype=torch.int64, device=device)
    p.xsc_ptrs = torch.empty(E, dtype=torch.int64, device=device)
    p.marr = torch.empty(E, dtype=torch.int32, device=device)
    p.karr = torch.full((E,), K, dtype=torch.int32, device=device)
    p.nwarr = torch.full((E,), N, dtype=torch.int32, device=device)
    p.pid = ext.create_plan(
        p.w_ptrs, p.x_ptrs, p.o_ptrs, p.wsc_ptrs, p.xsc_ptrs,
        p.karr, p.nwarr, p.marr,
        N, K, max(1, sum_m // E), alpha, _workspace,
    )
    _plans[key] = plan = p
    print(  # stderr: dispatch evidence, must survive log_level=error (see load_ext)
        f"[cublaslt-group-gemm] plan {p.pid} created: E={E} N={N} K={K} "
        f"alpha={alpha:.6g} algos={ext.num_algos(p.pid)}",
        file=sys.stderr,
        flush=True,
    )
    return plan


def cublaslt_fp4_group_mm(
    a_fp4,
    b_fp4,
    a_blockscale,
    b_blockscale,
    alphas,
    out_dtype,
    device,
    params,
):
    """Same signature/semantics as sgl_kernel.cutlass_fp4_group_mm."""
    assert out_dtype == torch.bfloat16, "cublasLt nvfp4 grouped path emits bf16"
    ext = load_ext()
    plan = _get_plan(ext, b_fp4, b_blockscale, alphas, a_fp4.shape[0], device)
    E, N, K = plan.E, plan.N, plan.K

    out = torch.empty((a_fp4.shape[0], N), device=device, dtype=out_dtype)
    # device-side per-step updates (graph-capturable); array ADDRESSES are baked
    # into the plan, contents are read by the kernel at execution time
    eo = params["expert_offsets"].to(torch.int64)  # [E] rows into a_fp4/out
    bo = params["blockscale_offsets"].to(torch.int64)  # [E] 128-padded scale rows
    plan.x_ptrs.copy_(a_fp4.data_ptr() + eo * (K // 2))  # uint8 rows of K/2 B
    plan.o_ptrs.copy_(out.data_ptr() + eo * (N * 2))  # bf16 rows of N*2 B
    plan.xsc_ptrs.copy_(a_blockscale.data_ptr() + bo * (K // 16))  # e4m3 rows
    plan.marr.copy_(params["problem_sizes"][:, 0])  # per-expert m
    ext.run_plan(plan.pid)
    if _CROSSCHECK and not torch.cuda.is_current_stream_capturing():
        _crosscheck(out, a_fp4, b_fp4, a_blockscale, b_blockscale, alphas,
                    out_dtype, device, params, E, N, K)
    return out


def _crosscheck(out, a_fp4, b_fp4, a_blockscale, b_blockscale, alphas,
                out_dtype, device, params, E, N, K):
    from sgl_kernel import cutlass_fp4_group_mm

    ref = cutlass_fp4_group_mm(
        a_fp4, b_fp4, a_blockscale, b_blockscale, alphas, out_dtype, device, params
    )
    _cc_stats["n"] += 1
    if torch.equal(out, ref):
        if _cc_stats["n"] <= 5 or _cc_stats["n"] % 1000 == 0:
            logger.info(
                "[cublaslt-group-gemm] crosscheck OK call %d (N=%d K=%d sumM=%d)",
                _cc_stats["n"], N, K, a_fp4.shape[0],
            )
        return
    _cc_stats["bad"] += 1
    diff = (out.float() - ref.float()).abs()
    bad_rows = (out != ref).any(dim=1).sum().item()
    logger.warning(
        "[cublaslt-group-gemm] CROSSCHECK MISMATCH call %d (N=%d K=%d sumM=%d): "
        "bad_rows=%d/%d max_abs=%.5f",
        _cc_stats["n"], N, K, a_fp4.shape[0], bad_rows, out.shape[0], diff.max().item(),
    )
    if _cc_stats["bad"] == 1:
        path = f"/tmp/cublaslt_mismatch_pid{os.getpid()}.pt"
        torch.save(
            {"a_fp4": a_fp4, "b_fp4": b_fp4, "a_bs": a_blockscale,
             "b_bs": b_blockscale, "alphas": alphas, "out_lt": out, "out_ct": ref,
             "problem_sizes": params["problem_sizes"],
             "expert_offsets": params["expert_offsets"],
             "blockscale_offsets": params["blockscale_offsets"]},
            path,
        )
        logger.warning("[cublaslt-group-gemm] first mismatch dumped to %s", path)
