"""cuBLASLt strided-batched NVFP4 bmm (nvfp4 x nvfp4 -> bf16), JIT extension.

Self-written because NO installed wrapper exposes a batched (3D) nvfp4 matmul:
torch._scaled_mm_v2 / flashinfer.mm_fp4 / sgl_kernel.cutlass_scaled_fp4_mm are all
2D-only, and deep_gemm has no pure nvfp4 gemm at all. Writing the cuBLAS(Lt) host
API directly is fine -- it is a black box with few tunables, so a hand-written call
is representative of the library. (A hand-written CUTLASS kernel would NOT be
equally representative, since its tile/cluster/schedule config drives performance;
that path is the last resort.)

Semantics per batch b:
    D[b] = alpha * dequant(X[b] : [M,K]) @ dequant(W[b] : [N,K])^T
"""
import glob
import os

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_WORKSPACE_BYTES = 64 << 20

_ext = None
_workspace = None


def _find_cublaslt_libdir():
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
        name="dots3_cublaslt_batched_nvfp4",
        sources=[os.path.join(_HERE, "csrc", "cublaslt_batched_nvfp4.cpp")],
        extra_cflags=["-O3", "-std=c++20"],   # torch 2.13 headers need c++20
        extra_include_paths=[os.path.join(cuda_home, "include")],
        extra_ldflags=[
            f"-L{libdir}", "-lcublasLt", f"-Wl,-rpath,{libdir}", "-lcudart",
            f"-L{os.path.join(cuda_home, 'lib64')}",
        ],
        verbose=verbose,
    )
    return _ext


def workspace(device="cuda"):
    global _workspace
    if _workspace is None:
        _workspace = torch.empty(_WORKSPACE_BYTES, dtype=torch.uint8, device=device)
    return _workspace


def make_plan(w, x, out, w_sc, x_sc, B, M, N, K, alpha,
              alpha_vec=None, scale_mode=0):
    """-> (ext, plan_id). Heuristic runs here; run_plan(id) only enqueues.

    scale_mode: 0 = single host scalar alpha (per-TENSOR global scale)
                1 = per-batch alpha vector [B, N] (ALPHA_VECTOR_BATCH_STRIDE)
                2 = per-batch D scale [B]        (D_SCALE_MODE PER_BATCH_SCALAR_32F)
    """
    ext = load_ext()
    if alpha_vec is None:
        alpha_vec = torch.empty(0, dtype=torch.float32, device=w.device)
    pid = ext.create_plan(w, x, out, w_sc, x_sc, B, M, N, K, float(alpha),
                          alpha_vec, int(scale_mode), workspace(w.device))
    return ext, pid
