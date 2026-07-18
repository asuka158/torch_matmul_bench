"""JIT loader for the CUTLASS dense mxfp8 GEMM (see csrc/cutlass_dense_mx.cu).

Compiled against the cutlass headers vendored in ../third_party/cutlass
(v4.5.2) -- the bench does not depend on any external checkout. First build
takes a few minutes (sm100a cutlass kernels), cached in ~/.cache/torch_extensions.

Exports:
    mxfp8_mm_2sm(a, b, a_sf, b_sf, alpha) -> bf16   256x256x128 / 2SM / cluster(4,4)
    mxfp8_mm_1sm(a, b, a_sf, b_sf, alpha) -> bf16   128x128x128 / 1SM / cluster(1,1)
a/b: e4m3 row-major [M,K]/[N,K]; a_sf/b_sf: e8m0 swizzled 32x4x4 (to_blocked);
alpha: fp32 scalar tensor (pass 1.0 -- mxfp8 has no per-tensor global scale).
"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_CUTLASS = os.path.join(os.path.dirname(_HERE), "third_party", "cutlass")

_ext = None


def load_ext(verbose=False):
    global _ext
    if _ext is not None:
        return _ext
    from torch.utils.cpp_extension import load

    os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
    _ext = load(
        name="dots3_cutlass_dense_mx",
        sources=[os.path.join(_HERE, "csrc", "cutlass_dense_mx.cu")],
        extra_cflags=["-O3", "-std=c++20"],
        extra_cuda_cflags=[
            "-O3",
            "-std=c++20",
            "-gencode=arch=compute_100a,code=sm_100a",
            "--expt-relaxed-constexpr",
            "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1",
        ],
        extra_include_paths=[
            os.path.join(_CUTLASS, "include"),
            os.path.join(_CUTLASS, "tools", "util", "include"),
        ],
        verbose=verbose,
    )
    return _ext


def mxfp8_mm_2sm(a, b, a_sf, b_sf, alpha):
    return load_ext().mm_2sm(a, b, a_sf, b_sf, alpha)


def mxfp8_mm_1sm(a, b, a_sf, b_sf, alpha):
    return load_ext().mm_1sm(a, b, a_sf, b_sf, alpha)
