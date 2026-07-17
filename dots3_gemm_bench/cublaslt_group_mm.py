"""Python entry for the cuBLASLt NVFP4 grouped GEMM, for use in dots3_gemm_bench.

Reuses the JIT extension living in the sglang tree
(srt/layers/moe/cublaslt_group_gemm: host-API .cpp, torch cpp_extension JIT,
~30 s first build per node, cached in ~/.cache/torch_extensions) by loading that
package directly from its file path — no `import sglang`, no copy of the csrc.

Exposes:
    cublaslt_fp4_group_mm(a_fp4, b_fp4, a_blockscale, b_blockscale, alphas,
                          out_dtype, device, params)
same signature/semantics as sgl_kernel.cutlass_fp4_group_mm, with the cublasLt
grouped-matmul restriction: all per-expert alphas must be EQUAL (single scalar
alpha per call) -> quantize weights with one per-tensor global scale shared by
all experts.
"""
import importlib.util
import os

_SGLANG_EXT_DIR = (
    "/mnt/3fs/dots-pretrain/daijiangkun/sglang/python/sglang/srt/layers/moe/"
    "cublaslt_group_gemm"
)


def _load_pkg():
    os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
    spec = importlib.util.spec_from_file_location(
        "cublaslt_group_gemm_standalone",
        os.path.join(_SGLANG_EXT_DIR, "__init__.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_pkg = _load_pkg()
cublaslt_fp4_group_mm = _pkg.cublaslt_fp4_group_mm
load_ext = _pkg.load_ext
