"""CUTLASS batched nvfp4 bmm, driven from the official CuTe-DSL example.

Why this module exists: no packaged wrapper exposes a batched (3D) nvfp4 matmul --
flashinfer's mm_fp4 / torch._scaled_mm_v2 / sgl_kernel.cutlass_scaled_fp4_mm are all
2D-only, and flashinfer's cute-dsl path compiles 2D fake tensors so it pins L=1. The
underlying CUTLASS kernel, however, is natively batched (problem shape is (M,N,K,L)),
and it is ONE kernel launch for the whole batch -- verified in probe_cutlass_batched.py
(L=1/8/64 all launch exactly 1 kernel), so this is a true bmm and not a loop of dense
GEMMs.

We drive the official example kernel that flashinfer bundles (nvidia-cutlass-dsl is a
pure-Python DSL -- no C++/nvcc involved) and reuse ITS helpers for the scale-factor
re-ordering and pointer construction, so the layout contract is the kernel author's,
not ours. The only thing we choose is the tile/cluster config (see DEFAULT_TILE).

Global scale: nvfp4's per-tensor fp32 global scale is folded into the kernel's
compile-time `epilogue_op` (`lambda x: x * alpha`), so it is applied INSIDE the kernel
exactly like cuBLASLt applies its alpha -- keeping the two backends comparable. This
works because the global scale is per-TENSOR; the batched dense kernel has no
per-batch alpha (only the masked-grouped MoE kernel does), and neither does cuBLASLt.
"""
import importlib.util
import os
import sys

import torch

_EXAMPLE = ('/opt/venvs/sglang-dev/lib/python3.12/site-packages/flashinfer/data/cutlass/'
            'examples/python/CuTeDSL/blackwell/dense_blockscaled_gemm_persistent.py')

# CUTLASS perf is config-sensitive; (128,128)/(1,1) is the example's own default and is
# valid for every dots3 bmm shape. probe_cutlass_cfg.py measures alternatives.
DEFAULT_TILE = (128, 128)
DEFAULT_CLUSTER = (1, 1)

_mod = None


def example():
    """the bundled official CUTLASS CuTe-DSL blockscaled-gemm example module"""
    global _mod
    if _mod is None:
        spec = importlib.util.spec_from_file_location('cutlass_bs_gemm_example', _EXAMPLE)
        m = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = m
        spec.loader.exec_module(m)
        _mod = m
    return _mod


class CutlassBatchedNvfp4:
    """Compiled batched nvfp4 gemm for one (M, N, K, L, alpha) point.

    D[b] = alpha * dequant(A[b] : [M,K]) @ dequant(B[b] : [N,K])^T   (bf16 out)

    Operands follow the example's layout contract:
      a_fp4/b_fp4 : uint8 CUDA, contiguous [L, M, K//2] / [L, N, K//2] (packed fp4)
      sfa/sfb     : e4m3 CPU,   contiguous [L, M, K//16] / [L, N, K//16] (UN-swizzled;
                    the example's reorder helper produces the CUTLASS layout)
    """

    def __init__(self, M, N, K, L, alpha,
                 mma_tiler_mn=DEFAULT_TILE, cluster_shape_mn=DEFAULT_CLUSTER):
        import cutlass
        import cuda.bindings.driver as cuda_driver
        from cutlass.cute.runtime import make_ptr  # noqa: F401 (used via example)

        ex = example()
        self.ex, self.M, self.N, self.K, self.L = ex, M, N, K, L
        self.mnkl = (M, N, K, L)
        self.sf_vec_size = 16
        self.ab_dtype = cutlass.Float4E2M1FN
        self.sf_dtype = cutlass.Float8E4M3FN
        self.c_dtype = cutlass.BFloat16
        self.a_major, self.b_major, self.c_major = 'k', 'k', 'n'

        gemm = ex.Sm100BlockScaledPersistentDenseGemmKernel(
            self.sf_vec_size, mma_tiler_mn, cluster_shape_mn)
        if not gemm.can_implement(self.mnkl, self.ab_dtype, self.sf_dtype, self.c_dtype,
                                  self.a_major, self.b_major, self.c_major,
                                  self.sf_vec_size, mma_tiler_mn, cluster_shape_mn):
            raise TypeError(f'CUTLASS cannot implement mnkl={self.mnkl} '
                            f'tile={mma_tiler_mn} cluster={cluster_shape_mn}')

        self.stream = cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)
        max_active_clusters = ex.utils.HardwareInfo().get_max_active_clusters(
            cluster_shape_mn[0] * cluster_shape_mn[1])
        # per-TENSOR global scale applied INSIDE the kernel (same as cuBLASLt's alpha).
        # The kernel calls epilogue_op(acc.to(c_dtype)), so x is already bf16 and the
        # multiply promotes to f32 -> cast back or the epilogue store type-mismatches.
        a, c_dt = float(alpha), self.c_dtype
        self.compiled = ex.scaled_mm(
            gemm, self.ab_dtype, self.c_dtype, self.sf_dtype,
            self.a_major, self.b_major, self.c_major,
            max_active_clusters, self.stream,
            epilogue_op=lambda x: (x * a).to(c_dt),
            options='--opt-level 2')
        self._ptrs = None

    def bind(self, a_fp4, b_fp4, sfa, sfb, out):
        """a_fp4 [L,M,K//2] uint8 cuda, sfa [L,M,K//16] e4m3 CPU, out [L,M,N] bf16 cuda"""
        ex, (M, N, K, L) = self.ex, self.mnkl
        # example layout: storage [L, X, ...] viewed as [X, ..., L]
        a = a_fp4.view(torch.int8).permute(1, 2, 0).view(torch.float4_e2m1fn_x2)
        b = b_fp4.view(torch.int8).permute(1, 2, 0).view(torch.float4_e2m1fn_x2)
        c = out.permute(1, 2, 0)
        sfa_r = ex.create_and_reorder_scale_factor_tensor(
            L, M, K, self.sf_vec_size, self.sf_dtype, sfa.permute(1, 2, 0))
        sfb_r = ex.create_and_reorder_scale_factor_tensor(
            L, N, K, self.sf_vec_size, self.sf_dtype, sfb.permute(1, 2, 0))
        self._keep = (a, b, c, sfa_r, sfb_r, out)   # keep alive
        self._ptrs = ex.construct_cute_pointers(
            a, b, sfa_r, sfb_r, c, self.ab_dtype, self.sf_dtype, self.c_dtype)
        return self

    def __call__(self):
        a_ptr, b_ptr, c_ptr, sfa_ptr, sfb_ptr = self._ptrs
        self.compiled(a_ptr, b_ptr, sfa_ptr, sfb_ptr, c_ptr, self.mnkl, self.stream)
