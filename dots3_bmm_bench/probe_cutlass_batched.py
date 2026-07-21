"""Does CUTLASS really do BATCHED nvfp4, and is it one kernel or a loop of dense GEMMs?

Drives the official CUTLASS CuTe-DSL example that flashinfer bundles
(dense_blockscaled_gemm_persistent.py, defaults ab=Float4E2M1FN sf=Float8E4M3FN
sf_vec_size=16 -> exactly nvfp4). Its `run_scaled_mm(mnkl=(M,N,K,L), ...)` takes an
explicit L (batch) mode and validates against its own reference.

Three questions:
  1. does batched nvfp4 (L>1) actually run and pass the reference check?
  2. is it ONE batched kernel launch, or L separate dense-GEMM launches?
     -> profile and count GEMM kernel launches per call; true bmm => 1, loop => L
  3. does the kernel expose per-batch alpha (alpha_tensor of shape (l,))?

Run: /opt/venvs/sglang-dev/bin/python probe_cutlass_batched.py
"""
import importlib.util
import inspect
import os
import sys

import torch

EX = ('/opt/venvs/sglang-dev/lib/python3.12/site-packages/flashinfer/data/cutlass/'
      'examples/python/CuTeDSL/blackwell/dense_blockscaled_gemm_persistent.py')


def load_example():
    spec = importlib.util.spec_from_file_location('cutlass_bs_gemm_example', EX)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def gemm_kernels(prof):
    """(total launches, [(name, count)]) for GPU kernels that look like the gemm."""
    rows = [(ka.key, ka.count) for ka in prof.key_averages()
            if ka.device_type == torch.profiler.DeviceType.CUDA and ka.count]
    # the CuTe DSL kernel shows up with a cutlass/cute-ish or generated name; exclude
    # obvious torch elementwise/copy helpers used by tensor setup
    skip = ('memcpy', 'memset', 'elementwise', 'vectorized_', 'fill_', 'copy_')
    ker = [(n, c) for (n, c) in rows if not any(s in n.lower() for s in skip)]
    return sum(c for _, c in ker), ker


if __name__ == '__main__':
    import cutlass  # noqa: F401  (nvidia-cutlass-dsl, pure python)

    print(f'device: {torch.cuda.get_device_name(0)}')
    print(f'cutlass-dsl: {cutlass.__version__ if hasattr(cutlass, "__version__") else "?"}')
    mod = load_example()
    run = mod.run_scaled_mm
    print(f'run_scaled_mm params: {list(inspect.signature(run).parameters)[:10]}')

    # per-batch alpha support?
    kcls = mod.Sm100BlockScaledPersistentDenseGemmKernel
    doc = (inspect.getdoc(kcls.__call__) or '')
    has_alpha = 'alpha_tensor' in inspect.signature(kcls.__call__).parameters
    print(f'kernel __call__ has alpha_tensor param: {has_alpha}')
    if has_alpha:
        line = [l.strip() for l in doc.splitlines() if 'alpha_tensor' in l and 'shape' in l]
        print(f'  doc: {line[0] if line else "(see source)"}')

    common = dict(ab_dtype=cutlass.Float4E2M1FN, sf_dtype=cutlass.Float8E4M3FN,
                  sf_vec_size=16, c_dtype=cutlass.BFloat16,
                  a_major='k', b_major='k', c_major='n',
                  mma_tiler_mn=(128, 128), cluster_shape_mn=(1, 1),
                  tolerance=1e-01, warmup_iterations=0, iterations=1,
                  use_cold_l2=False)

    # ---- Q1: does batched nvfp4 run + pass the example's own reference check? ----
    print('\n=== Q1: correctness (example validates internally) ===')
    for (M, N, K, L) in [(128, 512, 128, 1), (128, 512, 128, 8), (128, 512, 128, 128),
                         (128, 1024, 192, 64)]:
        try:
            run(mnkl=(M, N, K, L), skip_ref_check=False, **common)
            print(f'M={M} N={N} K={K} L={L:4} | OK (ref check passed)', flush=True)
        except Exception as e:
            print(f'M={M} N={N} K={K} L={L:4} | FAIL {type(e).__name__}: '
                  f'{str(e)[:200]}', flush=True)

    # ---- Q2: one batched kernel, or L dense-GEMM launches? ----
    print('\n=== Q2: kernel launches per call (true bmm => 1, loop => L) ===')
    for L in (1, 8, 64):
        try:
            run(mnkl=(128, 512, 128, L), skip_ref_check=True, **common)  # warm/compile
            torch.cuda.synchronize()
            with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
                run(mnkl=(128, 512, 128, L), skip_ref_check=True, **common)
                torch.cuda.synchronize()
            total, ker = gemm_kernels(prof)
            top = sorted(ker, key=lambda t: -t[1])[:3]
            print(f'L={L:4} | kernel launches={total} | '
                  + '; '.join(f'{n[:52]} x{c}' for n, c in top), flush=True)
        except Exception as e:
            print(f'L={L:4} | FAIL {type(e).__name__}: {str(e)[:160]}', flush=True)
