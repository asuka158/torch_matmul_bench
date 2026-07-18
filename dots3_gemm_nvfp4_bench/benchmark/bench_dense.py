"""dots3 dense nvfp4*nvfp4=bf16 GEMM benchmark, CUTLASS (sgl_kernel) vs cuBLASLt (torch).

One CSV per shape category under ../result/, both backends interleaved.
Run: /opt/uv/bin/python bench_dense.py [category ...]      (default: all)

NOTE: fused_qkv_a_g_proj_with_mqa 'swa' was requested as [2176, 5210]; K=5210 is not a
multiple of 16 and cannot be quantized to NVFP4 (16-element block scales along K), so it
is benchmarked as [2176, 5120] (assumed typo for the 5120 hidden size).
"""
import sys

import torch

from bench_lib import (
    M_ALL, RESULT_DIR, SUBSTR, NvmlSampler, build_dense, csv_writer, measure,
)

CATEGORIES = {
    'fused_qkv_a_g_proj_with_mqa': [('nsa', 1920, 5120), ('swa', 2176, 5120)],
    'fused_q_b_wq_b_proj':         [('nsa', 32768, 1024), ('swa', 16384, 1024)],
    'kv_b_proj':                   [('nsa', 32768, 512), ('swa', 20480, 1024)],
    'o_proj':                      [('nsa', 5120, 16384), ('swa', 5120, 8192)],
    'gate_up_proj_dense':          [('tp1', 27648, 5120), ('tp4', 6912, 5120), ('tp8', 3456, 5120)],
    'down_proj_dense':             [('tp1', 5120, 13824), ('tp4', 5120, 3456), ('tp8', 5120, 1728)],
}

COLUMNS = ['variant', 'm', 'n', 'k', 'us', 'tflops', 'gbps', 'sm_mhz', 'power_w', 'backend']


def bench_category(cat, sampler):
    emit, f = csv_writer(f'{RESULT_DIR}/{cat}.csv', COLUMNS)
    print(f'=== {cat} ===', flush=True)
    for variant, n, k in CATEGORIES[cat]:
        for m in M_ALL:
            try:
                run_ct, run_lt, nbytes = build_dense(m, n, k)
                flops = 2.0 * m * n * k
                for backend, run in (('CUTLASS', run_ct), ('CUBLASLT', run_lt)):
                    us, tflops, gbps, sm, pw = measure(
                        run, SUBSTR[('dense', backend)], sampler, flops, nbytes)
                    emit([variant, m, n, k, f'{us:.3f}', f'{tflops:.2f}', f'{gbps:.2f}',
                          f'{sm:.0f}', f'{pw:.1f}', backend])
                    print(f'{variant:4} m={m:6} n={n:6} k={k:6} | {us:9.2f} us '
                          f'{tflops:8.1f} TF {gbps:8.1f} GB/s | {backend}', flush=True)
                del run_ct, run_lt
                torch.cuda.empty_cache()
            except Exception as ex:
                print(f'{variant:4} m={m:6} n={n:6} k={k:6} | ERROR '
                      f'{type(ex).__name__}: {str(ex)[:120]}', flush=True)
                torch.cuda.empty_cache()
    f.close()
    print(f'-> {RESULT_DIR}/{cat}.csv\n', flush=True)


if __name__ == '__main__':
    cats = sys.argv[1:] or list(CATEGORIES)
    for c in cats:
        assert c in CATEGORIES, f'unknown category {c}; choose from {list(CATEGORIES)}'
    print(f'Device: {torch.cuda.get_device_name(0)}')
    sampler = NvmlSampler(index=0)
    try:
        for c in cats:
            bench_category(c, sampler)
    finally:
        sampler.close()
