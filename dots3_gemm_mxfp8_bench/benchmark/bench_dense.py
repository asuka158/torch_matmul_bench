"""dots3 dense mxfp8*mxfp8=bf16 GEMM benchmark, CUTLASS (cutlass_dense_mx JIT) vs
cuBLASLt (torch._scaled_mm_v2).

One CSV per shape category under ../result/, both backends interleaved.
Run: /opt/uv/bin/python bench_dense.py [category ...]      (default: all)

Shapes identical to dots3_gemm_nvfp4_bench (incl. the swa 5210->5120 typo fix).
"""
import csv
import os
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


def _complete(path):
    """{(variant, m)} that already have BOTH backend rows in an existing CSV."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return set()
    with open(path) as fh:
        rows = [r for r in csv.reader(fh)][1:]
    have = {}
    for r in rows:
        if len(r) < len(COLUMNS):
            continue
        have.setdefault((r[0], int(r[1])), set()).add(r[-1])
    return {k for k, b in have.items() if b >= {'CUTLASS', 'CUBLASLT'}}


def _sort_csv(path):
    """(variant, m, backend) order, so topped-up rows do not pile up at the end."""
    with open(path) as fh:
        rows = [r for r in csv.reader(fh) if r]
    head, body = rows[0], rows[1:]
    body.sort(key=lambda r: (r[0], int(r[1]), r[-1]))
    with open(path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(head)
        w.writerows(body)


def bench_category(cat, sampler, incremental=False):
    path = f'{RESULT_DIR}/{cat}.csv'
    done = _complete(path) if incremental else set()
    emit, f = csv_writer(path, COLUMNS, append=incremental)
    print(f'=== {cat} ===' + (f' (incremental, {len(done)} pts already done)'
                              if incremental else ''), flush=True)
    for variant, n, k in CATEGORIES[cat]:
        for m in M_ALL:
            if (variant, m) in done:
                continue
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
    if incremental:
        _sort_csv(path)
    print(f'-> {path}\n', flush=True)


if __name__ == '__main__':
    args = sys.argv[1:]
    incremental = '--incremental' in args
    cats = [a for a in args if not a.startswith('--')] or list(CATEGORIES)
    for c in cats:
        assert c in CATEGORIES, f'unknown category {c}; choose from {list(CATEGORIES)}'
    print(f'Device: {torch.cuda.get_device_name(0)}')
    sampler = NvmlSampler(index=0)
    try:
        for c in cats:
            bench_category(c, sampler, incremental=incremental)
    finally:
        sampler.close()
