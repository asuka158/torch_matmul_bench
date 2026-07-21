"""dots3 absorbed-bmm nvfp4*nvfp4=bf16 benchmark, CUTLASS vs cuBLASLt.

One CSV per bmm under ../result/, both backends interleaved.
Run: /opt/venvs/sglang-dev/bin/python bench_bmm.py [bmm ...]      (default: all four)

Both backends consume the SAME quantized buffers and apply the same per-TENSOR global
scale inside the kernel, so the comparison is like-for-like. Every call is a single
device kernel for both backends (profile_bmm_sweep.py: 0/112 multi-kernel points), so
the main-kernel timing convention here IS end-to-end device time.
"""
import csv
import os
import sys

import torch

from bmm_lib import (
    BMMS, M_ALL, RESULT_DIR, SUBSTR, NvmlSampler, build_bmm, csv_writer, measure,
)
import cublaslt_batched_nvfp4 as cb   # after bmm_lib, which puts PARENT on sys.path

LAYER_CLASS = {'A5_w_kc': 'full', 'A7_w_vc': 'full',
               'C5_w_kc': 'swa_mtp', 'C7_w_vc': 'swa_mtp'}

COLUMNS = ['bmm', 'layer_class', 'batch', 'm', 'k', 'n',
           'us', 'tflops', 'gbps', 'sm_mhz', 'power_w', 'relerr', 'backend']


def _complete(path):
    """{m} that already have BOTH backend rows in an existing CSV."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return set()
    with open(path) as fh:
        rows = [r for r in csv.reader(fh)][1:]
    have = {}
    for r in rows:
        if len(r) < len(COLUMNS):
            continue
        have.setdefault(int(r[3]), set()).add(r[-1])
    return {k for k, b in have.items() if b >= {'CUTLASS', 'CUBLASLT'}}


def _sort_csv(path):
    """(m, backend) order, so topped-up rows do not pile up at the end."""
    with open(path) as fh:
        rows = [r for r in csv.reader(fh) if r]
    head, body = rows[0], rows[1:]
    body.sort(key=lambda r: (int(r[3]), r[-1]))
    with open(path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(head)
        w.writerows(body)


def bench_one(bmm, sampler, incremental=False):
    B, K, N = BMMS[bmm]
    path = f'{RESULT_DIR}/{bmm}.csv'
    done = _complete(path) if incremental else set()
    emit, f = csv_writer(path, COLUMNS, append=incremental)
    print(f'=== {bmm}  (batch={B}, K={K}, N={N}) ===' +
          (f' (incremental, {len(done)} pts already done)' if incremental else ''),
          flush=True)
    for m in M_ALL:
        if m in done:
            continue
        try:
            run_ct, run_lt, flops, nbytes, ref, (out_ct, out_lt) = build_bmm(
                B, m, K, N, want_ref=True)
        except Exception as ex:
            print(f'm={m:6} | BUILD-ERROR {type(ex).__name__}: {str(ex)[:110]}',
                  flush=True)
            torch.cuda.empty_cache()
            continue
        for backend, run, out in (('CUTLASS', run_ct, out_ct),
                                  ('CUBLASLT', run_lt, out_lt)):
            try:
                us, tflops, gbps, sm, pw = measure(
                    run, SUBSTR[backend], sampler, flops, nbytes)
                rel = ((out.float() - ref).norm() / ref.norm()).item()
                flag = '  !! relerr' if rel > 0.3 else ''
                emit([bmm, LAYER_CLASS[bmm], B, m, K, N, f'{us:.3f}', f'{tflops:.2f}',
                      f'{gbps:.2f}', f'{sm:.0f}', f'{pw:.1f}', f'{rel:.4e}', backend])
                print(f'm={m:6} | {us:9.2f} us {tflops:8.1f} TF {gbps:8.1f} GB/s '
                      f'relerr={rel:.3e} | {backend}{flag}', flush=True)
            except Exception as ex:
                print(f'm={m:6} | ERROR {type(ex).__name__}: {str(ex)[:110]} | '
                      f'{backend}', flush=True)
        del run_ct, run_lt, ref, out_ct, out_lt
        cb.destroy_all()      # plans pin their operands C++-side; free before next point
        torch.cuda.empty_cache()
    f.close()
    if incremental:
        _sort_csv(path)
    print(f'-> {path}\n', flush=True)


if __name__ == '__main__':
    args = sys.argv[1:]
    incremental = '--incremental' in args
    todo = [a for a in args if not a.startswith('--')] or list(BMMS)
    for b in todo:
        assert b in BMMS, f'unknown bmm {b}; choose from {list(BMMS)}'
    print(f'Device: {torch.cuda.get_device_name(0)}')
    sampler = NvmlSampler(index=0)
    try:
        for b in todo:
            bench_one(b, sampler, incremental=incremental)
    finally:
        sampler.close()
