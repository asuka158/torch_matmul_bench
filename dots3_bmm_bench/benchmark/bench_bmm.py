"""dots3 absorbed-bmm nvfp4*nvfp4=bf16 benchmark, CUTLASS vs cuBLASLt.

One CSV per bmm under ../result/, both backends interleaved.
Run: /opt/venvs/sglang-dev/bin/python bench_bmm.py [bmm ...]      (default: all four)

Both backends consume the SAME quantized buffers and apply the same per-TENSOR global
scale inside the kernel, so the comparison is like-for-like. Every call is a single
device kernel for both backends (profile_bmm_sweep.py: 0/112 multi-kernel points), so
the main-kernel timing convention here IS end-to-end device time.
"""
import sys

import torch

from bmm_lib import (
    BMMS, M_ALL, RESULT_DIR, SUBSTR, NvmlSampler, build_bmm, csv_writer, measure,
)

LAYER_CLASS = {'A5_w_kc': 'full', 'A7_w_vc': 'full',
               'C5_w_kc': 'swa_mtp', 'C7_w_vc': 'swa_mtp'}

COLUMNS = ['bmm', 'layer_class', 'batch', 'm', 'k', 'n',
           'us', 'tflops', 'gbps', 'sm_mhz', 'power_w', 'relerr', 'backend']


def bench_one(bmm, sampler):
    B, K, N = BMMS[bmm]
    emit, f = csv_writer(f'{RESULT_DIR}/{bmm}.csv', COLUMNS)
    print(f'=== {bmm}  (batch={B}, K={K}, N={N}) ===', flush=True)
    for m in M_ALL:
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
        torch.cuda.empty_cache()
    f.close()
    print(f'-> {RESULT_DIR}/{bmm}.csv\n', flush=True)


if __name__ == '__main__':
    todo = sys.argv[1:] or list(BMMS)
    for b in todo:
        assert b in BMMS, f'unknown bmm {b}; choose from {list(BMMS)}'
    print(f'Device: {torch.cuda.get_device_name(0)}')
    sampler = NvmlSampler(index=0)
    try:
        for b in todo:
            bench_one(b, sampler)
    finally:
        sampler.close()
