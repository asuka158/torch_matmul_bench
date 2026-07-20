"""Sweep ALL dense benchmark points and census per-call device kernels for BOTH
backends (see profile_dense_e2e.py for the single-point version).

Purpose: find every point where a python call launches MORE than the main GEMM
kernel (e.g. cublasLt split-K adds a splitKreduce_kernel that the benchmark's
'nvjet' substring did NOT count), and quantify the e2e-vs-main-kernel gap.

Run: /opt/uv/bin/python profile_dense_sweep.py            (nvfp4 categories)
"""
import sys

sys.path.insert(0, 'benchmark')

import torch  # noqa: E402

from bench_lib import M_ALL, build_dense  # noqa: E402
from bench_dense import CATEGORIES  # noqa: E402

ITERS = 5


def census(run):
    run()
    torch.cuda.synchronize()
    with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(ITERS):
            run()
        torch.cuda.synchronize()
    rows = [(ka.key, ka.count / ITERS, ka.self_device_time_total / ITERS)
            for ka in prof.key_averages()
            if ka.device_type == torch.profiler.DeviceType.CUDA]
    total = sum(us for _, _, us in rows)
    gemm = sum(us for name, _, us in rows
               if 'nvjet' in name or 'device_kernel' in name)
    return sum(c for _, c, _ in rows), total, gemm, rows


multi = []
for cat, variants in CATEGORIES.items():
    for variant, n, k in variants:
        for m in M_ALL:
            run_ct, run_lt, _ = build_dense(m, n, k)
            for backend, run in (('CT', run_ct), ('LT', run_lt)):
                nk, total, gemm, rows = census(run)
                if round(nk) != 1:
                    extra = [r[0][:60] for r in rows
                             if 'nvjet' not in r[0] and 'device_kernel' not in r[0]]
                    multi.append((cat, variant, m, n, k, backend, nk, total, gemm, extra))
                    print(f'MULTI {cat}/{variant} m={m} {backend}: kernels={nk:.0f} '
                          f'gemm={gemm:.2f}us total={total:.2f}us '
                          f'share={100 * gemm / total:.1f}% extra={extra}', flush=True)
            del run_ct, run_lt
            torch.cuda.empty_cache()
        print(f'. {cat}/{variant} done', flush=True)

print(f'\n{len(multi)} multi-kernel points total')
