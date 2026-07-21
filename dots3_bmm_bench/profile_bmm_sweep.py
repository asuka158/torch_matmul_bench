"""Kernel census over ALL bmm benchmark points, both backends.

Purpose (same as the GEMM benches' profile_dense_sweep.py): find every point where one
python call launches MORE than the main BMM kernel, so we know whether the
"main kernel only" timing convention equals true end-to-end device time. The GEMM bench
found cuBLASLt silently adding a splitKreduce_kernel that the 'nvjet' substring missed;
this checks for the same class of gap here.

Also prints the matched kernel names so SUBSTR in benchmark/bmm_lib.py is grounded.

Run: /opt/venvs/sglang-dev/bin/python profile_bmm_sweep.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'benchmark'))

import torch  # noqa: E402

from bmm_lib import BMMS, M_ALL, SUBSTR, build_bmm  # noqa: E402

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
            if ka.device_type == torch.profiler.DeviceType.CUDA and ka.count]
    total = sum(us for _, _, us in rows)
    return sum(c for _, c, _ in rows), total, rows


if __name__ == '__main__':
    print(f'device: {torch.cuda.get_device_name(0)}\n')
    names = {}
    multi = []
    for bmm, (B, K, N) in BMMS.items():
        for m in M_ALL:
            try:
                run_ct, run_lt, _, _ = build_bmm(B, m, K, N)
            except Exception as ex:
                print(f'{bmm} m={m}: BUILD-FAIL {type(ex).__name__}: {str(ex)[:90]}',
                      flush=True)
                continue
            for backend, run in (('CUTLASS', run_ct), ('CUBLASLT', run_lt)):
                nk, total, rows = census(run)
                main = sum(us for n, _, us in rows if SUBSTR[backend] in n)
                names.setdefault(backend, set()).update(
                    n[:70] for n, _, _ in rows if SUBSTR[backend] in n)
                if round(nk) != 1 or main == 0:
                    extra = [r[0][:60] for r in rows if SUBSTR[backend] not in r[0]]
                    share = 100 * main / total if total else float('nan')
                    multi.append((bmm, m, backend, nk, share, extra))
                    print(f'MULTI {bmm} m={m} {backend}: kernels={nk:.0f} '
                          f'main={main:.2f}us total={total:.2f}us share={share:.1f}% '
                          f'extra={extra}', flush=True)
            del run_ct, run_lt
            torch.cuda.empty_cache()
        print(f'. {bmm} done', flush=True)

    print('\n=== matched main-kernel names ===')
    for backend, ns in names.items():
        for n in sorted(ns):
            print(f'  {backend}: {n}')
    print(f'\n{len(multi)} multi-kernel points out of '
          f'{len(BMMS) * len(M_ALL) * 2} total')
