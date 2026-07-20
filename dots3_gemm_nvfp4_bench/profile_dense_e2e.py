"""Per-call device-kernel census for the two nvfp4 DENSE entries.

Question: from the python call to its return, how many device kernels does ONE call
launch, and what share of the call's total device time is the main GEMM kernel?
(The benchmark times ONLY the main kernel; a framework pays the whole call.)

Method: build the point's buffers once (untimed, outside), warm up, then profile
`iters` bare calls with torch.profiler (CUDA activity). Aggregate kineto device
events by kernel name; counts/times divided by `iters` = per-call. Memsets/memcpys
are listed too (allocation itself launches no kernel; buffers are all pre-created
except each call's own output/workspace, which the caching allocator serves).

Run: /opt/uv/bin/python profile_dense_e2e.py
"""
import sys

sys.path.insert(0, 'benchmark')

import torch  # noqa: E402

from bench_lib import build_dense  # noqa: E402

ITERS = 20
SHAPES = [  # (tag, n, k), m values probed per shape
    ('gate_up_tp4', 6912, 5120),
    ('o_proj_nsa', 5120, 16384),
    ('fused_qkv_nsa', 1920, 5120),
]
MS = [16, 4096]


def census(run, iters=ITERS):
    run()
    torch.cuda.synchronize()
    with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            run()
        torch.cuda.synchronize()
    rows = []
    for ka in prof.key_averages():
        if ka.device_type != torch.profiler.DeviceType.CUDA:
            continue
        rows.append((ka.key, ka.count / iters, ka.self_device_time_total / iters))
    rows.sort(key=lambda r: -r[2])
    return rows


def is_gemm(name):
    return 'nvjet' in name or 'device_kernel' in name


for tag, n, k in SHAPES:
    for m in MS:
        run_ct, run_lt, _ = build_dense(m, n, k)
        for backend, run in (('CUTLASS(sgl_kernel)', run_ct), ('CUBLASLT(torch)', run_lt)):
            rows = census(run)
            total = sum(us for _, _, us in rows)
            gemm = sum(us for name, _, us in rows if is_gemm(name))
            nk = sum(c for _, c, _ in rows)
            print(f'{tag:14} m={m:5} {backend:20} | kernels/call={nk:.0f} '
                  f'device_us/call={total:8.2f} gemm_us={gemm:8.2f} '
                  f'gemm_share={100 * gemm / total:6.2f}%')
            for name, c, us in rows:
                mark = ' <-- GEMM' if is_gemm(name) else ''
                print(f'    x{c:4.1f} {us:9.3f} us  {name[:110]}{mark}')
        del run_ct, run_lt
        torch.cuda.empty_cache()
        print()
