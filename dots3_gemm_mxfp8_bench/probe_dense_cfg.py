"""One-time dense CUTLASS config selection: 2SM 256x256x128/cluster(4,4) vs
1SM 128x128x128/cluster(1,1), measured with the standard benchmark metric
(bench_kineto num_tests=10 flush_l2, main kernel only) across a representative
shape sweep. The winner is FIXED for the whole benchmark (no per-shape selection,
mirroring the "one deterministic kernel config per call" convention).

Run: /opt/uv/bin/python probe_dense_cfg.py
"""
import sys

sys.path.insert(0, 'benchmark')

import cutlass_dense_mx as cdm  # noqa: E402
from bench_lib import M_ALL, SUBSTR, measure, build_dense  # noqa: E402
import bench_lib  # noqa: E402

SHAPES = [  # one per dense category (nsa / tp4 variants)
    ('fused_qkv_a', 1920, 5120),
    ('fused_q_b', 32768, 1024),
    ('kv_b', 32768, 512),
    ('o_proj', 5120, 16384),
    ('gate_up_d', 6912, 5120),
    ('down_d', 5120, 3456),
]
MS = [1, 16, 256, 4096, 16384]

cdm.load_ext()
wins = {'2sm': 0, '1sm': 0}
print(f'{"shape":12} {"m":>6} | {"2sm us":>10} {"1sm us":>10} | winner  ratio(1sm/2sm)')
for tag, n, k in SHAPES:
    for m in MS:
        res = {}
        for cfg in ('2sm', '1sm'):
            bench_lib.DENSE_CUTLASS_CFG = cfg
            run_ct, _, _ = build_dense(m, n, k)
            us, *_ = measure(run_ct, SUBSTR[('dense', 'CUTLASS')], None, 1, 1)
            res[cfg] = us
        w = min(res, key=res.get)
        wins[w] += 1
        print(f'{tag:12} {m:6} | {res["2sm"]:10.2f} {res["1sm"]:10.2f} | {w}  '
              f'{res["1sm"] / res["2sm"]:.3f}', flush=True)
print(f'\nwins: {wins}')
