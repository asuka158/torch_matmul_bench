"""dots3 MoE group nvfp4*nvfp4=bf16 GEMM benchmark, CUTLASS (sgl_kernel) vs cuBLASLt
(JIT ext via ../cublaslt_group_mm.py).

E=257 (256 routed + 1 fused shared expert), topk=9 (8 distinct routed + shared on every
token), sum_m = T*9. Weights are quantized once per (n,k) with ONE global scale shared by
all experts (equal alphas = the cublasLt grouped semantics; cutlass gets the same alphas).
Before timing, both backends run once on the same buffers and the outputs are compared
bitwise (a mismatch prints a warning; rows are still recorded).

Isolation + retry: each (category, variant) runs in its OWN subprocess -- a CUDA fault
(observed sporadically/nondeterministically around the grouped GEMM paths after long
async histories; never reproducible in isolation) then only kills that attempt. The
runner prunes incomplete points from the CSV and re-spawns workers for the missing T
values, up to MAX_ATTEMPTS rounds, so sporadic faults cost one retry instead of a sweep.

One CSV per category under ../result/. Run:
  /opt/uv/bin/python bench_group.py [category ...]      (default: all)
Worker mode (internal): bench_group.py --worker <category> <variant> [T1,T2,...]
"""
import csv as csv_mod
import os
import subprocess
import sys

MAX_ATTEMPTS = int(os.environ.get('BENCH_GROUP_MAX_ATTEMPTS', 10))

CATEGORIES = {
    'gate_up_proj_group': [('tp1', 3072, 5120), ('tp4', 768, 5120), ('tp8', 384, 5120)],
    'down_proj_group':    [('tp1', 5120, 1536), ('tp4', 5120, 384), ('tp8', 5120, 192)],
}

COLUMNS = ['variant', 'E', 'n', 'k', 'T', 'sum_m',
           'us', 'tflops', 'gbps', 'sm_mhz', 'power_w', 'backend']


def worker(cat, variant, t_values=None):
    import csv

    import torch

    from bench_lib import (
        GROUP_E, M_ALL, RESULT_DIR, SUBSTR, NvmlSampler, build_group_point,
        build_group_weights, measure, reset_cublaslt_plans,
    )

    (n, k), = [(n, k) for v, n, k in CATEGORIES[cat] if v == variant]
    f = open(f'{RESULT_DIR}/{cat}.csv', 'a', newline='')
    w = csv.writer(f)
    sampler = NvmlSampler(index=0)
    try:
        w_fp4, w_bs, w_gs, w_hp = build_group_weights(n, k, keep_hp=True)
        for T in (t_values or M_ALL):
            try:
                reset_cublaslt_plans()
                run_ct, run_lt, flops, nbytes, sum_m, ref = build_group_point(
                    w_fp4, w_bs, w_gs, T, n, k, w_hp=w_hp)
                out_ct = run_ct()
                out_lt = run_lt()
                torch.cuda.synchronize()
                if not torch.equal(out_ct, out_lt):
                    bad = (out_ct != out_lt).any(dim=1).sum().item()
                    print(f'!! {variant} T={T}: backends disagree on {bad}/{out_ct.shape[0]} '
                          f'rows', flush=True)
                relerr = ((out_ct.float() - ref).norm() / ref.norm()).item()
                if not (0.08 < relerr < 0.20):   # nvfp4 quant error on randn is ~0.134
                    print(f'!! {variant} T={T}: relerr vs fp32 ref = {relerr:.4f} '
                          f'(outside nvfp4 expectation)', flush=True)
                del out_ct, out_lt, ref
                # cublasLt runs the heuristic's algo[0] (its predicted best) -- fixed, no
                # autotune: measured best == algo[0] at 81/84 points, and a framework
                # integration would run algo[0] anyway. Known exceptions (measured via
                # per-algo kineto sweep): gate_up T=16384 tp4/tp8, where algo2 is
                # ~20%/~5% faster -- noted in the README, not benchmarked in.
                for backend, run in (('CUTLASS', run_ct), ('CUBLASLT', run_lt)):
                    us, tflops, gbps, sm, pw = measure(
                        run, SUBSTR[('group', backend)], sampler, flops, nbytes)
                    w.writerow([variant, GROUP_E, n, k, T, sum_m,
                                f'{us:.3f}', f'{tflops:.2f}', f'{gbps:.2f}',
                                f'{sm:.0f}', f'{pw:.1f}', backend])
                    f.flush()
                    print(f'{variant:4} T={T:6} sum_m={sum_m:7} n={n:5} k={k:5} | '
                          f'{us:9.2f} us {tflops:8.1f} TF {gbps:8.1f} GB/s | {backend}',
                          flush=True)
                del run_ct, run_lt
                torch.cuda.empty_cache()
            except torch.AcceleratorError as ex:
                print(f'{variant:4} T={T:6} | CUDA FAULT, aborting variant: '
                      f'{str(ex)[:120]}', flush=True)
                break
            except Exception as ex:
                print(f'{variant:4} T={T:6} n={n:5} k={k:5} | ERROR '
                      f'{type(ex).__name__}: {str(ex)[:120]}', flush=True)
                torch.cuda.empty_cache()
    finally:
        sampler.close()
        f.close()


def _prune_and_missing(path, variants):
    """Keep only points with BOTH backend rows; return {variant: [missing T...]}."""
    from bench_lib import M_ALL
    rows = []
    if os.path.exists(path):
        with open(path) as f:
            rows = [r for r in csv_mod.reader(f)][1:]
    have = {}
    for r in rows:
        have.setdefault((r[0], int(r[4])), set()).add(r[-1])
    complete = {key for key, b in have.items() if b >= {'CUTLASS', 'CUBLASLT'}}
    with open(path, 'w', newline='') as f:
        w = csv_mod.writer(f)
        w.writerow(COLUMNS)
        seen = set()
        for r in rows:
            key = (r[0], int(r[4]), r[-1])
            if (key[0], key[1]) in complete and key not in seen:
                seen.add(key)
                w.writerow(r)
    return {v: [T for T in M_ALL if (v, T) not in complete] for v, _, _ in variants}


def runner(cats):
    from bench_lib import RESULT_DIR
    os.makedirs(RESULT_DIR, exist_ok=True)
    for cat in cats:
        path = f'{RESULT_DIR}/{cat}.csv'
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            with open(path, 'w', newline='') as f:
                f.write(','.join(COLUMNS) + '\n')
        print(f'=== {cat} ===', flush=True)
        for attempt in range(1, MAX_ATTEMPTS + 1):
            missing = _prune_and_missing(path, CATEGORIES[cat])
            todo = {v: ts for v, ts in missing.items() if ts}
            if not todo:
                break
            for variant, ts in todo.items():
                if attempt > 1:
                    print(f'-- retry {attempt} for {cat}/{variant}: T={ts}', flush=True)
                r = subprocess.run([sys.executable, os.path.abspath(__file__), '--worker',
                                    cat, variant, ','.join(map(str, ts))])
                if r.returncode != 0:
                    print(f'!! worker {cat}/{variant} exited with code {r.returncode}',
                          flush=True)
        missing = _prune_and_missing(path, CATEGORIES[cat])
        left = {v: ts for v, ts in missing.items() if ts}
        if left:
            print(f'!! {cat}: still missing after {MAX_ATTEMPTS} attempts: {left}', flush=True)
        print(f'-> {path}\n', flush=True)


if __name__ == '__main__':
    if len(sys.argv) >= 2 and sys.argv[1] == '--worker':
        ts = [int(x) for x in sys.argv[4].split(',')] if len(sys.argv) > 4 else None
        worker(sys.argv[2], sys.argv[3], ts)
    else:
        cats = sys.argv[1:] or list(CATEGORIES)
        for c in cats:
            assert c in CATEGORIES, f'unknown category {c}; choose from {list(CATEGORIES)}'
        runner(cats)
