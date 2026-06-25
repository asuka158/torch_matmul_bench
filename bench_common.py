"""
Shared helpers for the Python-side FP4 dense GEMM benchmarks (D = A @ B, no C/bias).

Two backends, reached purely through ``torch._scaled_mm_v2``:
  * NVFP4 x NVFP4  -> cuBLASLt  (block-scale recipe BlockWise1x16, UE4M3 scales)
  * MXFP4 x MXFP4  -> CUTLASS   (block-scale recipe BlockWise1x32, UE8M0 scales)
Backend is not user-selectable in torch scaled ops: NVFP4 always lands on cuBLASLt and MXFP4
always on CUTLASS (cuBLASLt has no MXFP4), so the format choice picks the backend.

Operands are quantized once per shape with torchao's NVFP4Tensor / MXTensor (scales pre-swizzled),
then the timed call is a bare ``torch._scaled_mm_v2`` (mirrors torchao's own _scaled_mm dispatch in
nvfp4_tensor.py / mx_tensor.py). The left operand A is [M,K] contiguous; the right operand is built
as a [N,K] weight and transposed (b.t() -> [K,N], so b.qdata.t() is contiguous).

Methodology (per request):
  * timing: deep_gemm.testing.bench_kineto -> CUPTI pure-kernel time, num_tests=30, flush_l2=False.
  * sm_mhz / power_w: a separate execution-aligned pass (back-to-back GEMMs, no L2 flush) with a
    background NVML sampler; we average only the samples that fall inside the kernel-execution window.
    No L2 flush anywhere, so the SMs run GEMMs back-to-back and every NVML sample is execution time
    (a flush memset would be a low-power gap that would drag the average down).

Run (note the libucs fix + PYTHONPATH for torch and deep_gemm):
  cd python/nvfp4
  LD_LIBRARY_PATH=/opt/hpcx/ucx/lib \
  PYTHONPATH=/root/workspace/gb_gemm_benchmark/python:/root/workspace/gb_gemm_benchmark/DeepGEMM \
  python bench_nvfp4_bf16.py
"""
import os
import sys
import time
import random
import threading

import torch
from torch._C import _ScalingType as ScalingType, _SwizzleType as SwizzleType

# --- make `deep_gemm` (for bench_kineto) importable regardless of how PYTHONPATH was set ---
_REPO_ROOT = '/root/workspace/gb_gemm_bench'
_DEEPGEMM_REPO = _REPO_ROOT + '/DeepGEMM'
if _DEEPGEMM_REPO not in sys.path:
    sys.path.insert(0, _DEEPGEMM_REPO)

from deep_gemm.testing import bench_kineto, count_bytes  # noqa: E402

from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor  # noqa: E402
from torchao.prototype.mx_formats.mx_tensor import MXTensor  # noqa: E402

import pynvml  # noqa: E402

SHAPE_PATH = _DEEPGEMM_REPO + '/tests/my_test/shape.txt'  # legacy space-separated 'm n k'
SHAPE_CSV_PATH = _REPO_ROOT + '/shape.csv'                # 50-shape sweep, header 'M,K,N'
NUM_TESTS = 30
_SWZ = SwizzleType.SWIZZLE_32_4_4.value


# ---------------------------------------------------------------------------
# shapes
# ---------------------------------------------------------------------------
def read_shapes(path=SHAPE_PATH):
    """Parse (m, n, k) triples; skip the header row and blank lines."""
    shapes = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.lower().startswith('m'):
                continue
            m, n, k = (int(x) for x in line.split())
            shapes.append((m, n, k))
    return shapes


def read_shapes_csv(path=SHAPE_CSV_PATH):
    """Parse the 50-shape sweep from shape.csv.

    Header is 'M,K,N' (comma-separated). The kernels want (m, n, k), so each row (M, K, N)
    maps to (M, N, K) -- identical to DeepGEMM/tests/dg_test/bench_fp4_run30.py:read_shapes.
    """
    shapes = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.lower().startswith('m'):
                continue
            M, K, N = (int(x) for x in line.split(','))
            shapes.append((M, N, K))
    return shapes


# ---------------------------------------------------------------------------
# NVML window-level sampler (SM clock + power), aligned to the execution window
# ---------------------------------------------------------------------------
class NvmlSampler:
    def __init__(self, index=0, interval=0.001):
        pynvml.nvmlInit()
        self._h = pynvml.nvmlDeviceGetHandleByIndex(index)
        self._interval = interval
        self._samples = []  # (t, sm_mhz, power_w)
        self._stop = False
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while not self._stop:
            try:
                t = time.time()
                mhz = pynvml.nvmlDeviceGetClockInfo(self._h, pynvml.NVML_CLOCK_SM)
                pw = pynvml.nvmlDeviceGetPowerUsage(self._h) / 1000.0  # mW -> W
                self._samples.append((t, mhz, pw))
            except Exception:
                pass
            time.sleep(self._interval)

    def window_avg(self, t0, t1, start=0):
        """Average SM MHz / power (W) over samples whose timestamp is in [t0, t1].

        ``start`` is an index into the sample buffer (from a prior ``n_samples()`` call) so a long
        run can bound the scan to the samples it produced instead of the whole accumulated history.
        """
        win = [(c, p) for (t, c, p) in self._samples[start:] if t0 <= t <= t1]
        if not win:
            return float('nan'), float('nan'), 0
        sm = sum(c for c, _ in win) / len(win)
        pw = sum(p for _, p in win) / len(win)
        return sm, pw, len(win)

    def window_median(self, t0, t1, start=0):
        """Median SM MHz / power (W) over samples in [t0, t1] -- matches the median() in
        DeepGEMM/tests/dg_test/bench_fp4_run30.py (sort, take element at len//2, NaN if empty).

        Returns (sm_mhz, power_w, n_samples). ``start`` bounds the scan to samples produced
        after a prior ``n_samples()`` call.
        """
        clk = sorted(c for (t, c, _) in self._samples[start:] if t0 <= t <= t1)
        pwr = sorted(p for (t, _, p) in self._samples[start:] if t0 <= t <= t1)
        med = lambda v: v[len(v) // 2] if v else float('nan')
        return med(clk), med(pwr), len(clk)

    def n_samples(self):
        return len(self._samples)

    def close(self):
        self._stop = True
        self._thread.join()
        pynvml.nvmlShutdown()


# ---------------------------------------------------------------------------
# operand construction -> no-arg run() closure doing a bare _scaled_mm_v2
# ---------------------------------------------------------------------------
def _make_run(mat_a, mat_b, sa, sb, recipe, out_dtype):
    rec = [recipe]
    swz = [_SWZ]
    sca, scb = [sa], [sb]

    def run():
        return torch._scaled_mm_v2(
            mat_a, mat_b,
            sca, rec, swz,
            scb, rec, swz,
            None,          # bias
            out_dtype,     # output_dtype
            (),            # contraction_dim (default)
            False,         # use_fast_accum
        )
    return run


def _build_nvfp4(m, n, k, out_dtype):
    """NVFP4 x NVFP4 operands -> (run, in_tensors). in_tensors = [A_fp4, A_scale, B_fp4, B_scale]
    (the bytes touched on the input side, for count_bytes-based GB/s)."""
    a_hp = torch.randn(m, k, dtype=torch.bfloat16, device='cuda')
    b_hp = torch.randn(n, k, dtype=torch.bfloat16, device='cuda')  # weight [N,K]
    a = NVFP4Tensor.to_nvfp4(a_hp, is_swizzled_scales=True)        # [M,K]
    b = NVFP4Tensor.to_nvfp4(b_hp, is_swizzled_scales=True)        # [N,K]
    bt = b.t()                                                     # [K,N]
    mat_a = a.qdata.view(torch.float4_e2m1fn_x2)
    mat_b = bt.qdata.view(torch.float4_e2m1fn_x2)
    sa = a.scale.view(torch.float8_e4m3fn)
    sb = bt.scale.t().view(torch.float8_e4m3fn)
    run = _make_run(mat_a, mat_b, sa, sb, ScalingType.BlockWise1x16.value, out_dtype)
    return run, [mat_a, sa, mat_b, sb]


def _build_mxfp4(m, n, k, out_dtype):
    """MXFP4 x MXFP4 operands -> (run, in_tensors), same convention as _build_nvfp4."""
    a_hp = torch.randn(m, k, dtype=torch.bfloat16, device='cuda')
    b_hp = torch.randn(n, k, dtype=torch.bfloat16, device='cuda')
    a = MXTensor.to_mx(a_hp, torch.float4_e2m1fn_x2, 32, is_swizzled_scales=True)
    b = MXTensor.to_mx(b_hp, torch.float4_e2m1fn_x2, 32, is_swizzled_scales=True)
    bt = b.t()
    mat_a = a.qdata.view(torch.float4_e2m1fn_x2)
    mat_b = bt.qdata.view(torch.float4_e2m1fn_x2)
    sa = a.scale
    sb = bt.scale.t()
    run = _make_run(mat_a, mat_b, sa, sb, ScalingType.BlockWise1x32.value, out_dtype)
    return run, [mat_a, sa, mat_b, sb]


def prep_nvfp4(m, n, k, out_dtype):
    return _build_nvfp4(m, n, k, out_dtype)[0]


def prep_mxfp4(m, n, k, out_dtype):
    return _build_mxfp4(m, n, k, out_dtype)[0]


_PREP = {'nvfp4': prep_nvfp4, 'mxfp4': prep_mxfp4}
_BUILD = {'nvfp4': _build_nvfp4, 'mxfp4': _build_mxfp4}
_KERNEL_SUBSTR = {'nvfp4': 'nvjet_sm100', 'mxfp4': 'cutlass'}


# ---------------------------------------------------------------------------
# measurement
# ---------------------------------------------------------------------------
def measure_time(run, kernel_substr, m, n, k, out_elem_bytes):
    """CUPTI pure-kernel time via bench_kineto -> (us, tflops, gbps)."""
    sec = bench_kineto(run, kernel_substr, num_tests=NUM_TESTS,
                       suppress_kineto_output=True, flush_l2=False)
    tflops = 2.0 * m * n * k / sec / 1e12
    # bytes: fp4 A + fp4 B + out-dtype D (scales negligible) -- same formula as bench_nvfp4.cu
    gbytes = (m * k * 0.5 + n * k * 0.5 + m * n * out_elem_bytes) / sec / 1e9
    return sec * 1e6, tflops, gbytes


def measure_telemetry(run, sampler, min_samples=8, max_seconds=0.1):
    """
    Average SM clock + power over the kernel-execution window. Runs NUM_TESTS GEMMs back-to-back
    (no flush -> pure execution); if that window is too short for NVML to catch enough samples
    (tiny shapes), keep issuing NUM_TESTS-sized blocks until min_samples are gathered or max_seconds
    elapses (cap so we never chase steady state). Big shapes -- the ones that actually throttle --
    stop after the first block of NUM_TESTS.
    """
    for _ in range(5):  # warmup
        run()
    torch.cuda.synchronize()
    t0 = time.time()
    while True:
        for _ in range(NUM_TESTS):
            run()
        torch.cuda.synchronize()
        t1 = time.time()
        _, _, ns = sampler.window_avg(t0, t1)
        if ns >= min_samples or (t1 - t0) >= max_seconds:
            break
    sm, pw, _ = sampler.window_avg(t0, t1)
    return sm, pw


# ---------------------------------------------------------------------------
# csv + driver
# ---------------------------------------------------------------------------
_COLUMNS = ['m', 'n', 'k', 'us', 'tflops', 'gbps', 'sm_mhz', 'power_w', 'backend']


def write_csv(path, rows):
    import csv
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(_COLUMNS)
        for r in rows:
            w.writerow([r['m'], r['n'], r['k'],
                        f"{r['us']:.3f}", f"{r['tflops']:.2f}", f"{r['gbps']:.2f}",
                        f"{r['sm_mhz']:.0f}", f"{r['power_w']:.1f}", r['backend']])


def run_suite(fmt, out_dtype, csv_path, backend):
    assert fmt in _PREP, fmt
    prep = _PREP[fmt]
    kernel_substr = _KERNEL_SUBSTR[fmt]
    out_elem_bytes = torch.empty(0, dtype=out_dtype).element_size()

    torch.manual_seed(0)
    shapes = read_shapes()

    print(f'Device: {torch.cuda.get_device_name(0)}')
    print(f'Backend: {backend} | format={fmt} | out_dtype={out_dtype} | kernel~"{kernel_substr}"')
    print(f'Benchmarking {len(shapes)} shapes (D = A @ B, num_tests={NUM_TESTS}, flush_l2=False)\n')
    print(f'{"m":>6} {"n":>6} {"k":>6} | {"us":>9} {"TFLOPS":>8} {"GB/s":>8} | {"sm":>5} {"pw":>6}')

    sampler = NvmlSampler(index=0)
    rows = []
    try:
        for idx, (m, n, k) in enumerate(shapes):
            run = prep(m, n, k, out_dtype)
            us, tflops, gbps = measure_time(run, kernel_substr, m, n, k, out_elem_bytes)
            sm, pw = measure_telemetry(run, sampler)
            rows.append(dict(m=m, n=n, k=k, us=us, tflops=tflops, gbps=gbps,
                             sm_mhz=sm, power_w=pw, backend=backend))
            print(f'{m:6} {n:6} {k:6} | {us:9.2f} {tflops:8.1f} {gbps:8.1f} | '
                  f'{sm:5.0f} {pw:6.1f}')
            del run
            torch.cuda.empty_cache()
    finally:
        sampler.close()

    write_csv(csv_path, rows)
    print(f'\nDone. {len(rows)} shapes -> {csv_path}')


# ---------------------------------------------------------------------------
# run30 measurement -- identical method to DeepGEMM/tests/dg_test/bench_fp4_run30.py
# ---------------------------------------------------------------------------
# Single pass per shape, fully aligned with bench_fp4_run30.py's test_fp4-style measurement:
#   * t = bench_kineto(run, kernel, num_tests=30, flush_l2=True)  -> CUPTI pure-kernel time
#   * sm_mhz / power_w = NVML MEDIAN over the wall-clock window [t0, t1] wrapping THAT single
#     bench_kineto call (one pass -- no separate telemetry loop, unlike run_suite()).
#   * tflops = 2*m*n*k / t ;  gbps = count_bytes(A_fp4, A_scale, B_fp4, B_scale, D) / t.
#   * CSV columns m,n,k,us,tflops,gbps,sm_mhz,power_w,backend.
#   * shapes from shape.csv (row M,K,N -> m,n,k).
#
# NOTE (same caveat as bench_fp4_run30.py): flush_l2=True memsets 8 GB before each of the 30+30
# fn() calls, and the window is short, so sm_mhz / power_w are diluted by the low-power memset gaps
# and do NOT represent the GEMM's true sustained operating point. tflops is still the pure GEMM
# kernel time (bench_kineto extracts it by kernel name), but reflects a short, near-peak window.
def measure_run30(run, kernel_substr, sampler, m, n, k, nbytes,
                  num_tests=NUM_TESTS, agg='median'):
    """One shape, bench_fp4_run30.py-style measurement. ``run`` must launch ONLY the GEMM.

    Returns (us, tflops, gbps, sm_mhz, power_w):
      * us/tflops from bench_kineto(num_tests, flush_l2=True)  (CUPTI pure-kernel time)
      * gbps = nbytes / t  (caller supplies the operand+scale+output byte count)
      * sm_mhz/power_w = NVML over the window wrapping that single bench_kineto call,
        aggregated as ``agg``: 'median' (bench_fp4_run30 default) or 'avg' (window mean).
    """
    run()
    torch.cuda.synchronize()
    i0 = sampler.n_samples()
    t0 = time.time()
    sec = bench_kineto(run, kernel_substr, num_tests=num_tests,
                       suppress_kineto_output=True, flush_l2=True)
    t1 = time.time()
    window = sampler.window_avg if agg == 'avg' else sampler.window_median
    sm, pw, _ = window(t0, t1, start=i0)
    return sec * 1e6, 2.0 * m * n * k / sec / 1e12, nbytes / 1e9 / sec, sm, pw


def run_suite_run30(fmt, out_dtype, csv_path, backend, num_tests=NUM_TESTS, agg='median'):
    assert fmt in _BUILD, fmt
    build = _BUILD[fmt]
    kernel_substr = _KERNEL_SUBSTR[fmt]

    # seed exactly as bench_fp4_run30.py: same torch.randn bf16 stream for A/B operands
    torch.manual_seed(0)
    random.seed(0)
    shapes = read_shapes_csv()

    print(f'Device: {torch.cuda.get_device_name(0)}')
    print(f'Backend: {backend} | format={fmt} | out_dtype={out_dtype} | kernel~"{kernel_substr}"')
    print(f'run method: num_tests={num_tests}, flush_l2=True, NVML {agg} over the bench_kineto window')
    print(f'Benchmarking {len(shapes)} shapes (D = A @ B)\n')
    print(f'{"m":>6} {"n":>6} {"k":>6} | {"us":>9} {"TFLOPS":>8} {"GB/s":>8} | {"sm":>5} {"pw":>6}')

    sampler = NvmlSampler(index=0)
    rows = []
    write_csv(csv_path, rows)  # write header up front (matches bench_fp4_run30.py)
    try:
        for idx, (m, n, k) in enumerate(shapes, 1):
            try:
                run, ins = build(m, n, k, out_dtype)
                d = run()                     # one untimed call -> output tensor for count_bytes
                torch.cuda.synchronize()
                nbytes = count_bytes(*ins, d)
                us, tflops, gbps, sm, pw = measure_run30(
                    run, kernel_substr, sampler, m, n, k, nbytes, num_tests=num_tests, agg=agg)
                rows.append(dict(m=m, n=n, k=k, us=us, tflops=tflops, gbps=gbps,
                                 sm_mhz=sm, power_w=pw, backend=backend))
                write_csv(csv_path, rows)
                print(f'{m:6} {n:6} {k:6} | {us:9.2f} {tflops:8.1f} {gbps:8.1f} | '
                      f'{sm:5.0f} {pw:6.1f}', flush=True)
                del run, ins, d
                torch.cuda.empty_cache()
            except Exception as ex:
                print(f'{m:6} {n:6} {k:6} | ERROR {type(ex).__name__}: {str(ex)[:80]}', flush=True)
                torch.cuda.empty_cache()
    finally:
        sampler.close()

    write_csv(csv_path, rows)
    print(f'\nDone. {len(rows)}/{len(shapes)} shapes -> {csv_path}')


# ---------------------------------------------------------------------------
# graph-sustained measurement (steady-state under a CUDA-graph replay loop)
# ---------------------------------------------------------------------------
# Alternative methodology (mirrors nvfp4/diag_graph_sustained.py): instead of CUPTI pure-kernel
# time on isolated launches, capture GRAPH_N GEMMs into a CUDA graph and replay it back-to-back for
# ~SUSTAIN_S. Host activity is minimal (one replay() drives GRAPH_N kernels) so the SMs run truly
# back-to-back and settle into their throttled steady state; we then report ONLY the steady-state
# tail (last STEADY_TAIL_S), which is the number that matters for a sustained-throughput claim.
GRAPH_N = 10          # GEMMs captured per CUDA graph (one replay == GRAPH_N kernels)
SUSTAIN_S = 3.0       # total back-to-back replay duration per shape
SEG_MS = 20.0         # target wall time per timing segment (events bracket each segment)
STEADY_TAIL_S = 1.0   # only the last STEADY_TAIL_S of the run is recorded as steady state


def _median(xs):
    xs = sorted(v for v in xs if v == v)  # drop NaN
    if not xs:
        return float('nan')
    n = len(xs)
    mid = n // 2
    return xs[mid] if n % 2 else 0.5 * (xs[mid - 1] + xs[mid])


def measure_graph_sustained(run, m, n, k, out_elem_bytes, sampler,
                            graph_n=GRAPH_N, sustain_s=SUSTAIN_S,
                            seg_ms=SEG_MS, steady_tail_s=STEADY_TAIL_S):
    """
    Capture ``graph_n`` GEMMs into a CUDA graph, replay it back-to-back for ~``sustain_s``, and
    report only the steady-state tail (last ``steady_tail_s``).

    Per-segment CUDA events are the in-stream clock (a single host sync at the very end, so the
    GPU never stalls mid-run); the background NVML sampler's samples that fall inside each segment
    give that segment's SM clock / power. Returns (us, tflops, gbps, sm_mhz, power_w), each the
    median across the steady-state segments. ``us`` is per single GEMM.
    """
    flop = 2.0 * m * n * k

    # --- capture: warm up on a side stream, then record graph_n runs on the current stream ---
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(graph_n):
            run()

    # --- size the loop from one timed replay (warm replays first) ---
    for _ in range(2):
        g.replay()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record(); g.replay(); e.record(); torch.cuda.synchronize()
    replay_ms = max(s.elapsed_time(e), 1e-3)

    seg_replays = max(1, int(round(seg_ms / replay_ms)))
    seg_ms_eff = seg_replays * replay_ms
    nseg = max(int(sustain_s * 1000 / seg_ms_eff) + 1,
               int(steady_tail_s * 1000 / seg_ms_eff) + 2)
    kernels_per_seg = seg_replays * graph_n

    evs = [torch.cuda.Event(enable_timing=True) for _ in range(nseg + 1)]

    # idle briefly so the clock recovers -> the loop captures the transient down to steady state
    torch.cuda.synchronize()
    time.sleep(0.3)

    i0 = sampler.n_samples()
    t0 = time.time()
    evs[0].record()
    for sgi in range(nseg):
        for _ in range(seg_replays):
            g.replay()
        evs[sgi + 1].record()
    torch.cuda.synchronize()   # the ONLY host sync in the loop

    gpu_ms = evs[0].elapsed_time(evs[nseg])

    # --- per-segment table -> keep only the steady-state tail ---
    cum = 0.0
    t_us, t_tf, t_sm, t_pw = [], [], [], []
    for sgi in range(nseg):
        sms = evs[sgi].elapsed_time(evs[sgi + 1])
        h0 = t0 + cum / 1000.0
        h1 = t0 + (cum + sms) / 1000.0
        sm, pw, _ = sampler.window_avg(h0, h1, start=i0)
        cum += sms
        if cum >= gpu_ms - steady_tail_s * 1000.0:
            t_us.append(sms * 1000.0 / kernels_per_seg)
            t_tf.append(kernels_per_seg * flop / (sms / 1e3) / 1e12)
            t_sm.append(sm)
            t_pw.append(pw)

    us = _median(t_us)
    tflops = _median(t_tf)
    sm = _median(t_sm)
    pw = _median(t_pw)
    gbps = (m * k * 0.5 + n * k * 0.5 + m * n * out_elem_bytes) / (us / 1e6) / 1e9

    del g
    return us, tflops, gbps, sm, pw


def run_suite_graph_sustained(fmt, out_dtype, csv_path, backend):
    """Same shapes / CSV schema as ``run_suite``, but each shape is timed with a sustained
    CUDA-graph replay loop and only the steady-state tail is recorded (see
    ``measure_graph_sustained``)."""
    assert fmt in _PREP, fmt
    prep = _PREP[fmt]
    out_elem_bytes = torch.empty(0, dtype=out_dtype).element_size()

    torch.manual_seed(0)
    shapes = read_shapes()

    print(f'Device: {torch.cuda.get_device_name(0)}')
    print(f'Backend: {backend} | format={fmt} | out_dtype={out_dtype}')
    print(f'Graph-sustained: capture {GRAPH_N} GEMMs/graph, replay ~{SUSTAIN_S:.0f}s back-to-back, '
          f'record steady-state tail (last {STEADY_TAIL_S:.0f}s)')
    print(f'Benchmarking {len(shapes)} shapes (D = A @ B)\n')
    print(f'{"m":>6} {"n":>6} {"k":>6} | {"us":>9} {"TFLOPS":>8} {"GB/s":>8} | {"sm":>5} {"pw":>6}')

    sampler = NvmlSampler(index=0)
    rows = []
    try:
        for (m, n, k) in shapes:
            run = prep(m, n, k, out_dtype)
            us, tflops, gbps, sm, pw = measure_graph_sustained(
                run, m, n, k, out_elem_bytes, sampler)
            rows.append(dict(m=m, n=n, k=k, us=us, tflops=tflops, gbps=gbps,
                             sm_mhz=sm, power_w=pw, backend=backend))
            print(f'{m:6} {n:6} {k:6} | {us:9.2f} {tflops:8.1f} {gbps:8.1f} | '
                  f'{sm:5.0f} {pw:6.1f}')
            del run
            torch.cuda.empty_cache()
    finally:
        sampler.close()

    write_csv(csv_path, rows)
    print(f'\nDone. {len(rows)} shapes -> {csv_path}')
