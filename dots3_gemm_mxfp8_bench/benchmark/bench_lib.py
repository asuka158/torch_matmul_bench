"""Shared library for the dots3 mxfp8 GEMM benchmark (dense + group, CUTLASS vs cuBLASLt).

Methodology = same run10_avg convention as dots3_gemm_nvfp4_bench (see ../README.md):
  * timing: bench_kineto(num_tests=10, flush_l2=True) -> CUPTI pure-kernel time of the
    GEMM MAIN kernel only, selected by an exact-substring match:
        dense CUTLASS  'device_kernel'        (cutlass_dense_mx JIT ext)
        dense cuBLASLt 'nvjet'                (torch._scaled_mm_v2, BlockWise1x32)
        group CUTLASS  'GroupProblemShape'    (torch._scaled_grouped_mm_v2 -> cutlass_mslk;
                                               excludes the set_grouped_gemm_args prologue)
        group cuBLASLt 'nvjet'                (cublaslt_mxfp8_group_mm; excludes the python
                                               pointer-array updates)
  * sm_mhz / power_w: NVML sampler, AVERAGE over the wall-clock window wrapping the single
    bench_kineto call (diluted by the flush memsets -- not a sustained operating point).
  * tflops = 2*m*n*k / t (group: 2*sum_m*n*k / t); gbps = operand+scale+output bytes / t.
  * quantization (untimed, once per point): manual mxfp8 (quant_utils.quant_mxfp8: e4m3
    data + per-32 e8m0 scales, torchao to_blocked 32x4x4 swizzle). mxfp8 has NO per-tensor
    global scale -> alpha = 1.0 everywhere; no equal-alpha alignment needed for cublasLt.
  * group A-scale layout (both backends): concatenated per-expert swizzled segments, each
    128-row padded (verified against fp32 ref in ../probe_group.py, incl. m_e = 0).
"""
import csv
import os
import sys
import threading
import time

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(_HERE)  # dots3_gemm_mxfp8_bench
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

from quant_utils import ceil_div, quant_mxfp8, to_blocked  # noqa: E402
from dg_bench import bench_kineto, count_bytes  # noqa: E402

RESULT_DIR = os.path.join(PARENT, 'result')
NUM_TESTS = 10

M_DECODE = [1, 2, 4, 8, 16, 32, 64, 128, 256]
M_PREFILL = [1024, 2048, 4096, 8192, 16384]
M_ALL = M_DECODE + M_PREFILL

SUBSTR = {
    ('dense', 'CUTLASS'): 'device_kernel',
    ('dense', 'CUBLASLT'): 'nvjet',
    ('group', 'CUTLASS'): 'GroupProblemShape',
    ('group', 'CUBLASLT'): 'nvjet',
}

# dense CUTLASS config exported by cutlass_dense_mx: '2sm' (256x256x128/2SM/cluster4x4)
# or '1sm' (128x128x128/1SM). Selection is a FIXED M-threshold rule decided by the
# one-time measured sweep in ../probe_dense_cfg.py (min-max-regret; see README) --
# deterministic per shape, no runtime search, symmetric to cublasLt's per-shape
# heuristic. Env MXFP8_DENSE_CUTLASS_CFG=2sm|1sm forces a single config.
DENSE_CUTLASS_CFG = os.environ.get('MXFP8_DENSE_CUTLASS_CFG', 'auto')
DENSE_1SM_MAX_M = 1024


def dense_cutlass_cfg(m):
    if DENSE_CUTLASS_CFG in ('2sm', '1sm'):
        return DENSE_CUTLASS_CFG
    return '1sm' if m <= DENSE_1SM_MAX_M else '2sm'

from torch._C import _ScalingType as _ST, _SwizzleType as _SW  # noqa: E402
REC = [_ST.BlockWise1x32.value]
SWZ = [_SW.SWIZZLE_32_4_4.value]


def _sc_cols(k):
    return ceil_div(k // 32, 4) * 4


def _cat_e8m0(chunks):
    """cat for float8_e8m0fnu (cat_cuda not implemented for e8m0) via uint8 views."""
    return torch.cat([c.view(torch.uint8) for c in chunks]).view(torch.float8_e8m0fnu)


# ---------------------------------------------------------------------------
# NVML window sampler (same as dots3_gemm_nvfp4_bench)
# ---------------------------------------------------------------------------
class NvmlSampler:
    def __init__(self, index=0, interval=0.001):
        import pynvml
        self._nvml = pynvml
        pynvml.nvmlInit()
        self._h = pynvml.nvmlDeviceGetHandleByIndex(index)
        self._interval = interval
        self._samples = []
        self._stop = False
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while not self._stop:
            try:
                t = time.time()
                mhz = self._nvml.nvmlDeviceGetClockInfo(self._h, self._nvml.NVML_CLOCK_SM)
                pw = self._nvml.nvmlDeviceGetPowerUsage(self._h) / 1000.0
                self._samples.append((t, mhz, pw))
            except Exception:
                pass
            time.sleep(self._interval)

    def window_avg(self, t0, t1, start=0):
        win = [(c, p) for (t, c, p) in self._samples[start:] if t0 <= t <= t1]
        if not win:
            return float('nan'), float('nan')
        return (sum(c for c, _ in win) / len(win),
                sum(p for _, p in win) / len(win))

    def n_samples(self):
        return len(self._samples)

    def close(self):
        self._stop = True
        self._thread.join()
        self._nvml.nvmlShutdown()


# ---------------------------------------------------------------------------
# measurement (one backend, one point)
# ---------------------------------------------------------------------------
def measure(run, kernel_substr, sampler, flops, nbytes):
    """-> (us, tflops, gbps, sm_mhz, power_w). run() must be warm (called once before)."""
    run()
    torch.cuda.synchronize()
    i0 = sampler.n_samples() if sampler else 0
    t0 = time.time()
    sec = bench_kineto(run, kernel_substr, num_tests=NUM_TESTS,
                       suppress_kineto_output=True, flush_l2=True)
    t1 = time.time()
    sm, pw = sampler.window_avg(t0, t1, start=i0) if sampler else (float('nan'),) * 2
    return sec * 1e6, flops / sec / 1e12, nbytes / sec / 1e9, sm, pw


# ---------------------------------------------------------------------------
# dense: operands + run closures (both backends share the same quantized buffers)
# ---------------------------------------------------------------------------
def build_dense(m, n, k, seed=0):
    import cutlass_dense_mx as cdm

    torch.manual_seed(seed)
    a_hp = torch.randn(m, k, dtype=torch.bfloat16, device='cuda')
    b_hp = torch.randn(n, k, dtype=torch.bfloat16, device='cuda')  # weight [N, K]
    a_q, a_s = quant_mxfp8(a_hp)
    b_q, b_s = quant_mxfp8(b_hp)
    del a_hp, b_hp
    sa = to_blocked(a_s)
    sb = to_blocked(b_s)
    alpha = torch.ones(1, dtype=torch.float32, device='cuda')

    mm_ct = {'2sm': cdm.mxfp8_mm_2sm, '1sm': cdm.mxfp8_mm_1sm}[dense_cutlass_cfg(m)]
    run_ct = lambda: mm_ct(a_q, b_q, sa, sb, alpha)

    cols = _sc_cols(k)
    sa2, sb2 = sa.view(-1, cols), sb.view(-1, cols)
    mat_b = b_q.t()
    run_lt = lambda: torch._scaled_mm_v2(
        a_q, mat_b, [sa2], REC, SWZ, [sb2], REC, SWZ,
        None, torch.bfloat16, (), False)

    d = run_ct()
    torch.cuda.synchronize()
    nbytes = count_bytes(a_q, sa, b_q, sb, d)
    del d
    return run_ct, run_lt, nbytes


# ---------------------------------------------------------------------------
# group: weights once per (n, k); activations/routing per T
# ---------------------------------------------------------------------------
GROUP_E = 257   # 256 routed + 1 shared expert fused as expert id 256
GROUP_TOPK = 9  # 8 distinct routed experts + the shared expert on EVERY token


def build_group_weights(n, k, E=GROUP_E, seed=0, keep_hp=False):
    """-> (w_q [E,n,k] e4m3, w_sc [E, round_up(n,128)*sc_cols] e8m0 swizzled[, w_hp])."""
    torch.manual_seed(seed)
    w_hp = torch.randn(E, n, k, dtype=torch.bfloat16, device='cuda')
    w_q = torch.empty(E, n, k, dtype=torch.float8_e4m3fn, device='cuda')
    sc_list = []
    for e in range(E):
        q, s = quant_mxfp8(w_hp[e])
        w_q[e] = q
        sc_list.append(to_blocked(s))
    w_sc = _cat_e8m0(sc_list).view(E, -1)
    if keep_hp:
        return w_q, w_sc, w_hp
    del w_hp
    torch.cuda.empty_cache()
    return w_q, w_sc


def build_group_point(w_q, w_sc, T, n, k, E=GROUP_E, topk=GROUP_TOPK, seed=0,
                      w_hp=None):
    """Routing: each token picks 8 DISTINCT experts out of the 256 routed ones plus the
    shared expert (id 256) -> sum_m = T*topk, shared expert has m=T.
    -> (run_ct, run_lt, flops, nbytes, sum_m[, ref]) -- ref is the fp32 grouped-layout
    reference (computed from the bf16 sources) when w_hp is given."""
    import cublaslt_group_gemm_mx as cmx

    dev = 'cuda'
    torch.manual_seed(seed)
    routed = torch.rand(T, E - 1, device=dev).topk(topk - 1, dim=1).indices.to(torch.int32)
    shared = torch.full((T, 1), E - 1, dtype=torch.int32, device=dev)
    topk_ids = torch.cat([routed, shared], dim=1).contiguous()
    del routed, shared
    sum_m = T * topk

    # routing bookkeeping in pure torch (no sgl_kernel prepare_moe_input needed)
    flat = topk_ids.flatten().to(torch.int64)
    perm = torch.argsort(flat, stable=True)          # grouped row -> flat slot
    counts = torch.bincount(flat, minlength=E)       # [E] m_e
    ends = torch.cumsum(counts, 0)
    starts = ends - counts
    m_host = counts.cpu().tolist()

    a_hp = torch.randn(T, k, dtype=torch.bfloat16, device=dev)
    rep_a_hp = a_hp[(perm // topk)]                  # grouped-layout bf16 tokens
    del a_hp
    a_q, a_s = quant_mxfp8(rep_a_hp)

    bounds = [0]
    for c in m_host:
        bounds.append(bounds[-1] + c)
    sa_flat = _cat_e8m0([to_blocked(a_s[bounds[e]:bounds[e + 1]])
                         for e in range(E) if m_host[e] > 0])
    bs_rows = [0]
    for c in m_host:
        bs_rows.append(bs_rows[-1] + ceil_div(c, 128) * 128)

    ref = None
    if w_hp is not None:
        ref = torch.empty(sum_m, n, dtype=torch.float32, device=dev)
        for e in range(E):
            lo, hi = bounds[e], bounds[e + 1]
            if hi > lo:
                ref[lo:hi] = rep_a_hp[lo:hi].float() @ w_hp[e].float().t()
    del rep_a_hp, a_s

    cols = _sc_cols(k)
    offs = ends.to(torch.int32)                      # torch grouped: cumulative ENDs
    mat_b = w_q.transpose(-2, -1)                    # [E, K, N]
    sa_2d = sa_flat.view(-1, cols)
    run_ct = lambda: torch._scaled_grouped_mm_v2(
        a_q, mat_b, [sa_2d], REC, SWZ, [w_sc], REC, SWZ,
        offs, None, torch.bfloat16, (), False)

    params = {
        'expert_offsets': starts.to(torch.int32),
        'blockscale_offsets': torch.tensor(bs_rows[:-1], dtype=torch.int32, device=dev),
        'problem_sizes': torch.stack(
            [counts.to(torch.int32),
             torch.full((E,), n, dtype=torch.int32, device=dev),
             torch.full((E,), k, dtype=torch.int32, device=dev)], dim=1).contiguous(),
    }
    run_lt = lambda: cmx.cublaslt_mxfp8_group_mm(
        a_q, w_q, sa_flat, w_sc, torch.bfloat16, dev, params)

    flops = 2.0 * sum_m * n * k
    n_active = sum(1 for c in m_host if c > 0)
    per_w_sc = w_sc.shape[1]
    nbytes = (sum_m * k                              # A e4m3
              + sa_flat.numel()                      # A blockscales (128-padded rows)
              + n_active * n * k                     # B e4m3 (active experts)
              + n_active * per_w_sc                  # B blockscales (active experts)
              + sum_m * n * 2)                       # D bf16
    if w_hp is not None:
        return run_ct, run_lt, flops, nbytes, sum_m, ref
    return run_ct, run_lt, flops, nbytes, sum_m


def reset_cublaslt_plans():
    """Drop cached cublasLt grouped plans so the next call re-runs the heuristic with the
    current point's sum_m hint; destroys C++-side plan objects and zeroes the shared
    workspace so no state leaks between points."""
    import cublaslt_group_gemm_mx as cmx
    if cmx._plans:
        ext = cmx.load_ext()
        for p in cmx._plans.values():
            try:
                ext.destroy_plan(p.pid)
            except Exception:
                pass
        cmx._plans.clear()
    if cmx._workspace is not None:
        cmx._workspace.zero_()


# ---------------------------------------------------------------------------
# csv
# ---------------------------------------------------------------------------
def csv_writer(path, columns):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    f = open(path, 'w', newline='')
    w = csv.writer(f)
    w.writerow(columns)
    f.flush()

    def emit(row):
        w.writerow(row)
        f.flush()
    return emit, f
