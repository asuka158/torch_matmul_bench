"""Shared library for the dots3 nvfp4 BMM benchmark (CUTLASS vs cuBLASLt).

Methodology mirrors ../../dots3_gemm_nvfp4_bench (see ../dots3_bmm_plan.md):
  * timing: bench_kineto(num_tests=10, flush_l2=True) -> CUPTI pure-kernel time of the
    BMM MAIN kernel only, selected by an exact substring (see SUBSTR; both backends are
    single-kernel per call, verified by profile_bmm_sweep.py).
  * sm_mhz / power_w: NVML sampler averaged over the bench_kineto window (diluted by the
    flush memsets -- not a sustained operating point).
  * tflops = 2*B*M*N*K / t ; gbps = operand+scale+output bytes / t.
  * quantization (untimed, once per point): sgl_kernel.scaled_fp4_quant with ONE
    per-TENSOR global scale shared across all batches. Neither backend supports a
    per-BATCH global scale for this op (probe_per_batch_scale.py), so both consume the
    SAME quantized buffers: cuBLASLt takes the swizzled scales, CUTLASS the same numbers
    un-swizzled (from_blocked) since it re-orders them itself.
"""
import csv
import os
import sys
import threading
import time

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(_HERE)  # dots3_bmm_bench
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

from quant_utils import (  # noqa: E402
    FLOAT4_E2M1_MAX, FLOAT8_E4M3_MAX, ceil_div, from_blocked,
)
from dg_bench import bench_kineto  # noqa: E402

RESULT_DIR = os.path.join(PARENT, 'result')
NUM_TESTS = 10

M_DECODE = [1, 2, 4, 8, 16, 32, 64, 128, 256]
M_PREFILL = [1024, 2048, 4096, 8192, 16384]
M_ALL = M_DECODE + M_PREFILL

# the four dots3 absorbed bmms: (name, batch=heads, K, N); M = T = query tokens
# A* = full layers (13, NSA geometry); C* = swa+MTP layers (33+1)
BMMS = {
    'A5_w_kc': (128, 128, 512),
    'A7_w_vc': (128, 512, 128),
    'C5_w_kc': (64, 192, 1024),
    'C7_w_vc': (64, 1024, 128),
}

SUBSTR = {
    'CUTLASS': 'bs_gemm_example',   # cute-dsl kernel name carries the module name
    # cuBLASLt dispatches to TWO kernel families here: its own nvjet_sm100_* for most
    # shapes, and -- at M=1 -- an internally shipped cutlass3x_sm100_bstensorop_*
    # block-scaled kernel. 'sm100_' covers both; it does NOT collide with the cute-dsl
    # kernel (which spells it 'Sm100'). Each call launches exactly one kernel, so the
    # bench_kineto "<=1 match" assert holds. See profile_bmm_sweep.py.
    'CUBLASLT': 'sm100_',
}


def cutlass_cfg(M, N):
    """(mma_tiler_mn, cluster_shape_mn) for the CUTLASS batched nvfp4 kernel.

    Measured ONCE over CUTLASS's own candidate tiles on the real dots3 bmm shapes
    (probe_cutlass_cfg.py / .log) and frozen as this rule -- a compile-time dispatch,
    NO runtime search, symmetric with cuBLASLt running its heuristic once per shape.
    Worst-case regret vs the per-cell best measured is ~7.6% (A7 M=1); a single fixed
    (128,128)/(1,1) everywhere would instead cost ~11% (C7 small M).

    flashinfer's own tactic table is deliberately not used: it scores candidates on 2D
    wave quantization (total_ctas vs SM count) and manufactures CTAs at small M via
    swap_ab, which does not transfer to a batched problem where L multiplies the CTA
    count -- see probe_cutlass_cfg.py's docstring.
    """
    if N <= 128:                                   # w_vc bmms (A7, C7)
        return ((128, 64), (1, 1)) if M <= 256 else ((128, 128), (1, 1))
    if M <= 64:                                    # w_kc bmms (A5, C5)
        return ((128, 64), (1, 1))
    if M <= 256:
        return ((128, 128), (1, 1))
    return ((128, 128), (1, 2))


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
        return (sum(c for c, _ in win) / len(win), sum(p for _, p in win) / len(win))

    def n_samples(self):
        return len(self._samples)

    def close(self):
        self._stop = True
        self._thread.join()
        self._nvml.nvmlShutdown()


def measure(run, kernel_substr, sampler, flops, nbytes):
    """-> (us, tflops, gbps, sm_mhz, power_w). run() must already be warm."""
    run()
    torch.cuda.synchronize()
    i0 = sampler.n_samples() if sampler else 0
    t0 = time.time()
    sec = bench_kineto(run, kernel_substr, num_tests=NUM_TESTS,
                       suppress_kineto_output=True, flush_l2=True)
    t1 = time.time()
    sm, pw = sampler.window_avg(t0, t1, start=i0) if sampler else (float('nan'),) * 2
    return sec * 1e6, flops / sec / 1e12, nbytes / sec / 1e9, sm, pw


def quantize(x_hp):
    """[L,R,K] bf16 -> (packed [L,R,K//2] uint8, swizzled sf, un-swizzled sf CPU, gs).

    ONE global scale over the whole tensor (per-TENSOR semantics, shared by all batches).
    """
    from sgl_kernel import scaled_fp4_quant
    L, R, K = x_hp.shape
    gs = (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / x_hp.abs().amax().float()
    data, sw, unsw = [], [], []
    for b in range(L):
        d, s = scaled_fp4_quant(x_hp[b].contiguous(), gs)
        data.append(d)
        sw.append(s)
        prows, pcols = s.shape
        unsw.append(from_blocked(s.flatten(), prows, pcols)[:R, :ceil_div(K, 16)])
    return (torch.stack(data).contiguous(), torch.stack(sw).contiguous(),
            torch.stack(unsw).contiguous().cpu(), gs)


def build_bmm(B, M, K, N, seed=0, want_ref=False):
    """-> (run_ct, run_lt, flops, nbytes[, ref, outs]). Both backends share the same
    quantized buffers; setup (quant, cublasLt plan, cutlass compile) is untimed."""
    import cublaslt_batched_nvfp4 as cb
    from cutlass_batched_nvfp4 import CutlassBatchedNvfp4

    dev = 'cuda'
    torch.manual_seed(seed)
    x_hp = torch.randn(B, M, K, dtype=torch.bfloat16, device=dev)
    w_hp = torch.randn(B, N, K, dtype=torch.bfloat16, device=dev)
    x_fp4, x_sw, x_unsw, gs_x = quantize(x_hp)
    w_fp4, w_sw, w_unsw, gs_w = quantize(w_hp)
    alpha = float((1.0 / (gs_x * gs_w)).item())
    ref = torch.bmm(x_hp.float(), w_hp.float().transpose(1, 2)) if want_ref else None
    del x_hp, w_hp

    out_lt = torch.empty(B, M, N, dtype=torch.bfloat16, device=dev)
    ext, pid = cb.make_plan(w_fp4, x_fp4, out_lt, w_sw, x_sw, B, M, N, K, alpha)
    run_lt = lambda: ext.run_plan(pid)

    out_ct = torch.empty(B, M, N, dtype=torch.bfloat16, device=dev)
    tile, cluster = cutlass_cfg(M, N)
    kern = CutlassBatchedNvfp4(M, N, K, B, alpha, tile, cluster).bind(
        x_fp4, w_fp4, x_unsw, w_unsw, out_ct)
    run_ct = lambda: kern()

    flops = 2.0 * B * M * N * K
    sf_rows = ceil_div(M, 128) * 128           # scale rows actually present per batch
    nbytes = (B * M * (K // 2)                 # A fp4
              + B * sf_rows * (K // 16)        # A block scales
              + B * N * (K // 2)               # W fp4
              + B * N * (K // 16)              # W block scales
              + B * M * N * 2)                 # D bf16
    if want_ref:
        return run_ct, run_lt, flops, nbytes, ref, (out_ct, out_lt)
    return run_ct, run_lt, flops, nbytes


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
