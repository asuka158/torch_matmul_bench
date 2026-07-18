"""Shared library for the dots3 nvfp4 GEMM benchmark (dense + group, CUTLASS vs cuBLASLt).

Methodology = the agreed run10_avg convention (see ../README.md):
  * timing: bench_kineto(num_tests=10, flush_l2=True) -> CUPTI pure-kernel time of the
    GEMM MAIN kernel only, selected by an exact-substring match:
        dense CUTLASS  'device_kernel'        (sgl_kernel.cutlass_scaled_fp4_mm)
        dense cuBLASLt 'nvjet'                (torch._scaled_mm_v2, BlockWise1x16)
        group CUTLASS  'GroupProblemShape'    (sgl_kernel.cutlass_fp4_group_mm; excludes
                                               the __get_group_gemm_starts prologue)
        group cuBLASLt 'nvjet'                (cublaslt_fp4_group_mm; excludes the python
                                               pointer-array updates)
  * sm_mhz / power_w: NVML sampler, AVERAGE over the wall-clock window wrapping the single
    bench_kineto call (diluted by the flush memsets -- not a sustained operating point).
  * tflops = 2*m*n*k / t (group: 2*sum_m*n*k / t); gbps = operand+scale+output bytes / t.
  * quantization (untimed, once per point): sgl_kernel.scaled_fp4_quant with a per-tensor
    global scale; group weights share ONE global scale across all experts so the
    per-expert alphas are all equal (cublasLt grouped single-scalar-alpha semantics,
    cutlass fed the same alphas -> outputs bitwise comparable).
"""
import csv
import os
import sys
import threading
import time

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(_HERE)  # dots3_gemm_nvfp4_bench
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

from quant_utils import FLOAT4_E2M1_MAX, FLOAT8_E4M3_MAX, nvfp4_quant  # noqa: E402
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


# ---------------------------------------------------------------------------
# NVML window sampler (same as torch_matmul_bench/bench_common.py)
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
    from sgl_kernel import cutlass_scaled_fp4_mm
    from torch._C import _ScalingType as ScalingType, _SwizzleType as SwizzleType

    torch.manual_seed(seed)
    a_hp = torch.randn(m, k, dtype=torch.bfloat16, device='cuda')
    b_hp = torch.randn(n, k, dtype=torch.bfloat16, device='cuda')  # weight [N, K]
    a_fp4, a_sf, gs_a = nvfp4_quant(a_hp)
    b_fp4, b_sf, gs_b = nvfp4_quant(b_hp)
    del a_hp, b_hp
    alpha = (1.0 / (gs_a * gs_b)).float()

    run_ct = lambda: cutlass_scaled_fp4_mm(a_fp4, b_fp4, a_sf, b_sf, alpha, torch.bfloat16)

    mat_a = a_fp4.view(torch.float4_e2m1fn_x2)
    mat_b = b_fp4.view(torch.float4_e2m1fn_x2).t()
    rec = [ScalingType.BlockWise1x16.value]
    swz = [SwizzleType.SWIZZLE_32_4_4.value]
    run_lt = lambda: torch._scaled_mm_v2(
        mat_a, mat_b, [a_sf], rec, swz, [b_sf], rec, swz,
        None, torch.bfloat16, (), False)

    d = run_ct()
    torch.cuda.synchronize()
    nbytes = count_bytes(a_fp4, a_sf, b_fp4, b_sf, d)
    del d
    return run_ct, run_lt, nbytes


# ---------------------------------------------------------------------------
# group: weights once per (n, k); activations/routing per T
# ---------------------------------------------------------------------------
GROUP_E = 257   # 256 routed + 1 shared expert fused as expert id 256
GROUP_TOPK = 9  # 8 distinct routed experts + the shared expert on EVERY token


def _padded_i32(numel):
    """int32 CUDA tensor of `numel` entries inside a buffer with 16 ZEROED tail ints.

    Workaround for the sgl_kernel cvt_fp16_to_fp4 bug documented in
    sglang/backend_test/fp4_utils.py: the experts-quant kernel reads the offsets
    arrays in 16-int32 chunks PAST the end of the [E+1] tensor when (E+1)%16 != 1
    (dots3 E=257 hits this). Garbage tails can bracket a row index and reassign the
    row to a phantom expert, wild-writing its blockscale -> sporadic wrong outputs
    and memory corruption. Zero tails never match."""
    buf = torch.zeros(numel + 16, dtype=torch.int32, device='cuda')
    return buf[:numel]


def build_group_weights(n, k, E=GROUP_E, seed=0, keep_hp=False):
    """-> (w_fp4 [E,n,k/2], w_blockscale [E,n,k/16], w_gscale scalar[, w_hp]).
    ONE per-tensor global scale over all experts (equal-alpha semantics)."""
    from sgl_kernel import scaled_fp4_quant

    torch.manual_seed(seed)
    w_hp = torch.randn(E, n, k, dtype=torch.bfloat16, device='cuda')
    w_gs = (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / w_hp.abs().amax().float()
    w_fp4 = torch.empty(E, n, k // 2, dtype=torch.uint8, device='cuda')
    w_bs = torch.empty(E, n, k // 16, dtype=torch.float8_e4m3fn, device='cuda')
    for e in range(E):
        qd, sf = scaled_fp4_quant(w_hp[e], w_gs)
        w_fp4[e] = qd
        w_bs[e] = sf   # n is a multiple of 128 and k/16 of 4 -> no padding
    if keep_hp:
        return w_fp4, w_bs, w_gs, w_hp
    del w_hp
    torch.cuda.empty_cache()
    return w_fp4, w_bs, w_gs


def _experts_quant_zeroed(input_tensor, input_global_scale, expert_offsets,
                          blockscale_offsets, topk, expert_map):
    """sgl_kernel.scaled_fp4_experts_quant with a ZERO-initialized scale buffer.

    The stock wrapper allocates the blockscale output with torch.empty; the 128-row
    padding regions between experts are never written by the quant kernel, so their
    content depends on allocator history (fresh driver pages are zeroed, reused pages
    are dirty). Runs with dirty padding showed sporadic cutlass-vs-cublasLt output
    mismatches and one illegal-memory-access on GB200, so the benchmark pins the
    padding to zero to keep both backends deterministic."""
    from sgl_kernel import shuffle_rows

    assert input_tensor.ndim == 2
    m, k = input_tensor.shape
    input_tensor = shuffle_rows(input_tensor, expert_map, (m * topk, k))
    m_numtopk = m * topk
    MAX_TOKENS_PER_EXPERT = int(os.environ.get('MODELOPT_MAX_TOKENS_PER_EXPERT', 65536))
    assert m_numtopk <= MAX_TOKENS_PER_EXPERT * topk
    scales_k = k // 16
    padded_k = (scales_k + 3) // 4
    output = torch.empty(m_numtopk, k // 2, device=input_tensor.device, dtype=torch.uint8)
    output_scales = torch.zeros(MAX_TOKENS_PER_EXPERT * topk, padded_k,
                                dtype=torch.int32, device=input_tensor.device)
    torch.ops.sgl_kernel.scaled_fp4_experts_quant.default(
        output, output_scales, input_tensor, input_global_scale,
        expert_offsets, blockscale_offsets)
    return output, output_scales.view(torch.float8_e4m3fn)


def build_group_point(w_fp4, w_bs, w_gs, T, n, k, E=GROUP_E, topk=GROUP_TOPK, seed=0,
                      w_hp=None):
    """Routing: each token picks 8 DISTINCT experts out of the 256 routed ones plus the
    shared expert (id 256) -> sum_m = T*topk, shared expert has m=T.
    -> (run_ct, run_lt, flops, nbytes, sum_m[, ref]) -- ref is the fp32 grouped-layout
    reference (computed from the bf16 sources) when w_hp is given."""
    import cublaslt_group_mm as cm
    from sgl_kernel import cutlass_fp4_group_mm, prepare_moe_input

    dev = 'cuda'
    torch.manual_seed(seed)
    routed = torch.rand(T, E - 1, device=dev).topk(topk - 1, dim=1).indices.to(torch.int32)
    shared = torch.full((T, 1), E - 1, dtype=torch.int32, device=dev)
    topk_ids = torch.cat([routed, shared], dim=1).contiguous()
    del routed, shared
    sum_m = T * topk

    expert_offsets = _padded_i32(E + 1)       # zero-padded tails: see _padded_i32
    blockscale_offsets = _padded_i32(E + 1)
    problem_sizes1 = torch.empty(E, 3, dtype=torch.int32, device=dev)
    problem_sizes2 = torch.empty(E, 3, dtype=torch.int32, device=dev)
    a_map = torch.empty(sum_m, dtype=torch.int32, device=dev)
    c_map = torch.empty(sum_m, dtype=torch.int32, device=dev)
    prepare_moe_input(topk_ids, expert_offsets, problem_sizes1, problem_sizes2,
                      a_map, c_map, E, n // 2, k, blockscale_offsets)

    a_hp = torch.randn(T, k, dtype=torch.bfloat16, device=dev)
    a_gs = ((FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / a_hp.abs().amax().float()).repeat(E)
    rep_a_fp4, rep_a_bs = _experts_quant_zeroed(
        a_hp, a_gs, expert_offsets, blockscale_offsets, topk, expert_map=a_map)
    alphas = (1.0 / (a_gs * w_gs)).float()

    ref = None
    if w_hp is not None:
        rep_a_hp = a_hp[a_map.long()]                     # grouped-layout bf16 tokens
        eo_h = expert_offsets.cpu().tolist()
        ref = torch.empty(sum_m, n, dtype=torch.float32, device=dev)
        for e in range(E):
            lo, hi = eo_h[e], eo_h[e + 1]
            if hi > lo:
                ref[lo:hi] = rep_a_hp[lo:hi].float() @ w_hp[e].float().t()
        del rep_a_hp
    del a_hp

    params = {
        'ab_strides': torch.full((E,), k, dtype=torch.int64, device=dev),
        'c_strides': torch.full((E,), n, dtype=torch.int64, device=dev),
        'problem_sizes': problem_sizes1,
        'expert_offsets': expert_offsets[:-1],
        'blockscale_offsets': blockscale_offsets[:-1],
    }

    run_ct = lambda: cutlass_fp4_group_mm(
        rep_a_fp4, w_fp4, rep_a_bs, w_bs, alphas, torch.bfloat16, dev, params)
    run_lt = lambda: cm.cublaslt_fp4_group_mm(
        rep_a_fp4, w_fp4, rep_a_bs, w_bs, alphas, torch.bfloat16, dev, params)

    flops = 2.0 * sum_m * n * k
    sf_rows = int(blockscale_offsets[E].item())  # 128-padded scale rows actually read
    # weights: count only ACTIVE experts (m_e > 0) -- both kernels skip m=0 groups, and at
    # small T most of the 257 experts are inactive (full-weight counting would exceed HBM BW)
    n_active = int((problem_sizes1[:, 0] > 0).sum().item())
    nbytes = (sum_m * (k // 2)                          # A fp4
              + sf_rows * (k // 16)                     # A blockscales
              + n_active * n * (k // 2)                 # B fp4 (active experts)
              + n_active * n * (k // 16)                # B blockscales (active experts)
              + sum_m * n * 2)                          # D bf16
    if w_hp is not None:
        return run_ct, run_lt, flops, nbytes, sum_m, ref
    return run_ct, run_lt, flops, nbytes, sum_m


def reset_cublaslt_plans():
    """Drop cached cublasLt grouped plans so the next call re-runs the heuristic with the
    current point's sum_m hint (plans are keyed by weight ptr and would otherwise be
    reused across the whole T sweep with the first T's hint). Also destroys the C++-side
    plan objects and zeroes the shared workspace so no state leaks between points."""
    import cublaslt_group_mm as cm
    pkg = cm._pkg
    if pkg._plans:
        ext = cm.load_ext()
        for p in pkg._plans.values():
            try:
                ext.destroy_plan(p.pid)
            except Exception:
                pass
        pkg._plans.clear()
    if pkg._workspace is not None:
        pkg._workspace.zero_()


def cublaslt_live_plan():
    """(ext, pid, n_algos) for the single live cublasLt plan, or (None, 0, 0).

    The heuristic returns its algos ranked by an internal cost model; algo[0] is its
    prediction, not a measurement. The bench measures EVERY returned algo with the
    standard bench_kineto metric and keeps the fastest (event-timed autotune of the
    full call is CPU-launch-bound at small T and picks wrongly there)."""
    import cublaslt_group_mm as cm
    plans = list(cm._pkg._plans.values())
    if not plans:
        return None, 0, 0
    ext = cm.load_ext()
    pid = plans[0].pid
    return ext, pid, int(ext.num_algos(pid))


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
