"""
Diagnostic: compare torch._scaled_mm_v2 NVFP4 16384^3 fp32 against the C++ graph bench.

(1) COLD single-kernel time with NVML clock readout (is the kernel itself the same speed at the
    same clock as the C++ bench's 1.045 ms @ 2062 MHz?).
(2) SUSTAINED >=1.5 s eager replay loop, mirroring bench_smmhz.cu: per-segment CUDA events as
    in-stream timestamps, async NVML sampler, NO host sync inside -> Python's true steady-state
    TFLOPS / sm_mhz / power, comparable apples-to-apples with the C++ steady state.
(3) The bench_kineto number (what the suite CSV reports) for reference.
"""
import os, sys, time, threading
import torch

sys.path.insert(0, '/root/workspace/gb_gemm_benchmark/python')
from bench_common import prep_nvfp4, NUM_TESTS  # noqa
sys.path.insert(0, '/root/workspace/gb_gemm_benchmark/DeepGEMM')
from deep_gemm.testing import bench_kineto  # noqa
import pynvml  # noqa

M = N = K = 16384
KERNEL = 'nvjet_sm100'
FLOP = 2.0 * M * N * K

pynvml.nvmlInit()
H = pynvml.nvmlDeviceGetHandleByIndex(0)
def clk(): return pynvml.nvmlDeviceGetClockInfo(H, pynvml.NVML_CLOCK_SM)
def pw():  return pynvml.nvmlDeviceGetPowerUsage(H) / 1000.0

run = prep_nvfp4(M, N, K, torch.float32)

# ---- (1) cold single-kernel time + clock ----
for _ in range(5): run()
torch.cuda.synchronize()
time.sleep(0.6)  # let clock/power recover to peak (cold)
s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
s.record(); run(); e.record(); torch.cuda.synchronize()
cold_ms = s.elapsed_time(e)
print(f"(1) COLD single kernel: {cold_ms*1000:.1f} us  ({FLOP/(cold_ms/1e3)/1e12:.0f} TFLOPS)  "
      f"clk~{clk()} MHz  (C++ cold ref: ~1045 us @ 2062 MHz / 8413 TFLOPS)")

# ---- (3) bench_kineto (what the suite reports) ----
sec = bench_kineto(run, KERNEL, num_tests=NUM_TESTS, suppress_kineto_output=True, flush_l2=False)
print(f"(3) bench_kineto (flush_l2=False, num_tests={NUM_TESTS}): {sec*1e6:.1f} us  "
      f"({FLOP/sec/1e12:.0f} TFLOPS)  <- this is the suite CSV value")

# ---- (2) sustained >=1.5 s eager loop, per-segment events + async NVML, no mid-run sync ----
samples = []  # (t, mhz, pw)
stop = False
def sampler():
    while not stop:
        samples.append((time.time(), clk(), pw()))
        time.sleep(0.001)
th = threading.Thread(target=sampler, daemon=True); th.start()

SEG = 20          # kernels per event segment (~20*1.1ms ~ 22 ms)
TARGET_S = 3.0
nseg = int(TARGET_S * 1000 / (SEG * cold_ms)) + 1
evs = [torch.cuda.Event(enable_timing=True) for _ in range(nseg + 1)]

for _ in range(3): run()           # warmup
torch.cuda.synchronize()
time.sleep(0.6)                    # idle so we capture the transient from peak

t0 = time.time()
evs[0].record()
for sgi in range(nseg):
    for _ in range(SEG):
        run()
    evs[sgi + 1].record()
torch.cuda.synchronize()           # the ONLY sync
t1 = time.time()

gpu_ms = evs[0].elapsed_time(evs[nseg])
host_ms = (t1 - t0) * 1000
print(f"(2) SUSTAINED eager: GPU continuous = {gpu_ms:.0f} ms ; host wall = {host_ms:.0f} ms ; "
      f"drift = {100*(host_ms-gpu_ms)/gpu_ms:.2f}%")

# per-segment table + steady-state (last 1 s)
def med(lst): lst=sorted(lst); return lst[len(lst)//2] if lst else float('nan')
cum = 0.0
out = open('result/diag_sustained_16384.csv', 'w')
out.write('seg,cum_run_ms,seg_ms,kernel_us,tflops,sm_mhz,power_w\n')
tail_tf, tail_sm, tail_pw = [], [], []
rows_print = []
for sgi in range(nseg):
    sms = evs[sgi].elapsed_time(evs[sgi + 1])
    h0, h1 = t0 + cum/1000, t0 + (cum+sms)/1000
    win = [(c, p) for (t, c, p) in samples if h0 <= t <= h1]
    sm = med([c for c, _ in win]); pwr = med([p for _, p in win])
    kus = sms*1000/SEG
    tf = SEG*FLOP/(sms/1e3)/1e12
    cum += sms
    out.write(f"{sgi},{cum:.1f},{sms:.3f},{kus:.1f},{tf:.1f},{sm},{pwr}\n")
    rows_print.append((cum, sm, pwr, tf))
    if cum >= gpu_ms - 1000:
        tail_tf.append(tf); tail_sm.append(sm); tail_pw.append(pwr)
out.close()
# print coarse trajectory
last = -1
print("    cum_s   sm   pw    tflops")
for cum, sm, pwr, tf in rows_print:
    if cum/1000 - last >= 0.3:
        print(f"   {cum/1000:5.2f}  {sm:>4}  {pwr:>4.0f}  {tf:7.0f}"); last = cum/1000
print(f"(2) STEADY-STATE (last 1s): tflops~{med(tail_tf):.0f}  sm~{med(tail_sm):.0f}  power~{med(tail_pw):.0f}")

stop = True; th.join(); pynvml.nvmlShutdown()
print("-> result/diag_sustained_16384.csv")
