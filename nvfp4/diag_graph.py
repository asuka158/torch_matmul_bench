"""Isolate eager-vs-graph: time the SAME torch._scaled_mm_v2 NVFP4 16384^3 kernel three ways
at full clock (cold, short bursts so no throttle), to see if a CUDA graph closes the gap to the
C++ graph bench (1045 us @ 2062 MHz)."""
import os, sys, time
import torch
sys.path.insert(0, '/root/workspace/gb_gemm_benchmark/python')
from bench_common import prep_nvfp4  # noqa
import pynvml
pynvml.nvmlInit(); H = pynvml.nvmlDeviceGetHandleByIndex(0)
def clk(): return pynvml.nvmlDeviceGetClockInfo(H, pynvml.NVML_CLOCK_SM)

M=N=K=16384; FLOP=2.0*M*N*K
def tf(ms): return FLOP/(ms/1e3)/1e12

run = prep_nvfp4(M, N, K, torch.float32)
for _ in range(5): run()
torch.cuda.synchronize()

# --- A) eager: N back-to-back, one event pair, /N (short burst -> full clock) ---
def eager(n):
    time.sleep(0.4)
    s,e=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize(); c0=clk()
    s.record()
    for _ in range(n): run()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e)/n, c0
for n in (1, 10, 30):
    ms,c=eager(n); print(f"A) eager x{n:<3}: {ms*1000:7.1f} us/kernel  {tf(ms):6.0f} TFLOPS  clk@start~{c}")

# --- B) CUDA graph: capture G calls, replay once, one event pair, /G ---
G=10
s_ = torch.cuda.Stream()
s_.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s_):
    for _ in range(3): run()
torch.cuda.current_stream().wait_stream(s_)
torch.cuda.synchronize()
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    for _ in range(G): run()
# warmup replays
for _ in range(2): g.replay()
torch.cuda.synchronize()
time.sleep(0.4)
s,e=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
torch.cuda.synchronize(); c0=clk()
s.record(); g.replay(); e.record(); torch.cuda.synchronize()
ms=s.elapsed_time(e)/G
print(f"B) graph  x{G:<3}: {ms*1000:7.1f} us/kernel  {tf(ms):6.0f} TFLOPS  clk@start~{c0}  (C++ graph ref: 1045 us / 8413)")

# --- C) graph replayed many times back-to-back, short (~200ms, still ~full clock) ---
torch.cuda.synchronize(); time.sleep(0.4)
s,e=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
c0=clk(); s.record()
R=15
for _ in range(R): g.replay()
e.record(); torch.cuda.synchronize()
ms=s.elapsed_time(e)/(G*R)
print(f"C) graph x{G*R} ({R} replays): {ms*1000:7.1f} us/kernel  {tf(ms):6.0f} TFLOPS  clk@start~{c0}")
pynvml.nvmlShutdown()
