"""Does driving the SAME torch._scaled_mm_v2 kernel from a CUDA graph (minimal host activity)
let the SM sustain a higher clock than the eager Python loop? Sustained >=1.5 s, NVML-sampled,
to compare steady-state clock against eager-Python (~1192 MHz) and C++ graph (~1582 MHz)."""
import sys, time, threading
import torch
sys.path.insert(0, '/root/workspace/gb_gemm_benchmark/python')
from bench_common import prep_nvfp4
import pynvml
pynvml.nvmlInit(); H = pynvml.nvmlDeviceGetHandleByIndex(0)
def clk(): return pynvml.nvmlDeviceGetClockInfo(H, pynvml.NVML_CLOCK_SM)
def pw():  return pynvml.nvmlDeviceGetPowerUsage(H)/1000.0
M=N=K=16384; FLOP=2.0*M*N*K

run = prep_nvfp4(M, N, K, torch.float32)
G = 20
s_ = torch.cuda.Stream(); s_.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s_):
    for _ in range(3): run()
torch.cuda.current_stream().wait_stream(s_); torch.cuda.synchronize()
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    for _ in range(G): run()

samples=[]; stop=False
def smp():
    while not stop:
        samples.append((time.time(), clk(), pw())); time.sleep(0.001)
th=threading.Thread(target=smp, daemon=True); th.start()

for _ in range(2): g.replay()
torch.cuda.synchronize(); time.sleep(0.6)

# sustained: ~3 s of back-to-back graph replays, per-segment events, single final sync
NREP = int(3.0*1000/(G*1.1))+1
evs=[torch.cuda.Event(enable_timing=True) for _ in range(NREP+1)]
t0=time.time(); evs[0].record()
for i in range(NREP):
    g.replay(); evs[i+1].record()
torch.cuda.synchronize(); t1=time.time()
gpu_ms=evs[0].elapsed_time(evs[NREP])
print(f"GRAPH sustained: GPU={gpu_ms:.0f}ms host={(t1-t0)*1000:.0f}ms drift={100*((t1-t0)*1000-gpu_ms)/gpu_ms:.2f}%")
def med(x): x=sorted(x); return x[len(x)//2] if x else float('nan')
cum=0; tail=[]
print("  cum_s   sm   pw   tflops")
last=-1
for i in range(NREP):
    sms=evs[i].elapsed_time(evs[i+1]); h0=t0+cum/1000; h1=t0+(cum+sms)/1000
    win=[(c,p) for (t,c,p) in samples if h0<=t<=h1]
    sm=med([c for c,_ in win]); pwr=med([p for _,p in win]); tf=G*FLOP/(sms/1e3)/1e12
    cum+=sms
    if cum/1000-last>=0.3:
        print(f"  {cum/1000:5.2f} {sm:>4} {pwr:>5.0f} {tf:7.0f}"); last=cum/1000
    if cum>=gpu_ms-1000: tail.append((sm,pwr,tf))
print(f"STEADY graph-Python: sm~{med([s for s,_,_ in tail]):.0f} pw~{med([p for _,p,_ in tail]):.0f} "
      f"tflops~{med([t for _,_,t in tail]):.0f}   (eager-Py ~1192/5562 ; C++ ~1582/7250)")
stop=True; th.join(); pynvml.nvmlShutdown()
