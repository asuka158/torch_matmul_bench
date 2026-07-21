"""Support matrix: both nvfp4 batched backends on IDENTICAL quantized inputs.

Quantization is done ONCE with sgl_kernel.scaled_fp4_quant (authoritative packed fp4 +
swizzled scales). cuBLASLt consumes the swizzled scales directly; CUTLASS gets the same
numbers un-swizzled via quant_utils.from_blocked (it re-orders them itself). So any
difference between the two results is the kernel, not the data.

Global scale is per-TENSOR (one shared value across all batches) -- neither backend
supports a per-batch global scale for this op (probe_per_batch_scale.py).

Run: /opt/venvs/sglang-dev/bin/python probe_bmm.py
"""
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from quant_utils import FLOAT4_E2M1_MAX, FLOAT8_E4M3_MAX, ceil_div, from_blocked  # noqa: E402

# the four dots3 absorbed bmms (see dots3_bmm_plan.md 1.2)
BMMS = [
    ('A5_w_kc', 128, 128, 512),   # (name, B, K, N)
    ('A7_w_vc', 128, 512, 128),
    ('C5_w_kc', 64, 192, 1024),
    ('C7_w_vc', 64, 1024, 128),
]
DEV = 'cuda'


def quantize(x_hp):
    """[L,R,K] bf16 -> (packed [L,R,K//2] uint8, swizzled sf, unswizzled sf CPU, gs).

    One global scale over the whole tensor (per-TENSOR semantics).
    """
    from sgl_kernel import scaled_fp4_quant
    L, R, K = x_hp.shape
    gs = (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / x_hp.abs().amax().float()
    data, sw, unsw = [], [], []
    for b in range(L):
        d, s = scaled_fp4_quant(x_hp[b].contiguous(), gs)
        data.append(d)
        sw.append(s)
        prows, pcols = s.shape                      # padded extents in the buffer
        unsw.append(from_blocked(s.flatten(), prows, pcols)[:R, :ceil_div(K, 16)])
    return (torch.stack(data).contiguous(), torch.stack(sw).contiguous(),
            torch.stack(unsw).contiguous().cpu(), gs)


def check(name, B, M, K, N):
    torch.manual_seed(0)
    x_hp = torch.randn(B, M, K, dtype=torch.bfloat16, device=DEV)
    w_hp = torch.randn(B, N, K, dtype=torch.bfloat16, device=DEV)
    x_fp4, x_sw, x_unsw, gs_x = quantize(x_hp)
    w_fp4, w_sw, w_unsw, gs_w = quantize(w_hp)
    alpha = float((1.0 / (gs_x * gs_w)).item())
    ref = torch.bmm(x_hp.float(), w_hp.float().transpose(1, 2))   # fp32 baseline

    def rel(out):
        return ((out.float() - ref).norm() / ref.norm()).item()

    res = {}
    # ---- cuBLASLt (swizzled scales, host scalar alpha) ----
    try:
        import cublaslt_batched_nvfp4 as cb
        out = torch.empty(B, M, N, dtype=torch.bfloat16, device=DEV)
        ext, pid = cb.make_plan(w_fp4, x_fp4, out, w_sw, x_sw, B, M, N, K, alpha)
        ext.run_plan(pid)
        torch.cuda.synchronize()
        ext.destroy_plan(pid)
        res['cuBLASLt'] = f'{rel(out):.4e}'
        lt_out = out
    except Exception as e:
        res['cuBLASLt'] = f'FAIL {type(e).__name__}: {str(e)[:70]}'
        lt_out = None
    # ---- CUTLASS (un-swizzled scales, alpha in epilogue) ----
    try:
        from cutlass_batched_nvfp4 import CutlassBatchedNvfp4
        out2 = torch.empty(B, M, N, dtype=torch.bfloat16, device=DEV)
        k = CutlassBatchedNvfp4(M, N, K, B, alpha).bind(
            x_fp4, w_fp4, x_unsw, w_unsw, out2)
        k()
        torch.cuda.synchronize()
        res['CUTLASS'] = f'{rel(out2):.4e}'
        if lt_out is not None:
            d = (out2.float() - lt_out.float()).abs()
            res['ct-vs-lt'] = f'max_abs={d.max().item():.3e} rel={rel(out2)/max(rel(lt_out),1e-30):.3f}'
    except Exception as e:
        res['CUTLASS'] = f'FAIL {type(e).__name__}: {str(e)[:70]}'
    return res


if __name__ == '__main__':
    print(f'device: {torch.cuda.get_device_name(0)}')
    print('relerr vs fp32 baseline (nvfp4 expected ~1.3e-1)\n')
    Ms = [int(v) for v in (sys.argv[1:] or ['1', '16', '128', '1024'])]
    for (name, B, K, N) in BMMS:
        for M in Ms:
            r = check(name, B, M, K, N)
            print(f'{name:9} B={B:4} M={M:6} K={K:5} N={N:5} | '
                  + ' | '.join(f'{k}={v}' for k, v in r.items()), flush=True)
