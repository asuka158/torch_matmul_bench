"""Does cuBLASLt support STRIDED-BATCHED nvfp4 (VEC16_UE4M3) matmul at all?

cublasLt exposes no per-batch stride attribute for block-scale tensors, so this is
undocumented and has to be settled empirically. Checks, for a few (B,M,N,K):
  * does create_plan / the heuristic even accept a batched block-scaled fp4 problem
  * is the result numerically right vs an fp32 reference (relerr), i.e. does cublasLt
    stride the SCALE buffers per batch the way the data buffers are strided

Run: /opt/venvs/sglang-dev/bin/python probe_cublaslt_batched.py
"""
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from quant_utils import FLOAT4_E2M1_MAX, FLOAT8_E4M3_MAX  # noqa: E402
import cublaslt_batched_nvfp4 as cb  # noqa: E402


def quant_batched(x, gs):
    """x [B,R,C] bf16 + shared global scale -> ([B,R,C//2] uint8, stacked swizzled sf)."""
    from sgl_kernel import scaled_fp4_quant
    data, sf = [], []
    for b in range(x.shape[0]):
        d, s = scaled_fp4_quant(x[b].contiguous(), gs)
        data.append(d)
        sf.append(s)
    return torch.stack(data).contiguous(), torch.stack(sf).contiguous()


def run_case(B, M, N, K, seed=0):
    dev = 'cuda'
    torch.manual_seed(seed)
    x_hp = torch.randn(B, M, K, dtype=torch.bfloat16, device=dev)
    w_hp = torch.randn(B, N, K, dtype=torch.bfloat16, device=dev)

    # ONE global scale per operand across ALL batches -> single scalar alpha
    gs_x = (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / x_hp.abs().amax().float()
    gs_w = (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / w_hp.abs().amax().float()
    x_fp4, x_sf = quant_batched(x_hp, gs_x)
    w_fp4, w_sf = quant_batched(w_hp, gs_w)
    alpha = float((1.0 / (gs_x * gs_w)).item())

    out = torch.empty(B, M, N, dtype=torch.bfloat16, device=dev)
    try:
        ext, pid = cb.make_plan(w_fp4, x_fp4, out, w_sf, x_sf, B, M, N, K, alpha)
    except Exception as e:
        return f'PLAN-FAIL {type(e).__name__}: {str(e)[:160]}'
    try:
        ext.run_plan(pid)
        torch.cuda.synchronize()
    except Exception as e:
        ext.destroy_plan(pid)
        return f'RUN-FAIL {type(e).__name__}: {str(e)[:160]}'
    n_algos = ext.num_algos(pid)
    ext.destroy_plan(pid)

    ref = torch.bmm(x_hp.float(), w_hp.float().transpose(1, 2))
    rel = ((out.float() - ref).norm() / ref.norm()).item()
    per_b = [((out[b].float() - ref[b]).norm() / ref[b].norm()).item() for b in range(B)]
    tag = 'OK  ' if rel < 0.3 else 'BAD '
    return (f'{tag} relerr={rel:.4e} algos={n_algos} '
            f'per-batch=[{", ".join(f"{r:.3f}" for r in per_b[:6])}'
            f'{", ..." if B > 6 else ""}]')


def sf_shape(M, K):
    """what sgl_kernel.scaled_fp4_quant returns for a [M,K] operand (row padding?)"""
    from sgl_kernel import scaled_fp4_quant
    x = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
    _, s = scaled_fp4_quant(x, torch.tensor(1.0, device='cuda'))
    return tuple(s.shape)


if __name__ == '__main__':
    print(f'device: {torch.cuda.get_device_name(0)}')
    print(f'cublasLt: {cb.load_ext().cublaslt_version()}\n')
    # B=1 first (degenerates to the known-good 2D case), then real batching.
    cases = [
        (1, 128, 512, 128),    # B=1 sanity
        (2, 128, 512, 128),
        (4, 128, 512, 128),
        (128, 128, 512, 128),  # A5 w_kc geometry, M=128 (no scale row padding)
        (64, 128, 1024, 192),  # C5 w_kc geometry, K=192
    ]
    for (B, M, N, K) in cases:
        res = run_case(B, M, N, K)
        print(f'B={B:4} M={M:5} N={N:5} K={K:5} | {res}', flush=True)

    # --- M NOT a multiple of 128: the decode regime, where the activation scale
    #     buffer needs 128-row tile padding per batch (same ambiguity as the group
    #     A-scale layout). Establish empirically which M work.
    print('\n=== M sweep, A5 geometry (B=128, N=512, K=128) ===')
    for M in [1, 2, 4, 8, 16, 32, 64, 128, 256, 1024]:
        shp = sf_shape(M, 128)
        res = run_case(128, M, 512, 128)
        print(f'M={M:6} sf_shape={str(shp):14} | {res}', flush=True)
