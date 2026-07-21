"""Does cuBLASLt support a PER-BATCH nvfp4 global scale (not just per-tensor)?

nvfp4 has two scale levels; only the second is in question here:
  * per-16 block scale (e4m3 tensor, VEC16_UE4M3) -- already shown per-batch OK
  * per-TENSOR global scale (fp32 scalar) -- cublasLtMatmul takes ONE host scalar,
    so all batches must share it. This probe tests the two documented routes to a
    per-BATCH global scale that do NOT disturb the block scales:
      mode 1: POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_HOST + ALPHA_VECTOR_BATCH_STRIDE
      mode 2: D_SCALE_MODE = PER_BATCH_SCALAR_32F + D_SCALE_POINTER
  (A/B_SCALE_MODE = PER_BATCH_SCALAR_32F is unusable: one mode per operand and
   nvfp4 needs VEC16_UE4M3 there.)

Discriminating design: each batch is quantized with its OWN global scale, and the
batches are given DELIBERATELY different magnitudes (batch b scaled by 4**b), so the
alphas differ by orders of magnitude. Then mode 0 (one shared alpha) MUST be visibly
wrong -- that is the control proving the test can detect the difference at all.

Run: /opt/venvs/sglang-dev/bin/python probe_per_batch_scale.py
"""
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from quant_utils import FLOAT4_E2M1_MAX, FLOAT8_E4M3_MAX  # noqa: E402
import cublaslt_batched_nvfp4 as cb  # noqa: E402

B, M, N, K = 8, 128, 512, 128
DEV = 'cuda'


def build():
    """per-batch global scales that differ by orders of magnitude."""
    from sgl_kernel import scaled_fp4_quant
    torch.manual_seed(0)
    x_hp = torch.randn(B, M, K, dtype=torch.bfloat16, device=DEV)
    w_hp = torch.randn(B, N, K, dtype=torch.bfloat16, device=DEV)
    for b in range(B):                      # spread magnitudes -> spread alphas
        x_hp[b] *= float(4 ** b)
    xd, xs, wd, ws, alphas = [], [], [], [], []
    for b in range(B):
        gx = (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / x_hp[b].abs().amax().float()
        gw = (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / w_hp[b].abs().amax().float()
        d, s = scaled_fp4_quant(x_hp[b].contiguous(), gx); xd.append(d); xs.append(s)
        d, s = scaled_fp4_quant(w_hp[b].contiguous(), gw); wd.append(d); ws.append(s)
        alphas.append((1.0 / (gx * gw)).item())
    return (x_hp, w_hp,
            torch.stack(xd).contiguous(), torch.stack(xs).contiguous(),
            torch.stack(wd).contiguous(), torch.stack(ws).contiguous(),
            torch.tensor(alphas, dtype=torch.float32, device=DEV))


def run(mode, x_fp4, x_sf, w_fp4, w_sf, alphas, invert=False):
    out = torch.empty(B, M, N, dtype=torch.bfloat16, device=DEV)
    a = (1.0 / alphas) if invert else alphas
    if mode == 0:
        av, scalar = None, float(alphas[0].item())      # shared alpha (control)
    elif mode == 1:
        av, scalar = a.view(B, 1).expand(B, N).contiguous(), 1.0
    else:
        av, scalar = a.contiguous(), 1.0
    try:
        ext, pid = cb.make_plan(w_fp4, x_fp4, out, w_sf, x_sf, B, M, N, K,
                                scalar, av, mode)
    except Exception as e:
        return None, f'PLAN-FAIL {type(e).__name__}: {str(e)[:140]}'
    try:
        ext.run_plan(pid)
        torch.cuda.synchronize()
    except Exception as e:
        ext.destroy_plan(pid)
        return None, f'RUN-FAIL {type(e).__name__}: {str(e)[:140]}'
    ext.destroy_plan(pid)
    return out, None


if __name__ == '__main__':
    print(f'device: {torch.cuda.get_device_name(0)}  cublasLt {cb.load_ext().cublaslt_version()}')
    x_hp, w_hp, x_fp4, x_sf, w_fp4, w_sf, alphas = build()
    ref = torch.bmm(x_hp.float(), w_hp.float().transpose(1, 2))
    print(f'per-batch alphas span {alphas.min():.3e} .. {alphas.max():.3e} '
          f'(ratio {alphas.max()/alphas.min():.1f}x)\n')

    labels = {0: 'mode0 shared scalar alpha (CONTROL, must be WRONG)',
              1: 'mode1 ALPHA_VECTOR_BATCH_STRIDE (per-batch alpha)',
              2: 'mode2 D_SCALE PER_BATCH_SCALAR_32F (per-batch)'}
    for mode in (0, 1, 2):
        for invert in ((False,) if mode != 2 else (False, True)):
            out, err = run(mode, x_fp4, x_sf, w_fp4, w_sf, alphas, invert)
            name = labels[mode] + (' [1/alpha]' if invert else '')
            if err:
                print(f'{name}\n    {err}')
                continue
            rel = ((out.float() - ref).norm() / ref.norm()).item()
            per_b = [((out[b].float() - ref[b]).norm() / ref[b].norm()).item()
                     for b in range(B)]
            ok = 'OK  ' if rel < 0.3 else 'BAD '
            print(f'{name}\n    {ok} relerr={rel:.4e} '
                  f'per-batch=[{", ".join(f"{r:.3f}" for r in per_b)}]')
