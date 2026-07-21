"""One-time CUTLASS tile/cluster config measurement for the batched nvfp4 bmm.

Why measure instead of reusing flashinfer's tactic table: that selector
(_select_sm100_mm_fp4_cute_dsl_tactic) scores candidates on 2D wave quantization --
"how well total_ctas fills the SMs" -- and manufactures CTAs at small M via swap_ab.
Our problem is BATCHED: total CTAs are L x that, so at M=1 with L=128 there is already
plenty of parallelism and its small-M reasoning does not transfer. So we take the
CANDIDATE TILE SET from CUTLASS/flashinfer (_SM100_MMA_TILER_MN_CANDIDATES, i.e. the
kernel author's legitimate configs, not invented ones) and measure them on the ACTUAL
dots3 bmm shapes, then fix a documented rule -- same approach as the mxfp8 dense
2SM/1SM decision.

Run: /opt/venvs/sglang-dev/bin/python probe_cutlass_cfg.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'benchmark'))

import torch  # noqa: E402

from bmm_lib import BMMS, SUBSTR, quantize  # noqa: E402
from dg_bench import bench_kineto  # noqa: E402

# subset of CUTLASS's own candidates that make sense for these N values
CANDIDATES = [((128, 64), (1, 1)), ((128, 128), (1, 1)), ((128, 256), (1, 1)),
              ((256, 128), (2, 1)), ((256, 256), (2, 1)),
              ((128, 128), (1, 2)), ((128, 64), (1, 4))]
MS = [1, 8, 64, 256, 2048, 16384]


def time_cfg(B, M, K, N, tile, cluster):
    from cutlass_batched_nvfp4 import CutlassBatchedNvfp4
    torch.manual_seed(0)
    x_hp = torch.randn(B, M, K, dtype=torch.bfloat16, device='cuda')
    w_hp = torch.randn(B, N, K, dtype=torch.bfloat16, device='cuda')
    x_fp4, _, x_unsw, gs_x = quantize(x_hp)
    w_fp4, _, w_unsw, gs_w = quantize(w_hp)
    alpha = float((1.0 / (gs_x * gs_w)).item())
    del x_hp, w_hp
    out = torch.empty(B, M, N, dtype=torch.bfloat16, device='cuda')
    try:
        k = CutlassBatchedNvfp4(M, N, K, B, alpha, tile, cluster).bind(
            x_fp4, w_fp4, x_unsw, w_unsw, out)
    except Exception as e:
        return None, f'{type(e).__name__}: {str(e)[:60]}'
    try:
        sec = bench_kineto(lambda: k(), SUBSTR['CUTLASS'], num_tests=10,
                           suppress_kineto_output=True, flush_l2=True)
    except Exception as e:
        return None, f'{type(e).__name__}: {str(e)[:60]}'
    finally:
        del k, x_fp4, w_fp4, out
        torch.cuda.empty_cache()
    return sec * 1e6, None


if __name__ == '__main__':
    print(f'device: {torch.cuda.get_device_name(0)}')
    print('CUTLASS tile/cluster sweep, us (lower better)\n')
    hdr = 'bmm       M      ' + ' '.join(f'{str(t)+str(c):>18}' for t, c in CANDIDATES)
    print(hdr)
    best_of = {}
    for bmm, (B, K, N) in BMMS.items():
        for M in MS:
            cells, best, bestcfg = [], float('inf'), None
            for tile, cluster in CANDIDATES:
                us, err = time_cfg(B, M, K, N, tile, cluster)
                if us is None:
                    cells.append(f'{"x":>18}')
                else:
                    cells.append(f'{us:>18.2f}')
                    if us < best:
                        best, bestcfg = us, (tile, cluster)
            best_of[(bmm, M)] = (bestcfg, best)
            print(f'{bmm:9} {M:6} ' + ' '.join(cells) + f'   <- best {bestcfg} {best:.2f}us',
                  flush=True)
    print('\n=== best config per (bmm, M) ===')
    for (bmm, M), (cfg, us) in best_of.items():
        print(f'  {bmm:9} M={M:6} -> {cfg}  {us:.2f}us')
