"""Quantization helpers for the support probes (no torchao in this image).

- NVFP4: use sgl_kernel.scaled_fp4_quant (per-16 e4m3 block scales, swizzled 32x4x4).
- MXFP8: manual e4m3 data + per-32 e8m0 scales, plus torchao-style to_blocked swizzle.
"""
import torch

FLOAT8_E4M3_MAX = 448.0
FLOAT4_E2M1_MAX = 6.0


def ceil_div(a, b):
    return (a + b - 1) // b


def to_blocked(scales: torch.Tensor) -> torch.Tensor:
    """torchao's to_blocked: [R, S] scale matrix -> flat swizzled (32x4x4) layout,
    rows padded to 128, cols padded to 4. Layout required by cuBLASLt/cutlass
    block-scaled GEMMs and torch._scaled_mm_v2 with SWIZZLE_32_4_4."""
    rows, cols = scales.shape
    n_row_blocks = ceil_div(rows, 128)
    n_col_blocks = ceil_div(cols, 4)
    padded = scales
    if rows != n_row_blocks * 128 or cols != n_col_blocks * 4:
        padded = torch.zeros(
            n_row_blocks * 128, n_col_blocks * 4, dtype=scales.dtype, device=scales.device
        )
        padded[:rows, :cols] = scales
    blocks = padded.view(n_row_blocks, 4, 32, n_col_blocks, 4).permute(0, 3, 2, 1, 4)
    return blocks.reshape(-1, 32, 16).flatten()


def from_blocked(flat: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """Inverse of to_blocked: flat swizzled (32x4x4) buffer -> [rows, cols] scales.

    `rows`/`cols` are the PADDED extents actually present in the buffer (rows a
    multiple of 128, cols of 4); slice the result to the logical size afterwards.
    Used to hand CUTLASS the un-swizzled scales it re-orders itself, while cuBLASLt
    keeps the swizzled ones -- both then provably derive from the SAME numbers.
    """
    n_row_blocks = ceil_div(rows, 128)
    n_col_blocks = ceil_div(cols, 4)
    blocks = flat.reshape(n_row_blocks, n_col_blocks, 32, 4, 4)
    padded = blocks.permute(0, 3, 2, 1, 4).reshape(n_row_blocks * 128, n_col_blocks * 4)
    return padded[:rows, :cols]


def quant_mxfp8(x: torch.Tensor):
    """x [R, C] bf16 (C % 32 == 0) -> (e4m3 data [R, C], e8m0 scales [R, C//32] unswizzled)."""
    R, C = x.shape
    assert C % 32 == 0
    xb = x.float().view(R, C // 32, 32)
    amax = xb.abs().amax(dim=-1).clamp(min=2**-127)
    # e8m0 scale: power of two >= amax / 448 (RCP-style, matches OCP MX spec intent)
    exp = torch.ceil(torch.log2(amax / FLOAT8_E4M3_MAX)).clamp(-127, 127)
    scale = torch.pow(2.0, exp)
    q = (xb / scale.unsqueeze(-1)).clamp(-FLOAT8_E4M3_MAX, FLOAT8_E4M3_MAX)
    data = q.to(torch.float8_e4m3fn).view(R, C)
    e8m0 = scale.to(torch.float8_e8m0fnu)
    return data, e8m0


def dequant_mxfp8(data, e8m0):
    R, C = data.shape
    s = e8m0.float().unsqueeze(-1)
    return (data.float().view(R, C // 32, 32) * s).view(R, C)


def nvfp4_quant(x: torch.Tensor):
    """x [R, C] bf16 -> (packed fp4 [R, C//2] uint8, swizzled e4m3 scales, global_scale fp32).
    Uses sgl_kernel.scaled_fp4_quant. Dequant scale for the gemm: alpha = 1/(gs_a*gs_b)."""
    from sgl_kernel import scaled_fp4_quant

    gs = (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / x.abs().amax().float()
    data, sf = scaled_fp4_quant(x, gs)
    return data, sf, gs
