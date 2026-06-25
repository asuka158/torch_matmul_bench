#!/usr/bin/env python3
"""GB200 FP4 dense-GEMM roofline plotter.

把多个 GEMM shape 画到同一张 log-log roofline 图上(ncu 原生只能一次一个 kernel)。

两个坐标:
  x = arithmetic intensity (FLOP/byte) —— 解析算,不用跑 kernel
  y = 达到的吞吐 (TFLOPS)              —— 需要实测时间;不给就把点放在屋顶(理论上界)

用法:
  # 1) 只看 shape 落在屋顶哪一段(compute / memory bound),纯解析:
  python roofline_plot.py

  # 2) 叠加实测点:准备 CSV,表头 M,N,K,time_us(time_us = 单次 kernel 设备时间)
  python roofline_plot.py --measured measured.csv

  # 3) 用 ncu 实测的真实 DRAM 字节算 AI(而不是 compulsory),CSV 再加一列 dram_bytes
  python roofline_plot.py --measured measured.csv   # 有 dram_bytes 列就自动用它

硬件常数默认 = GB200 单 GPU FP4 dense,务必按自己实测的"可达峰值"改 --peak。
"""
import argparse
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- 默认硬件常数:GB200 单 GPU,FP4 dense ----
DEFAULT_PEAK_TFLOPS = 10000.0   # FP4 dense 峰值 ~10 PFLOPS(请按实测可达峰值改)
DEFAULT_BW_TBS      = 8.0       # HBM3e ~8 TB/s

# ---- test_fp4.py 的 shape ----
M_LIST  = [128, 4096]
NK_LIST = [(2112, 7168), (576, 7168), (24576, 1536), (32768, 512),
           (7168, 16384), (4096, 7168), (7168, 2048)]

FP4_BYTES = 0.5     # 4-bit operand = 0.5 byte/元素
OUT_BYTES = 4.0     # D = torch.float (test_fp4 用的 out_dtype)


def compulsory_bytes(M, N, K, scale_block=0):
    """读一次的强制流量:A + B(fp4) + D(fp32)[+ 可选 E8M0 block scale]。"""
    b = FP4_BYTES * M * K + FP4_BYTES * N * K + OUT_BYTES * M * N
    if scale_block:                       # 每 scale_block 个元素一个 1-byte E8M0
        b += (M * K + N * K) / scale_block
    return b


def flops(M, N, K):
    return 2.0 * M * N * K


def roof(ai, peak_flops, bw):
    """屋顶 = min(算力天花板, 带宽斜坡)。ai: FLOP/byte。返回 FLOP/s。"""
    return np.minimum(peak_flops, ai * bw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--peak", type=float, default=DEFAULT_PEAK_TFLOPS,
                    help="算力峰值 TFLOPS (default GB200 FP4 dense ~10000)")
    ap.add_argument("--bw", type=float, default=DEFAULT_BW_TBS,
                    help="HBM 带宽 TB/s (default ~8)")
    ap.add_argument("--scale-block", type=int, default=0,
                    help="microscaling block 大小(如 32);0=忽略 scale 流量")
    ap.add_argument("--measured", type=str, default=None,
                    help="CSV: M,N,K,time_us[,dram_bytes]")
    ap.add_argument("--out", type=str, default="roofline.png")
    args = ap.parse_args()

    peak_flops = args.peak * 1e12          # -> FLOP/s
    bw         = args.bw * 1e12            # -> byte/s
    ridge      = peak_flops / bw           # FLOP/byte

    # 读实测 CSV(可选)
    measured = {}
    if args.measured:
        with open(args.measured) as f:
            for row in csv.DictReader(f):
                key = (int(row["M"]), int(row["N"]), int(row["K"]))
                measured[key] = dict(
                    time_us=float(row["time_us"]),
                    dram_bytes=float(row["dram_bytes"]) if row.get("dram_bytes") else None,
                )

    # 组装所有点
    pts = []
    for M in M_LIST:
        for (N, K) in NK_LIST:
            F = flops(M, N, K)
            comp_b = compulsory_bytes(M, N, K, args.scale_block)
            m = measured.get((M, N, K))
            if m and m["dram_bytes"]:
                ai = F / m["dram_bytes"]            # 实测 AI
                ai_kind = "actual"
            else:
                ai = F / comp_b                     # compulsory AI(理论上界)
                ai_kind = "compulsory"
            if m:
                achieved = F / (m["time_us"] * 1e-6)   # FLOP/s
            else:
                achieved = roof(ai, peak_flops, bw)    # 没实测 -> 放屋顶
            pts.append(dict(M=M, N=N, K=K, ai=ai, ai_kind=ai_kind,
                            achieved=achieved, comp_ai=F / comp_b,
                            bound="compute" if (F / comp_b) > ridge else "memory"))

    # ---- 画图 ----
    fig, ax = plt.subplots(figsize=(9, 6.5))
    ai_min = min(p["ai"] for p in pts) / 3
    ai_max = max(p["ai"] for p in pts) * 3
    xs = np.logspace(np.log10(ai_min), np.log10(ai_max), 400)
    ax.plot(xs, roof(xs, peak_flops, bw) / 1e12, "k-", lw=2, label="Roofline (GB200 FP4)")
    ax.axhline(peak_flops / 1e12, ls="--", c="gray", lw=1)
    ax.axvline(ridge, ls=":", c="gray", lw=1)
    ax.text(ridge, ax.get_ylim()[0], f" ridge={ridge:.0f}", rotation=90,
            va="bottom", ha="right", fontsize=8, color="gray")

    colors = {"compute": "tab:red", "memory": "tab:blue"}
    print(f"{'idx':>3} {'M':>5} {'N':>6} {'K':>6} {'AI':>8} {'bound':>7} "
          f"{'achv_TFLOPS':>11}  ({'AI from'})")
    for i, p in enumerate(pts):
        ax.scatter(p["ai"], p["achieved"] / 1e12, s=55,
                   c=colors[p["bound"]], edgecolors="k", linewidths=0.5, zorder=3)
        ax.annotate(str(i), (p["ai"], p["achieved"] / 1e12),
                    fontsize=7, ha="center", va="center", color="white", zorder=4)
        print(f"{i:>3} {p['M']:>5} {p['N']:>6} {p['K']:>6} {p['ai']:>8.0f} "
              f"{p['bound']:>7} {p['achieved']/1e12:>11.0f}  ({p['ai_kind']})")

    # 图例:索引 -> shape
    handles = [plt.Line2D([], [], marker="o", ls="", mfc=c, mec="k", label=k)
               for k, c in colors.items()]
    leg1 = ax.legend(handles=handles, title="bound", loc="lower right", fontsize=9)
    ax.add_artist(leg1)
    label_txt = "\n".join(
        f"{i}: m{p['M']} {p['N']}x{p['K']}" for i, p in enumerate(pts))
    ax.text(1.02, 1.0, label_txt, transform=ax.transAxes, va="top", fontsize=7,
            family="monospace")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Arithmetic Intensity (FLOP/byte)")
    ax.set_ylabel("Performance (TFLOPS)")
    title = f"GB200 FP4 GEMM Roofline  (peak={args.peak:.0f} TFLOPS, BW={args.bw:.0f} TB/s)"
    if not measured:
        title += "\n[点在屋顶 = 理论上界,未实测;用 --measured 灌真实时间]"
    ax.set_title(title, fontsize=11)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
