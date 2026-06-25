#!/usr/bin/env python3
"""DeepGEMM MXFP4 dense-GEMM roofline 图.

先画一条 roofline 峰值折线(屋顶 = min(算力天花板, 带宽斜坡)),
再把 result CSV 里 50 个 shape 的实测点叠上去,这样一眼能看到
每个 shape 离峰值(屋顶)还差多远。

坐标:
  x = arithmetic intensity = FLOP/byte  (log)
  y = achieved performance = TFLOPS      (log)

CSV 里已有 tflops(实测吞吐)和 gbps(按 compulsory 流量算的带宽),
两者直接给出每个点的坐标:
  AI       = tflops*1e12 / (gbps*1e9) = tflops/gbps*1000   (FLOP/byte)
  achieved = tflops                                         (TFLOPS)
(已核对:gbps 用的 compulsory 字节 = 0.5*MK + 0.5*NK + 4*MN + (MK+NK)/32,
 含 32 元素一块的 E8M0 scale,row1 完全对得上。)

用法:
  python dg_roofline.py
  python dg_roofline.py --csv 别的.csv --peak 10000 --bw 8 --out dg_roofline.png
"""
import argparse
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

DEFAULT_CSV = "DeepGEMM/tests/dg_test/result/mxfp4_fp32_dense_DG_run30_50shape.csv"
# GB200 单 GPU FP4 dense 规格峰值 ~10 PFLOPS;本次 run 在 2062 MHz 不降频、
# 功耗 <600W,没进 power-bound 区间(实测最高点已到 7.29 PFLOPS),所以
# 用规格峰值当屋顶,而不是 power-bound 的 ~6 PFLOPS。
DEFAULT_PEAK_TFLOPS = 10000.0   # 算力天花板
DEFAULT_BW_TBS      = 8.0       # HBM3e ~8 TB/s


def roof_tflops(ai, peak_tflops, bw_tbs):
    """屋顶(TFLOPS) = min(算力峰值, AI * 带宽)。ai 单位 FLOP/byte。
    AI[FLOP/byte] * bw[byte/s] = FLOP/s;bw=bw_tbs*1e12 byte/s,
    除 1e12 转 TFLOPS => 斜坡 = AI * bw_tbs。"""
    return np.minimum(peak_tflops, ai * bw_tbs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=DEFAULT_CSV)
    ap.add_argument("--peak", type=float, default=DEFAULT_PEAK_TFLOPS,
                    help="算力峰值 TFLOPS (default ~10000 = FP4 dense 规格峰值)")
    ap.add_argument("--bw", type=float, default=DEFAULT_BW_TBS,
                    help="HBM 带宽 TB/s (default 8)")
    ap.add_argument("--out", default="dg_roofline.png")
    ap.add_argument("--title", default="DeepGEMM MXFP4 dense GEMM",
                    help="图标题前缀(自动接 '— Roofline (GB200, N shapes)')")
    args = ap.parse_args()

    ridge = args.peak / args.bw   # ridge point AI = peak/bw (FLOP/byte)

    # ---- 读 CSV,直接用 tflops / gbps 得到每个点的 (AI, achieved) ----
    pts = []
    with open(args.csv) as f:
        for row in csv.DictReader(f):
            m, n, k = int(row["m"]), int(row["n"]), int(row["k"])
            tfl  = float(row["tflops"])
            gbps = float(row["gbps"])
            ai = tfl / gbps * 1000.0                 # FLOP/byte
            roof = roof_tflops(ai, args.peak, args.bw)
            pts.append(dict(
                m=m, n=n, k=k, ai=ai, tflops=tfl, gbps=gbps,
                roof=roof,
                eff=tfl / roof,                      # 离屋顶多远(达成率)
                bound="compute" if ai >= ridge else "memory",
            ))

    # ---- 画图 ----
    fig, ax = plt.subplots(figsize=(11, 7.5))

    # 峰值折线(屋顶)
    ai_lo = min(p["ai"] for p in pts) / 3
    ai_hi = max(p["ai"] for p in pts) * 3
    xs = np.logspace(np.log10(ai_lo), np.log10(ai_hi), 500)
    ax.plot(xs, roof_tflops(xs, args.peak, args.bw),
            "k-", lw=2.5, zorder=5,
            label=f"Roofline  (peak {args.peak/1000:.0f} PFLOPS, BW {args.bw:.0f} TB/s)")

    # 算力天花板 / 带宽斜坡 / ridge 辅助线
    ax.axhline(args.peak, ls="--", c="0.4", lw=1)
    ax.text(ai_lo, args.peak, f" compute roof = {args.peak/1000:.0f} PFLOPS",
            va="bottom", ha="left", fontsize=9, color="0.3")
    ax.axvline(ridge, ls=":", c="0.55", lw=1)
    ax.text(ridge, 35, f"ridge = {ridge:.0f} FLOP/byte ", rotation=90,
            va="bottom", ha="right", fontsize=8, color="0.45")

    # 本数据集实测最高点,当作现实可达上界参考
    best = max(pts, key=lambda p: p["tflops"])
    ax.axhline(best["tflops"], ls="-.", c="tab:green", lw=1, alpha=0.8)
    ax.text(ai_hi, best["tflops"],
            f"best measured {best['tflops']:.0f} TFLOPS "
            f"(m{best['m']} {best['n']}x{best['k']})  ",
            va="bottom", ha="right", fontsize=8, color="tab:green")

    # 每个点画一条到屋顶的竖线,直观显示“离峰值的差距”
    for p in pts:
        ax.plot([p["ai"], p["ai"]], [p["tflops"], p["roof"]],
                c="0.7", lw=0.7, alpha=0.5, zorder=2)

    # 按 M 上色(5 个 M × 10 个 NK = 50 个点)
    m_vals = sorted({p["m"] for p in pts})
    cmap = plt.cm.viridis(np.linspace(0.05, 0.9, len(m_vals)))
    m_color = {m: cmap[i] for i, m in enumerate(m_vals)}
    for p in pts:
        ax.scatter(p["ai"], p["tflops"], s=70, color=m_color[p["m"]],
                   edgecolors="k", linewidths=0.5, zorder=6)

    # M 图例
    m_handles = [Line2D([], [], marker="o", ls="", mfc=m_color[m], mec="k",
                        ms=9, label=f"M = {m}") for m in m_vals]
    leg_m = ax.legend(handles=m_handles, title="batch M", loc="lower right",
                      fontsize=9)
    ax.add_artist(leg_m)
    ax.legend(loc="upper left", fontsize=9)

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Arithmetic Intensity  (FLOP / byte)")
    ax.set_ylabel("Performance  (TFLOPS)")
    ax.set_title(f"{args.title} — Roofline (GB200, {len(pts)} shapes)\n"
                 "vertical stub = each shape's gap to the roof (peak)", fontsize=12)
    ax.grid(True, which="both", ls=":", alpha=0.35)
    ax.set_ylim(top=args.peak * 1.4)
    fig.tight_layout()
    fig.savefig(args.out, dpi=140, bbox_inches="tight")

    # ---- 命令行汇总:每个 shape 离峰值多远 ----
    print(f"peak={args.peak:.0f} TFLOPS  BW={args.bw:.0f} TB/s  "
          f"ridge={ridge:.0f} FLOP/byte\n")
    hdr = f"{'M':>6} {'N':>6} {'K':>6} {'AI':>7} {'bound':>7} " \
          f"{'TFLOPS':>8} {'roof':>7} {'eff%':>6}"
    print(hdr); print("-" * len(hdr))
    for p in sorted(pts, key=lambda p: p["eff"]):
        print(f"{p['m']:>6} {p['n']:>6} {p['k']:>6} {p['ai']:>7.0f} "
              f"{p['bound']:>7} {p['tflops']:>8.0f} {p['roof']:>7.0f} "
              f"{100*p['eff']:>6.1f}")
    effs = np.array([p["eff"] for p in pts])
    print(f"\neff vs roof: min {100*effs.min():.1f}%  "
          f"median {100*np.median(effs):.1f}%  max {100*effs.max():.1f}%")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
