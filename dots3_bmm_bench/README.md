# dots3 absorbed-BMM benchmark（nvfp4 × nvfp4 = bf16，CUTLASS vs cuBLASLt）

日期: 2026-07-20。机器: GB200 (SM100a), CUDA 13.2, libcublasLt 13.4.0.1,
torch 2.13.0a0+cu13.2, nvidia-cutlass-dsl 4.5.2, flashinfer 0.6.15。
**解释器: `/opt/venvs/sglang-dev/bin/python`**（部署 venv；base `/opt/uv/bin/python` 没有
flashinfer）。计划见 `dots3_bmm_plan.md`。

姊妹项目 `../dots3_gemm_nvfp4_bench` / `../dots3_gemm_mxfp8_bench`（dense+group GEMM）：
计时口径、CSV 结构、自包含约定一致，被测算子换成 attention 吸收路径的 **batched matmul**。

## 结论速览

1. **没有任何现成 python 封装支持 batched（3D）nvfp4**——torch / flashinfer / sgl_kernel 的
   fp4 入口全是 2D，deep_gemm 根本没有纯 nvfp4 gemm。两条可用路径都得自己驱动：自写
   cuBLASLt host 调用，和直接驱动官方 CUTLASS CuTe-DSL kernel。
2. **两个后端都只支持 per-TENSOR 全局 scale，没有 per-batch**（实测，非推断）。
3. **cuBLASLt 基本全域更快**：decode 中位数快 7%~51%，prefill 快 6%~64%。
   **唯一例外是 M=1**——CUTLASS 在 A7/C7 反超（ct/lt 0.71~0.76）。
4. 两个后端**每次调用都恰好 1 个 device kernel**（112/112 点），所以"只计主 kernel"的口径
   在这里 **就是**端到端 device 时间，零偏差（不像 dense GEMM bench 有 split-K 陷阱）。

## 被测算子：dots3 的四个吸收 bmm

生产实现 `sglang/python/sglang/srt/agi/models/dots3.py::_absorbed_bmm`（line 1041）走
bf16 `torch.bmm(lhs.transpose(0,1), weight, out=out.transpose(0,1))`：**均匀 batched GEMM**,
batch 维 = 头数，每头一份独立 weight，各 batch 的 M/N/K 相同。M = T = 本步 query token 数。

| 名称 | class / 层 | batch B | M | K | N | 出现阶段 |
|---|---|---|---|---|---|---|
| A5 w_kc | full（13 层，NSA） | 128 | T | 128 | 512 | full 全阶段（prefill+decode/verify）|
| A7 w_vc | full | 128 | T | 512 | 128 | 同上 |
| C5 w_kc | swa+MTP（33+1 层）| 64 | T | 192 | 1024 | **仅 decode/verify + MTP** |
| C7 w_vc | swa+MTP | 64 | T | 1024 | 128 | 同上 |

> ⚠️ 本 bench 是**探索性**的：dots3 生产目前**没有** nvfp4 bmm——即便 nvfp4 ckpt（model30），
> attn 的 bmm 也走 bf16（backbone.md §3.2；dots3.py line 2044 "the absorbed bmm has no fp4
> path"）。这里回答的是"如果把吸收 bmm 量化到 nvfp4，谁能做、数值对不对、哪个快"。
> class C 的生产 T 只落在小档位（swa prefill 无 bmm），大 M 只为刻画 kernel。

## 支持矩阵（全部实跑验证，见 probe_*.log）

| backend | batched nvfp4 | 证据 |
|---|---|---|
| `torch._scaled_mm_v2` | ❌ 2D only | `ValueError: mat_a must be a matrix` |
| `flashinfer.mm_fp4` | ❌ 2D only | `ValueError: mm_fp4 accepts 2d tensors` |
| `sgl_kernel.cutlass_scaled_fp4_mm` | ❌ 2D only | AssertionError |
| flashinfer bmm 家族 | ❌ 无 fp4 | 只有 `bmm_bf16`/`bmm_fp8`/`bmm_mxfp8` |
| deep_gemm | ❌ 无纯 nvfp4 | 只有 `fp8_fp4_gemm_*`(混合)；`einsum` bf16 / `fp8_einsum` fp8 |
| cuDNN | ❌ 未装 | flashinfer 的 batched cudnn fp4 图是 **bf16×fp4 混合** |
| **cuBLASLt strided-batched**（本目录自写） | ✅ | relerr 1.34e-1，M=1~16384 全档 |
| **CUTLASS CuTe-DSL**（官方 example kernel） | ✅ | 过 example 自带 ref check + 我们的 fp32 对照 |

## 两条可用路径怎么驱动的

**cuBLASLt**（`cublaslt_batched_nvfp4/`，自写 JIT）：`cublasLtMatmul` +
`BATCH_COUNT`/`STRIDED_BATCH_OFFSET`，`VEC16_UE4M3` block scale，host 标量 alpha。
写 cuBLAS host 调用不影响代表性（黑盒、可调项少）。

> **未文档化的发现**：cuBLASLt **没有** block-scale 的 per-batch stride 属性（只有
> `A/B_SCALE_POINTER` + `A/B_SCALE_MODE`），所以 batch 时 scale buffer 的寻址布局无从声明。
> 实测它按**最自然的方式**分批寻址：各 batch 的 swizzle 块直接拼接即可
> （`scaled_fp4_quant` 已把 scale 行补到 128）。判据是 **per-batch 误差均匀**——若它广播了
> batch0 的 scale，其余 batch 会崩到 relerr≈1。

**CUTLASS**（`cutlass_batched_nvfp4.py`）：驱动 flashinfer 打包的 NV 官方 CuTe-DSL 例子
`dense_blockscaled_gemm_persistent.py`（默认参数就是 nvfp4）。
**nvidia-cutlass-dsl 是纯 Python DSL，不写 C++、不碰 nvcc**。复用它自己的
`create_and_reorder_scale_factor_tensor` / `construct_cute_pointers`，布局契约是 kernel
作者的。之所以要自己驱动：flashinfer 的封装把张量编译成 2D fake tensor，等于把 L 钉死在 1。

**为什么现成封装拿不到 batched**：底层 CUTLASS kernel 的问题形状是 **(M,N,K,L)**，L 就是
batch 模，原生支持；但 python 封装没暴露出来。

## per-tensor vs per-batch 全局 scale（实测定案）

nvfp4 有两级 scale。**per-16 的 block scale 两边都是 per-batch 的、正常工作**；有争议的是
**per-tensor 的 fp32 全局 scale（alpha）**：

| 路线 | 结果 |
|---|---|
| **对照**：共享标量 alpha，但各 batch 真实 alpha 差 15881 倍 | relerr 0.9999，per-batch=[0.135, 0.742, …, 1.000] —— 证明测试能分辨 |
| cuBLASLt `ALPHA_VECTOR_BATCH_STRIDE` | **NOT_SUPPORTED**（heuristic 返回 0 algo）|
| cuBLASLt `D_SCALE` + `PER_BATCH_SCALAR_32F` | **INVALID_VALUE** |
| CUTLASS batched dense kernel | **无 alpha_tensor**，只有通用 `epilogue_op` |
| flashinfer dense blockscaled kernel | alpha 是 `Single-element tensor`（`reshape(1)`）|

（唯一带 per-batch `alpha_tensor(l,)` 的是 flashinfer 的 **masked grouped MoE** kernel，
那是 group 路径、不是本算子。）

→ **两边统一 per-tensor 全局 scale**。这反而让对比更干净：两个后端消费**同一份量化 buffer**
（cuBLASLt 吃 swizzled scale，CUTLASS 吃同样数字的 un-swizzled 版本，`from_blocked` 换算），
alpha 都在 **kernel 内部**应用（cuBLASLt 用标量 alpha，CUTLASS 折进编译期 `epilogue_op`），
所以差异只来自 kernel 本身。实测两边 relerr 完全一致（均 1.34e-1）。

## CUTLASS 配置规则（一次性测定后固定，无运行时搜索）

CUTLASS 性能对 tile/cluster 敏感。取 CUTLASS/flashinfer 自己的候选 tile 集
（`_SM100_MMA_TILER_MN_CANDIDATES`，即 kernel 作者的合法配置，不是我们发明的），在**真实
dots3 bmm shape** 上实测（`probe_cutlass_cfg.py` / `.log`），固定为：

```
N ≤ 128（w_vc）: M ≤ 256 → (128,64)/(1,1)，否则 (128,128)/(1,1)
N ≥ 512（w_kc）: M ≤ 64 → (128,64)/(1,1)；M ≤ 256 → (128,128)/(1,1)；否则 (128,128)/(1,2)
```

对每格实测最优的最坏 regret ≈ **7.6%**（A7 M=1）；若全程单配置 (128,128)/(1,1) 则最坏 ~11%。
**没用 flashinfer 的 tactic 表**：它按 2D 的 wave quantization（total_ctas vs SM 数）打分、
并在小 M 用 swap_ab 制造 CTA，而 batched 情况下 L 已经把 CTA 数乘上去了，那套推理不迁移。
这条规则是编译期确定性分派，对称于 cuBLASLt 每 shape 跑一次 heuristic。

## bench 计时口径

`bench_kineto(num_tests=10, flush_l2=True)`，只计 BMM 主 kernel 的 CUPTI 纯 kernel 时间：

| 路径 | kernel_substr | 匹配到的 kernel |
|---|---|---|
| CUTLASS | `bs_gemm_example` | `kernel_cutlass_..._bs_gemm_exampleSm100BlockScaledPersistent...` |
| cuBLASLt | `sm100_` | `nvjet_sm100_ootst_*_Avec16UE4M3_Bvec16UE4M3_TNT`（多数）**或** `cutlass3x_sm100_bstensorop_*_block_scaled_ue4m3xf4_ue4m3`（M=1）|

> **cuBLASLt 在 M=1 会派发到它内部自带的 CUTLASS kernel**（`cutlass3x_sm100_bstensorop_*`），
> 不是 nvjet。所以 substr 用 `sm100_` 覆盖两族（不会撞上 CuTe-DSL kernel，那个写作 `Sm100`）。
> 这也解释了 M=1 那几个点的形态——那里"cuBLASLt"其实跑的是 CUTLASS kernel。

**kernel census（`profile_bmm_sweep.py`）：112/112 点、两个后端都恰好 1 个 device kernel**，
无前导/收尾 kernel、无 split-K。所以本口径 = 端到端 device 时间，零偏差。

`gbps` = A fp4 + A block scale + W fp4 + W block scale + D bf16 字节；A scale 按
`ceil(M/128)*128` 行计（buffer 里真实存在的 padding 行）。

## 结果（2026-07-20 全量，零 error、零 relerr 告警）

ct/lt = CUTLASS 时间 / cuBLASLt 时间（**<1 = CUTLASS 快**）：

| bmm | M=1 | decode 中位数 (M≤256) | prefill 中位数 (M≥1024) | 极值 |
|---|---|---|---|---|
| A5 w_kc (B=128, K=128, N=512) | 1.020 | 1.508 | 1.642 | 1.02 ~ 1.70 |
| A7 w_vc (B=128, K=512, N=128) | **0.762** | 1.265 | 1.153 | 0.76 ~ 1.32 |
| C5 w_kc (B=64, K=192, N=1024) | 0.996 | 1.462 | 1.575 | 1.00 ~ 1.59 |
| C7 w_vc (B=64, K=1024, N=128) | **0.707** | 1.072 | 1.064 | 0.71 ~ 1.17 |

- **cuBLASLt 基本全域更快**。w_kc（宽 N）优势大（1.4~1.7x），w_vc（N=128）优势小（1.05~1.3x）。
- **M=1 是唯一 CUTLASS 赢的点**：A7 快 24%、C7 快 29%、C5 打平。正是 cuBLASLt 在 M=1 掉进
  内部 cutlass3x kernel 的那些点——对 decode（T=bs，可能就是个位数）值得注意。
- 带宽：cuBLASLt 大 M 达 6.0~6.9 TB/s；decode 段两边都远未打满（算子本身 launch/延迟受限）。
- 精度：全部点 relerr ≈ 1.34e-1（nvfp4 固有量化误差），两后端逐点一致。

## 复现

```bash
cd /mnt/3fs/dots-pretrain/daijiangkun/torch_matmul_bench/dots3_bmm_bench
PY=/opt/venvs/sglang-dev/bin/python
cd benchmark && $PY bench_bmm.py            # 全量 -> ../result/*.csv
$PY bench_bmm.py A5_w_kc                    # 单个 bmm
```

探测/一次性脚本（可复跑验证）:
- `probe_bmm.py` — 支持矩阵：两后端同输入 + relerr vs fp32
- `probe_cublaslt_batched.py` — cuBLASLt batched 可行性 + scale 分批布局
- `probe_per_batch_scale.py` — per-batch 全局 scale 两条路线（均不支持）
- `probe_cutlass_batched.py` — CUTLASS 正确性 + **真 bmm 而非 L 次 dense gemm** 的 kernel 计数
- `probe_cutlass_cfg.py` — tile/cluster 一次性对测（配置规则依据）
- `profile_bmm_sweep.py` — 全点位 kernel census

## 目录

- `cublaslt_batched_nvfp4/` — cuBLASLt strided-batched JIT 扩展（csrc + loader）
- `cutlass_batched_nvfp4.py` — 驱动官方 CuTe-DSL kernel（纯 python）
- `quant_utils.py` — nvfp4 量化 + `to_blocked`/`from_blocked` swizzle 换算（vendored）
- `dg_bench.py` + `dg_testing/` — bench_kineto 计时工具（vendored）
- `benchmark/` — `bmm_lib.py` + `bench_bmm.py`
- `result/` — 4 个 CSV
