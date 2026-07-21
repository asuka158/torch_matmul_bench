# dots3 mxfp8 GEMM benchmark（dense + group，CUTLASS vs cuBLASLt）

日期: 2026-07-17。机器: GB200 (SM100a), CUDA 13.2 (V13.2.78), libcublasLt 13.4.0.1,
torch 2.13.0a0+cu13.2。解释器: `/opt/uv/bin/python`。

姊妹项目 `../dots3_gemm_nvfp4_bench`（nvfp4 版）：shape 类目、M/T 档位、计时口径、CSV
格式与本项目完全一致，两套结果可直接对表。本目录自包含（vendored cutlass 头文件与
JIT csrc），不依赖 sglang / DeepGEMM / cutlass 等其他项目目录；外部依赖只有 pip 包
（torch、pynvml）与 CUDA toolchain。**mxfp8 不用 sgl_kernel**（其 0.3.6 无任何 mx op）。

## 四路入口（与 nvfp4 的差异）

| GEMM | 后端 | 调用 | 说明 |
|------|------|------|------|
| dense | CUTLASS | `cutlass_dense_mx.mxfp8_mm_2sm/_1sm`（本目录 JIT） | sgl_kernel 无 mxfp8 op，按 cutlass example 72 自写；配置见下 |
| dense | cuBLASLt | `torch._scaled_mm_v2` (BlockWise1x32 + SWIZZLE_32_4_4) | 主 kernel `nvjet_sm100_*_Avec32UE8M0_*` |
| group | CUTLASS | `torch._scaled_grouped_mm_v2` (BlockWise1x32, offs) | kernel 是 **cutlass_mslk**（torch 内置 CUTLASS，非 sgl_kernel）|
| group | cuBLASLt | `cublaslt_group_gemm_mx.cublaslt_mxfp8_group_mm`（本目录 JIT） | fp4 版扩展克隆：`CUDA_R_8F_E4M3` + `VEC32_UE8M0`，algo 固定 heuristic algo[0]（既定约定） |

与 nvfp4 的关键语义差异：
- **无 per-tensor global scale**：e8m0 逐 32 块 scale 自带动态范围，alpha 恒为 1.0。
  cublasLt grouped 的"单标量 alpha"限制自动满足，无需 nvfp4 那套全 expert 共享
  global scale 的对齐。
- **量化不走 sgl_kernel**：`quant_utils.quant_mxfp8`（手写 e4m3 + e8m0，OCP MX 语义）+
  torchao 风格 `to_blocked` 32x4x4 swizzle。nvfp4 的 `scaled_fp4_experts_quant`
  越界读 offsets bug 在本路径不存在；group 路由簿记用纯 torch（argsort/bincount）。
- 精度基线：randn 数据 relerr ≈ 3.77e-2（nvfp4 是 1.34e-1）。

## group A-scale 布局（实测定案，见 probe_group.py）

torch grouped 与 cublasLt ext 的 per-expert A-scale 均为「各 expert 的 swizzled 段
按 expert 顺序拼接，每段 128 行 padding」（`blockscale_offsets = cumsum(round_up(m_e,128))`，
与 fp4 cutlass_fp4_group_mm 同约定）。用不均匀 m_e=[3,0,17,129,64,1,250,33]（含空 expert、
m<32）实测：该布局 relerr=3.77e-2 正确；整矩阵 to_blocked + 行偏移是错的（relerr=inf）。
cublasLt ext 与 torch grouped 同 buffer 输出 **bitwise 一致**。

## dense CUTLASS 配置（一次性测定后固定为 M 阈值规则）

kernel 无 autotune（与 nvfp4 侧结论一致，配置编译期定死）。编了两个候选、用正式
口径（bench_kineto num_tests=10 flush_l2 主 kernel）在 6 shape × 8 M 档上一次性对测
（`probe_dense_cfg.py` + 边界补测，结果存 probe_dense_cfg.log）：
- `mm_2sm`: MmaTileShape 256x256x128, TmaWarpSpecialized2Sm, cluster(4,4) fallback(2,1)
  —— 镜像 nvfp4 sgl_kernel dense 默认配置
- `mm_1sm`: MmaTileShape 128x128x128, TmaWarpSpecialized1Sm, cluster(1,1)

实测分域明显：M≤256 时 1SM 全部 shape 胜（快 20~35%）；M≥4096 时 2SM 多数胜
（5~24%）；512~2048 边界区按 shape 交错（差距 ≤17%，无干净全局阈值）。
**选定规则（固定，不做运行时搜索）：M ≤ 1024 → 1SM，M > 1024 → 2SM**
（min-max-regret：最坏点 ~11%；若全程单配置 2SM，fused_qkv M=512/1024 慢 55%）。
该规则是编译期定死的确定性分派，对称于 cublasLt dense 每 shape 跑一次 heuristic
选 nvjet 变体的既有自由度。`MXFP8_DENSE_CUTLASS_CFG=2sm|1sm` 可强制单配置复测。

## bench 计时口径：只计 GEMM 主 kernel（与 nvfp4 相同）

bench_kineto 精确子串（num_tests=10, flush_l2=True）:

| 路径 | kernel_substr | 匹配到的主 kernel |
|------|---------------|-------------------|
| dense CUTLASS | `device_kernel` | `cutlass::device_kernel<...Mxf8...>`（本目录 JIT，进程内唯一）|
| dense cuBLASLt | `nvjet` | `nvjet_sm100_*_Avec32UE8M0_Bvec32UE8M0_TNT` |
| group CUTLASS | `GroupProblemShape` | `cutlass_mslk::device_kernel<...GroupProblemShape...>`；自动排除 `set_grouped_gemm_args_kernel` 前导 |
| group cuBLASLt | `nvjet` | `nvjet_sm100_*_ptrGroup*`；自动排除 python 侧指针数组更新 |

sm_mhz/power_w：NVML 1ms 采样窗口平均（被 flush memset 稀释，非持续工况点）。
gbps：operand+scale+output 字节；group 只计 **active expert**（m_e>0）的权重字节。

### dense 端到端 kernel census（2026-07-19，profile_dense_sweep.py）

对全部 364 个 dense 点 profile 裸 python 调用的 device kernel 构成：
- **CUTLASS：全部恰好 1 个 kernel**（本口径 = 端到端，零偏差）。
- **cuBLASLt：39 个点走 split-K**（额外 `splitKreduce_kernel`，`'nvjet'` 子串不匹配 →
  **CSV 少记 cublas ~19-30%**）：fused_qkv nsa m≤64 / swa m≤32、o_proj nsa/swa m≤16、
  gate_up tp4 m≤16 / tp8 m≤32、down tp1 m≤16（mxfp8 的 split-K 触发面比 nvfp4 宽，
  K=5120 也会触发）。
- 修正成端到端后**零翻转**：cublas 仍全域更快，受影响点 ct/lt 从 1.43~1.88 收窄到
  1.04~1.52（最紧 gate_up tp4 m=4~16 ≈1.04~1.08，接近平手）。明细见
  profile_dense_sweep.log。CSV 保持主 kernel 口径未改。

## shape 类目（同 nvfp4，8 类目 8 CSV）

dense（M 22 档，2026-07-21 补测后 —— decode `1,2,4,8,16,32,64,128,129,256,257,512,513`；
prefill `1024,1536,2048,4096,5120,8192,10240,16384,20480`，见"覆盖率补测"节）:

| 类目 | 变体 (N, K) |
|------|-------------|
| fused_qkv_a_g_proj_with_mqa | nsa (1920, 5120), swa (2176, 5120)（5210→5120 修正）|
| fused_q_b_wq_b_proj | nsa (32768, 1024), swa (16384, 1024) |
| kv_b_proj | nsa (32768, 512), swa (20480, 1024) |
| o_proj | nsa (5120, 16384), swa (5120, 8192) |
| gate_up_proj_dense | tp1 (27648, 5120), tp4 (6912, 5120), tp8 (3456, 5120) |
| down_proj_dense | tp1 (5120, 13824), tp4 (5120, 3456), tp8 (5120, 1728) |

group（E=257, topk=9, sum_m=T*9；T 档位同 M）:

| 类目 | 变体 (N, K) |
|------|-------------|
| gate_up_proj_group | tp1 (3072, 5120), tp4 (768, 5120), tp8 (384, 5120) |
| down_proj_group | tp1 (5120, 1536), tp4 (5120, 384), tp8 (5120, 192) |

## 结果结论（2026-07-17 全量跑完，零 mismatch/零 relerr 告警）

ct/lt = CUTLASS 时间 / cuBLASLt 时间（<1 = cutlass 快），按类目中位数：

| 类目 | decode (M≤256) | prefill (M≥1024) | 极值 |
|------|---------------|------------------|------|
| fused_qkv_a_g_proj_with_mqa | 1.610 | 1.261 | 1.09~1.98 |
| fused_q_b_wq_b_proj | 1.220 | 1.156 | 1.08~1.42 |
| kv_b_proj | 1.200 | 1.143 | 1.02~1.34 |
| o_proj | 1.578 | 1.142 | 1.08~1.88 |
| gate_up_proj_dense | 1.428 | 1.192 | 1.07~1.74 |
| down_proj_dense | 1.331 | 1.162 | 1.08~1.76 |
| gate_up_proj_group | 0.656 | 0.886 | 0.38~1.07 |
| down_proj_group | 0.651 | 0.947 | 0.37~1.09 |

- **dense：cuBLASLt (nvjet) 全域更快**（即便 cutlass 侧已按 M 阈值选 1SM/2SM 最优配置），
  decode 快 20~60%，prefill 快 14~26%。与 nvfp4（decode cublas 快 50~70%、prefill 近平）
  相比，mxfp8 的 cublas 优势延伸到了 prefill。
- **group：CUTLASS (torch cutlass_mslk) 基本全域更快**，decode 快 ~35%（中位数），
  prefill 接近（个别大 T 点 cublas 略快 ≤9%）。与 nvfp4 group（gate_up cutlass 快 /
  down 大 T cublas 快）方向一致但 cutlass 优势更大。

## 覆盖率补测（2026-07-21，增量，未重跑旧点）

原网格的 M 全是 2 的幂、每个都正好对齐 128 的 tile，且 256→1024 是唯一 4× 断层。补了 8 档：
`129,257,513`（tile 边界探针）、`512,1536`（填断层）、`5120,10240,20480`（真实工况：
`--chunked-prefill-size 20480`，开 dp-attention 后 `server_args.py` 会 `// dp_size`
→ 每 rank 5120；MoE 侧 gather 回 20480）。只跑缺的点、追加后重排，旧行未动；
拼接前用 M=256/1024 做控制点，与存档值差 -1.2%~+2.7%（正常抖动）。

- **tile 量化悬崖**：跨 128 边界（M→M+1）耗时跳变，dense 侧 CUTLASS 中位 1.011 / 最大 **1.422**，
  cuBLASLt 中位 1.042 / 最大 **1.530**；**group 基本免疫**（中位 1.000，最大 1.098）——
  group 本来就是各 expert 变长 m，kernel 早就在处理不齐整形状。
  与 nvfp4 不同的是：**mxfp8 dense 上 cuBLASLt 的跨界跳变比 CUTLASS 还大**。
- **结论未变**：mxfp8 dense 补完 8 档后**仍是 cuBLASLt 全域更快**（22×2×6 类目里没有任何一点
  ct/lt<1），没有出现 nvfp4 dense 那种 M=512 的窄反超窗口。group 侧 CUTLASS 依旧全域领先，
  新增档位的 ct/lt 与相邻档位连续（如 gate_up tp8 M=512:0.529、513:0.527）。

## 复现

```bash
cd /mnt/3fs/dots-pretrain/daijiangkun/torch_matmul_bench/dots3_gemm_mxfp8_bench/benchmark
bash run_all.sh                       # 全量：6 dense + 2 group -> ../result/*.csv
/opt/uv/bin/python bench_dense.py [类目...]   # 单独 dense
/opt/uv/bin/python bench_group.py [类目...]   # 单独 group（子进程隔离+断点续跑）
```

探测/一次性脚本（复跑可验证结论）:
- `probe_dense.py` — dense 三路 vs fp32 ref + cutlass-vs-cublas 逐位比较
- `probe_group.py` — group A-scale 布局实证 + cublasLt ext 三方对拍
- `probe_dense_cfg.py` — dense CUTLASS 2SM/1SM 一次性对测（配置选定依据）

## 目录

- `cutlass_dense_mx/` — dense CUTLASS JIT 扩展（csrc + loader；编译约几分钟，缓存于
  `~/.cache/torch_extensions`）
- `cublaslt_group_gemm_mx/` — group cuBLASLt JIT 扩展（fp4 版克隆改 mxfp8）
- `third_party/cutlass/` — vendored cutlass v4.5.2 头文件（include + tools/util/include）
- `quant_utils.py` / `dg_testing/`+`dg_bench.py` — mxfp8 量化 / DeepGEMM 计时工具（vendored）
- `benchmark/` — bench_lib.py + bench_dense.py + bench_group.py + run_all.sh
- `result/` — 8 个 CSV
