# dots3 BMM benchmark 计划（GB200 / SM100a，nvfp4）

日期：2026-07-20。**状态：已执行完毕，结论与实测数据见 `README.md`。**
本文件保留为立项计划（口径与取舍的由来）；执行中修正的两点记在这里：
- CUTLASS 走的是 **nvidia-cutlass-dsl（纯 Python DSL）**，不需要写 C++、不需要 vendored
  cutlass 头文件（§5 里的 `third_party/cutlass/` 未使用）。
- **两个后端都只支持 per-TENSOR 全局 scale**（per-batch 均实测不支持），所以两边同量化输入、
  同一份 fp32 参考，比 §0 设想的更干净。

机器：GB200 (SM100a)，CUDA 13.2，torch 2.13.0a0+cu13.2。
**解释器：`/opt/venvs/sglang-dev/bin/python`**（部署 venv，`--system-site-packages`：含 torch /
sgl_kernel / deep_gemm / cutlass-dsl / triton / **flashinfer 0.6.15 / flash_mla / deep_ep /
editable sglang**）。venv 在本地盘，换开发机要先跑 `hmmt_feb_2026_bench/src/00_dots3_env.sh` 重建。

姊妹项目：`../dots3_gemm_nvfp4_bench`（dense/group nvfp4 GEMM）——本项目复用它的 nvfp4 量化、
计时口径、CSV 结构、自包含（vendored）约定，被测算子换成 **batched matmul（bmm）**。

---

## 0. 目标与精度范围

对 dots3 attention 吸收路径里的 **w_kc / w_vc 两个 bmm**，做 **nvfp4 × nvfp4 = bf16** 的
bmm 支持面调研 + 性能对比：

- **只测 nvfp4 bmm**（fp4 e2m1 + per-16 e4m3 block scale + per-tensor fp32 global scale，
  输出 bf16）。不测 fp8、不测其它量化。
- **精度 baseline：fp32**（唯一高精度参考，untimed，**只给每个 nvfp4 kernel 算 relerr**，
  不进性能对比、不做"相对高精度快不快"的比较）。按 [[fp8-precision-baseline-caveat]] 用 fp32 作
  ground truth。

> 语义前提：dots3 生产目前没有 nvfp4 bmm——即便 nvfp4 ckpt（model30），attn 的 bmm 也走 bf16
> `torch.bmm`（backbone.md §3.2；dots3.py line 2044 明确 "the absorbed bmm has no fp4 path"）。
> 本 bench 是**探索性**的：假设把吸收 bmm 量化到 nvfp4，(a) 哪些 backend 能做、(b) 数值对不对、
> (c) 各 nvfp4 backend 之间哪个最快。

---

## 1. dots3 的 bmm 行为与 shape（backbone.md + dots3.py 核实）

### 1.1 算子形态（`dots3.py::_absorbed_bmm`, line 1041）

生产走 bf16 `torch.bmm(lhs.transpose(0,1), weight, out=out.transpose(0,1))`。它是
**均匀 batched GEMM**：batch 维 = 头数 H，每头一份独立 weight，所有 batch 的 M(=T)/N/K 相同。
布局 `lhs[T,H,R].transpose(0,1)=[H,T,R]` × `weight[H,R,D]` → `[H,T,D]`，以
`out[T,H,D].transpose(0,1)` view 写回。

**本 bench 要找的是真正支持 3D batched nvfp4 matmul 的 backend**（保持 `[B,M,K]×[B,K,N]`
算子形态）。

### 1.2 四个 bmm 的精确 shape（batch = 头数；M = T = 本步 query token 数）

| 名称 | class / 层 | batch B | M | K | N | 出现阶段 |
|---|---|---|---|---|---|---|
| A5 w_kc | A / full（13 层，NSA） | 128 | T | 128 | 512 | full 全阶段（prefill+decode/verify）|
| A7 w_vc | A / full | 128 | T | 512 | 128 | 同上 |
| C5 w_kc | C / swa+MTP（33+1 层）| 64 | T | 192 | 1024 | **仅 decode/verify + MTP 全部**（swa prefill 无 bmm）|
| C7 w_vc | C / swa+MTP | 64 | T | 1024 | 128 | 同上 |

**nvfp4 可表示性**：block scale per-16 沿 K，四个 K（128/512/192/1024）全整除 16
（8/32/12/64 块）→ 都能量化，K=192 也没问题（192/16=12）。swizzle 的 128 行 tile / 4 列 atom
padding 照常（[[mxfp8-bench-results]] 里 A-scale 布局那套）。

### 1.3 M(=T) 档位与生产工况

沿用 GEMM bench `M_ALL = [1,2,4,8,16,32,64,128,256, 1024,2048,4096,8192,16384]`：
- **class A（full，A5/A7）**：full 层 prefill 也吸收 → **全档位都真实**。
- **class C（swa/MTP，C5/C7）**：只在 decode/verify/MTP 微步出现 → **生产 T 只在小档位**
  （decode≈bs、verify=4×bs、draft 微步=bs）。大 M 照跑但只为刻画 kernel，结论按 M≤256 解读。

### 1.4 workload 特征（nvfp4 权重体量小，decode 偏 launch/带宽受限）

nvfp4 权重 = fp4（半字节）+ e4m3 block scale（per-16），每次调用重读：

| bmm | fp4 权重 | block scale | 合计 |
|---|---|---|---|
| A5 w_kc | 128·512·128/2 = 4.19 MB | 128·512·8 = 0.52 MB | ~4.7 MB |
| A7 w_vc | 128·128·512/2 = 4.19 MB | 128·128·32 = 0.52 MB | ~4.7 MB |
| C5 w_kc | 64·1024·192/2 = 6.29 MB | 64·1024·12 = 0.79 MB | ~7.1 MB |
| C7 w_vc | 64·128·1024/2 = 4.19 MB | 64·128·64 = 0.52 MB | ~4.7 MB |

- **decode（T=1..256）**：读 ~5-7MB nvfp4 权重 + 128/64 个极小 fp4 gemm → **强 launch/调度受限**。
  关键差异在 batch 排布效率、tiny-M tile、单次 python 调用有无额外 kernel（延续 GEMM bench 的
  split-K / 前导 kernel census 教训 [[mxfp8-bench-results]]）。
- **prefill（仅 class A，T≥1024）**：每头是像样 fp4 gemm，算力受限。

---

## 2. 候选 backend（真 batched nvfp4；优先现成封装）

venv 已装：torch 2.13、triton 3.7、deep_gemm 2.3(dots)、sgl_kernel 0.4.3、
nvidia-cutlass-dsl 4.5.2、flashinfer 0.6.15、flash_mla、deep_ep、cuda-python。

**CUTLASS 一律优先通过现成封装调用**（sgl_kernel / flashinfer / torch 等）：cutlass 不同 tile/
cluster/schedule 配置性能差距明显，自己配的配置未必代表 cutlass 的真实水平，封装里的配置是
kernel 作者调过的。**只有在所有现成封装都没有 batched nvfp4 入口时**，才按 NV 官方 cutlass 仓库
C++ 自己配（那时要如实说明配置是自选的、并按 mxfp8 dense 的做法一次性测定配置规则）。
**cuBLAS(Lt) 的 C++ 可以直接自己写**——它比较黑盒、可调项少，自写不影响代表性。

| # | backend | 入口 / 待查 API | 层级 | 预期 |
|---|---|---|---|---|
| 1 | **flashinfer nvfp4 batched** | 有 `bmm_fp8`；查有无 fp4 batched（`mm_fp4` 是 2D）| 上层→cutlass | 待验 |
| 2 | **sgl_kernel nvfp4 batched** | `cutlass_scaled_fp4_mm` 是 2D；查有无 bmm/batched fp4 | 上层→cutlass | 待验 |
| 3 | **torch scaled batched** | `_scaled_mm_v2` 是否收 3D / 有无 batched 变体 | 上层→cublas/cutlass | 待验 |
| 4 | **cuBLASLt batched nvfp4** | 自写 JIT：strided-batch layout + `VEC16_UE4M3` + `CUDA_R_4F_E2M1` | 底层 | 待验（允许自写）|
| 5 | DeepGEMM | 查有无纯 nvfp4 batched/einsum（只见 `m_grouped_fp8_fp4_gemm` 混合精度）| 上层→cutlass | 待验/存疑 |
| 6 | Triton nvfp4 | 手写 fp4 dequant + batched matmul | 底层 | 最低优先 |
| 7 | CUTLASS 自写 batched nvfp4 | 官方仓库 C++ 自配（per-batch swizzle scale 是难点）| 底层 | **末选**，仅当 1/2/3/5 全无入口 |

- **TransformerEngine / cuDNN(python)**：本机未装，**本次不测**（不动环境）。
- **精度 baseline**：fp32 `torch.bmm`（ground truth，untimed，只算 relerr，不进性能对比）。
- **fallback（本次不测）**：把 bmm 表达成"等长 group gemm（group=head，m_e=T 全等）"可以复用
  nvfp4 GEMM group bench 的路径，但那不是本 bench 的测试目标——**只有在确认没有任何 backend 支持
  真 batched nvfp4 时**才作为退路，届时先跟用户确认再测。
- **只参考不进对比**：FlashMLA / fa4 会把 w_vc（有时 w_kc）融进 attention core——那是改算子边界、
  不是独立 bmm，只在 README 定性提一句。

---

## 3. 支持情况调研（每个 backend：能跑 → 数值对 → 再 bench）

`probe_bmm.py`（[[verify-by-running-not-inferring]]）：
1. **API 盘点**：先在各包里找出真正的 batched/3D nvfp4 入口（有没有、签名如何、scale 布局要求）。
2. **能否跑通**本 shape/dtype（尤其 batch、tiny M=1、C5 的 K=192/N=1024）：接异常、记原因。
3. **数值对不对**：对 **fp32 batched 参考**算 max relerr。nvfp4 预期 relerr 量级 ~1.3e-1
   （对齐 GEMM nvfp4 dense 基线）；**relerr=inf/nan/明显偏大 = 不支持或用法错**，不能因
   "没报错"就算过。用真实/outlier 数据看精度、maxabs=0 只是正确性
   （[[fp8-precision-baseline-caveat]]）。

产出"支持矩阵"（backend × 四个 bmm → ✅/❌+原因），**先给用户过目**再进性能对比。

---

## 4. benchmark 方法（口径与 GEMM bench 对齐）

### 4.1 计时
- `bench_kineto(num_tests=10, flush_l2=True)`，只计 **bmm 主 kernel** CUPTI 纯 kernel 时间，
  substr 精确匹配。**各 backend substr 由 kernel census 执行时定**。
- NVML sampler（1ms 窗口平均，被 flush memset 稀释）出 sm_mhz/power_w，复用 `NvmlSampler`。

### 4.2 kernel census（**必做**，延续 split-K 教训 [[mxfp8-bench-results]]）
- `profile_bmm_sweep.py`：census 每 backend×点位裸 python 调用的 device kernel（个数、主 gemm
  占端到端比例、列"额外"kernel）。已知风险：前导 setup kernel、cublas split-K 多
  `splitKreduce_kernel`。主 kernel 占比明显 <100% 时，CSV 保主 kernel 口径、README 附端到端
  修正表（不翻结论就不改 CSV，同 GEMM bench）。

### 4.3 指标
- `tflops = 2·B·M·N·K / t`；`gbps =` A fp4 + A scale + weight fp4 + weight scale + out bf16 字节 / t。
  decode 段重点看 gbps 逼近 roofline；prefill 段看 tflops。

### 4.4 正确性
- 每点对 fp32 batched 参考算 max relerr，写日志；异常/relerr 超阈值标 ERROR，不进结果表。

### 4.5 CSV 列（比 GEMM bench 多 batch 维）
```
class, name, batch, m, k, n, us, tflops, gbps, sm_mhz, power_w, backend
```
每个 bmm（A5/A7/C5/C7）一个 CSV，backend 交错写入。

### 4.6 约定沿用
- **不做运行时 autotune 搜索**（cublasLt 每 shape 一次 heuristic 是既有自由度）。
- nvfp4 global scale：**全 head 共用一个 per-tensor global scale**（各 batch alpha 相等，满足
  cublasLt 单标量 alpha 语义、便于跨 backend 逐位对比），沿用 GEMM bench 约定。

---

## 5. 目录结构与自包含要求

**完整独立项目**，不依赖 sglang / DeepGEMM / cutlass 等其他目录 checkout（沿用 GEMM bench 要求）。

```
dots3_bmm_bench/
  dots3_bmm_plan.md              # 本计划
  README.md                      # 执行后：支持矩阵 + 结果 + 复现
  benchmark/
    bmm_lib.py                   # build_bmm / measure / NvmlSampler / csv
    bench_bmm.py                 # 四个 bmm × M 档 × 各 backend
    run_all.sh
  cublaslt_batched_nvfp4/        # 自写 cuBLASLt batched JIT（#4）
  third_party/cutlass/           # 仅当走到 #7 才需要（vendored cutlass 头）
  quant_utils.py                 # nvfp4 量化（从 nvfp4 bench 复制）
  dg_bench.py + dg_testing/      # vendored bench_kineto/count_bytes
  probe_bmm.py                   # API 盘点 + 支持矩阵探测（跑通 + relerr vs fp32）
  profile_bmm_sweep.py           # kernel census
  result/*.csv
```

JIT 编译沿用教训：torch 2.13 头文件必须 **`-std=c++20`**（[[mxfp8-bench-results]]）。

## 6. 交付物
1. 本 `dots3_bmm_plan.md`。
2. `README.md`（backend×bmm 支持矩阵 + 各档位结论 + 复现命令，含所用解释器）。
3. `result/` CSV（四个 bmm × 全档 × 各支持 backend）。
4. `probe_bmm.py` / `profile_bmm_sweep.py` 及 log。
5. memory：一条 `dots3-bmm-bench-results`，链回 [[nvfp4-mxfp8-gemm-support-matrix]]
   [[mxfp8-bench-results]]。

## 7. 风险与开放问题
- **可能没有任何现成封装支持真 batched nvfp4**：那 #4（自写 cuBLASLt batched）与 #7（自写
  cutlass）就是仅有的路；若两者也不行，才回到 §2 的 group fallback，**届时先问用户**。
- **substr 未知**：各 nvfp4 bmm kernel 名执行时才知道 → census 先行。
- **自写 cutlass 的配置代表性**（#7）：配置敏感，若走到这步须一次性测定配置规则并在 README 说明
  配置是自选的。
- **DeepGEMM 可能无纯 nvfp4×nvfp4**（只见 fp8×fp4 混合）→ 实测确认，不支持就如实记。
- **venv 存在性**：换开发机后 `/opt/venvs/sglang-dev` 需重跑 `00_dots3_env.sh` 才有 flashinfer。

## 8. 执行顺序
1. **API 盘点 + 骨架**：查 flashinfer / sgl_kernel / torch / deep_gemm 有无 batched nvfp4 入口；
   复制 vendored 文件（quant_utils、dg_testing），写 `bmm_lib.py`、四个 bmm shape 表。
2. **nvfp4 支持探测**：`probe_bmm.py` 出支持矩阵（跑通 + relerr vs fp32），**给用户过目**。
3. **kernel census**：`profile_bmm_sweep.py` 定各 backend substr、验主 kernel 口径。
4. **nvfp4 性能对比**：`bench_bmm.py` 全档跑通过探测的 backend，出 CSV + README 结论表。
5. 写 memory、更新 README 支持矩阵与结论。
