# dots3 GEMM 支持情况探测（8 kernel / 4 组对比）

日期: 2026-07-17。机器: GB200 (SM100a), CUDA 13.2 (V13.2.78), libcublasLt 13.4.0.1,
torch 2.13.0a0+cu13.2, sgl_kernel 0.3.6.post2。解释器: `/opt/uv/bin/python`。

探测脚本（只验证"能跑 + 数值对 + 实际执行的 kernel 是哪个后端"，不测性能）:
- `probe_dense.py` — 4 个 dense kernel
- `probe_group.py` — group kernel（sgl_kernel cutlass group + torch grouped v2）
- `quant_utils.py` — 手写 mxfp8 量化 + torchao 风格 to_blocked swizzle（本机无 torchao）

数值校验: fp32 baseline (randn 数据)。nvfp4 relerr≈1.34e-1、mxfp8 relerr≈3.77e-2，
两后端 relerr 一致（nvfp4 两后端 1.3435e-01 完全相同），即误差全部来自量化本身。

## 结论矩阵

| # | GEMM | 后端 | 支持 | 调用方式 |
|---|------|------|------|---------|
| 1 | nvfp4×nvfp4=bf16 dense | CUTLASS | ✅ | `sgl_kernel.cutlass_scaled_fp4_mm` |
| 2 | nvfp4×nvfp4=bf16 dense | cuBLASLt | ✅ | `torch._scaled_mm_v2` (BlockWise1x16 + SWIZZLE_32_4_4) → `nvjet_sm100_*_Avec16UE4M3` |
| 3 | mxfp8×mxfp8=bf16 dense | CUTLASS | ⚠️ 需自写 C++ | sgl_kernel 0.3.6 无 mxfp8 op；torch dense 只走 cuBLASLt。C++ 参考 cutlass `examples/72_blackwell_narrow_precision_gemm`。代理方案: `torch._scaled_grouped_mm_v2` G=1（cutlass_mslk kernel，实测可跑，但用 group kernel 测 dense 性能有代表性风险） |
| 4 | mxfp8×mxfp8=bf16 dense | cuBLASLt | ✅ | `torch._scaled_mm_v2` (BlockWise1x32 + SWIZZLE_32_4_4) → `nvjet_sm100_*_Avec32UE8M0` |
| 5 | nvfp4×nvfp4=bf16 group | CUTLASS | ✅ | `sgl_kernel.cutlass_fp4_group_mm`（配 `prepare_moe_input` + `scaled_fp4_experts_quant`；输出需 `shuffle_rows(c_map)` 反置换） |
| 6 | nvfp4×nvfp4=bf16 group | cuBLASLt | ⚠️ 需自写 C++ | torch grouped 不支持 fp4（"No gemm implementation was found"）。cuBLASLt grouped API 本机实测可跑（编译运行 `CUDALibrarySamples/cuBLASLt/LtNvfp4gemmGroupedSimple` 成功）。**已有现成扩展**: sglang 树 `sglang/srt/layers/moe/cublaslt_group_gemm`（JIT，python 入口 `cublaslt_fp4_group_mm`，与 sgl_kernel 同签名） |
| 7 | mxfp8×mxfp8=bf16 group | CUTLASS | ✅ | `torch._scaled_grouped_mm_v2` (BlockWise1x32 + SWIZZLE_32_4_4, offs)。注意: 虽由 torch 调用，profiler 确认 kernel 是 **cutlass_mslk**（CUTLASS），故归入 CUTLASS 侧。sgl_kernel 无 mxfp8 group op |
| 8 | mxfp8×mxfp8=bf16 group | cuBLASLt | ⚠️ 需自写 C++ | torch grouped 只有 cutlass 后端，无 cuBLASLt 路径。cuBLASLt grouped mxfp8 本机实测可跑（编译运行 `LtMxfp8gemmGroupedSimple` 成功，CUDA 13.2u1 起支持） |

## nvfp4 cuBLASLt group gemm 的 python 调用（2026-07-17 补充）

- `cublaslt_group_mm.py` — 按文件路径加载 sglang 树里的 JIT 扩展（不 import sglang），
  导出 `cublaslt_fp4_group_mm`，与 `sgl_kernel.cutlass_fp4_group_mm` 同签名。
  约束：per-expert alpha 必须全部相等（cublasLt grouped 只收单标量 alpha）→
  所有 expert 的权重共用一个 per-tensor global scale。
- `probe_cublaslt_group.py` — 实测：同一份量化 buffer 上 cublaslt 与 cutlass 输出
  **bitwise 完全一致**（0/4194304 mismatch），relerr vs fp32 = 1.3440e-1。
  主 kernel: `nvjet_sm100_*_ptrGroup_TNT`。注意每次调用前有 ~11 个小的指针数组
  更新 kernel + 3 个 DtoD memcpy（口径上不计入，子串匹配自动排除）。
- `dg_bench.py` — 按路径加载 DeepGEMM 的 `bench_kineto`/`count_bytes`
  （本机 torch 2.13 与 deep_gemm 的 _C.so ABI 不兼容，不能 import deep_gemm 包）。

## bench 计时口径（已定）：只计 GEMM 主 kernel

bench_kineto 精确子串（四路均已实测通过唯一性断言，num_tests=10, flush_l2=True）:

| 路径 | 调用 | kernel_substr |
|------|------|---------------|
| dense CUTLASS | `sgl_kernel.cutlass_scaled_fp4_mm` | `'device_kernel'` |
| dense cuBLASLt | `torch._scaled_mm_v2` | `'nvjet'` |
| group CUTLASS | `sgl_kernel.cutlass_fp4_group_mm` | `'GroupProblemShape'`（排除 `__get_group_gemm_starts` prologue） |
| group cuBLASLt | `cublaslt_fp4_group_mm` | `'nvjet'`（排除指针更新的 elementwise/memcpy） |

## group gemm 的 kernel 拆分可行性（2026-07-17 探测）

- **cublasLt 侧已天然三段式**：`create_plan`（host，每层一次，含 heuristic）/
  指针数组更新（12 个 python 小 kernel，~20.8µs device，内容上等价于 cutlass prologue，可融合成 1 个
  自定义 kernel；static routing 下可整体跳过）/ `ext.run_plan(pid)`。
  实测：单独调 run_plan **恰好只发射 1 个 nvjet kernel**，且输出与完整调用 bitwise 一致。
- **cutlass 侧目前不可拆但可改**：`sgl_kernel.cutlass_fp4_group_mm` 在一个 C++ 函数里
  每次调用都做「8 个小 tensor 分配 + `__get_group_gemm_starts`<<<1,E>>> prologue +
  gemm_op.initialize + run」（见 sglang/sgl-kernel/csrc/moe/nvfp4_blockwise_moe.cu）。
  主 GEMM（GemmUniversal ptr-array kernel）的全部参数都从 device 数组读
  （a/b/out/scale/alpha 指针数组 + layout_sfa/sfb + strides + problem_sizes），
  所以把 op 拆成 prepare/run 两个入口即可单独跑主 kernel——需要 JIT 编译一个自定义变体
  （源码/cutlass headers 本地都有），现有二进制 wheel 不支持跳过 prologue。
- `__get_group_gemm_starts` 干的活 = cublas 侧指针更新的活 + 额外写每 expert 的
  LayoutSFA/SFB（依赖每 expert 的 m，routing 变了必须重算，两侧同理）。
- 口径含义：两侧「不可避免的非主 kernel 工作」语义对齐且都是 ~2µs 级（融合后），
  所以只记主 kernel 的对比结论可外推到框架端到端（前提：接入时 cublas 侧把 12 个
  python 小 kernel 融成 1 个；decode 小 batch 下未融合版会额外吃 ~20µs + 12 次发射）。

## nvfp4 GEMM benchmark 复现（2026-07-17）

脚本在 `benchmark/`，结果 CSV 在 `result/`（8 个类别各一个，CUTLASS/CUBLASLT 两后端同文件，
`backend` 列区分）。完整复现：

```bash
cd /mnt/3fs/dots-pretrain/daijiangkun/torch_matmul_bench/dots3_gemm_bench/benchmark
bash run_all.sh                       # 全量：6 个 dense + 2 个 group 类别
# 或者分开/单类别跑：
/opt/uv/bin/python bench_dense.py                          # 全部 dense 类别
/opt/uv/bin/python bench_dense.py o_proj kv_b_proj         # 指定类别
/opt/uv/bin/python bench_group.py                          # 全部 group 类别
/opt/uv/bin/python bench_group.py gate_up_proj_group       # 指定类别
```

前置条件：GB200 + `/opt/uv/bin/python`（torch 2.13 + sgl_kernel）+ `CUDA_HOME=/usr/local/cuda`
可用（group 的 cublasLt 路径首次运行 JIT 编译 ~30s，缓存于 ~/.cache/torch_extensions）。

### 方法（与 new_test_* run10_avg 约定一致）
- 计时：`bench_kineto(num_tests=10, flush_l2=True)`，CUPTI 纯 kernel 时间，**只计 GEMM 主
  kernel**（子串见上一章节表格）；每次调用前 8GB memset 冲 L2。
- `tflops = 2·m·n·k/t`（group 为 `2·sum_m·n·k/t`）；`gbps` = 操作数+scale+输出字节/t。
- `sm_mhz`/`power_w`：NVML 1ms 采样，对包住该次 bench_kineto 的墙钟窗口取平均
  （被 flush memset 稀释，不代表持续工作点，仅供参考）。
- 量化（计时外）：`scaled_fp4_quant`，per-tensor global scale；数据 randn bf16，seed=0。
- M 取值：decode 1,2,4,8,16,32,64,128,256；prefill 1024,2048,4096,8192,16384。

### shape 说明
- dense 6 类：fused_qkv_a_g_proj_with_mqa / fused_q_b_wq_b_proj / kv_b_proj / o_proj 各
  nsa+swa 两组；gate_up_proj_dense / down_proj_dense 各 tp1/tp4/tp8 三组
  （gate_up 切 N：27648/6912/3456；down 切 K：13824/3456/1728）。
- **偏差标注**：fused_qkv_a_g_proj_with_mqa 的 swa 原始给的是 [2176, **5210**]，
  K=5210 不是 16 的倍数，nvfp4 无法量化（K 方向每 16 元素一个 block scale），
  按笔误处理为 [2176, **5120**]。如需其他值请告知重跑。
- group 2 类：E=257（256 routed + 1 shared 融合为 expert id 256），topk=9，sum_m=T·9。
  routing 模型：每 token 从 256 个 routed expert 中随机选 8 个**不同**的 + shared expert
  必选（即 shared expert 的 m=T，routed expert 平均 m≈T·8/256）。
  gate_up 切 N：3072/768/384；down 切 K：1536/384/192。
- group 语义对齐 cublasLt：所有 expert 权重共用一个 per-tensor global scale
  （per-expert alpha 全相等）；每个测试点两后端先在同一份量化 buffer 上比对 bitwise
  相等再计时。
- group 的 cublasLt plan（heuristic 选 algo）按测试点重建（`reset_cublaslt_plans`），
  避免 T sweep 复用第一个 T 的 heuristic hint。
- group 的 gbps 权重字节只计**活跃 expert**（m>0）：两侧 kernel 都会跳过 m=0 的组，
  小 T 下若按全部 257 份权重计会算出超过 HBM 带宽的假数字。
- group 的 cublasLt 侧：**固定使用 heuristic 的 algo[0]**（既定约定，与框架接入行为
  一致；不做 algo 搜索）。
- cutlass 侧无任何运行时配置选择：dense 按输出 dtype 编译期定死
  （bf16/fp16→256×256×256 tile/2SM/cluster4×4，fp32→128×128×256/1SM/cluster1×4），
  group 单一配置（128×128×128/1SM/cluster1×1）；op 接口不暴露 tile/cluster 选项。
  即两后端在本 bench 中都是"每次调用一个确定 kernel 配置"。

### group 稳定性问题：根因与修复（排查记录）
症状（均**非确定性**、依赖分配器历史、隔离/CUDA_LAUNCH_BLOCKING 下不复现）：个别点两
后端输出 bitwise 不一致（坏行随 T 增多）、偶发 CUDA illegal memory access /
"operation not supported on global/shared address space"（异步报错，污染整个 context）。

**根因**（backend_test/fp4_utils.py:95 已记录同一 bug 的 workaround）：sgl_kernel
`scaled_fp4_experts_quant` 底层 cvt_fp16_to_fp4（E≥16 的 register-chunk 变体）按 16 个
int32 的块读取 expert_offsets/blockscale_offsets，当 (E+1)%16 != 1 时**越界读过数组尾部**
（dots3 E=257 → 258%16=2 命中）。越界读到的垃圾若恰好构成包含某 rowIdx 的区间，该行被
归给幻影 expert → **blockscale 野写**到垃圾地址 + 正确 scale 槽位残留脏数据 → 输出错行、
内存损坏、偶发 illegal access。新鲜页（驱动清零）时垃圾恒为 0 永不匹配 → 隔离不复现。

**修复**：offsets 分配在尾部多 16 个 int32 且**清零**的 buffer 里（bench_lib._padded_i32，
与 backend_test.padded_offsets 相同）。防御层：① scale buffer 零初始化
（_experts_quant_zeroed）；② 每点 destroy 旧 cublasLt plan + 清零共享 workspace；
③ variant 级子进程 + runner 按缺失点重试（BENCH_GROUP_MAX_ATTEMPTS，默认 10；幂等补缺）；
④ 每点双后端 bitwise 比对 + 对 fp32 参考的 relerr 守护（randn 数据 nvfp4 预期 ~0.134）。

**对 sglang serving 的提示**：serving 路径（cutlass_moe_params 的 torch.empty offsets +
prepare_moe_input）与本 bench 修复前的构造相同，E=257 时同样暴露于该 bug；CUDA graph 下
offsets 尾部内存在 capture 时固定，垃圾是否有害取决于运气且**恒定复现**。建议上游修复
（offsets 零填充或修 kernel）。

## 关键事实

- torch scaled-mm 的后端不可选：dense (`_scaled_mm_v2`) 一律走 cuBLASLt（nvjet），
  grouped (`_scaled_grouped_mm_v2`) 一律走 CUTLASS（cutlass_mslk），fp4 grouped 不支持。
  所以「cublas group」两个 kernel 只能自写 cuBLASLt C++（API 已验证可用）。
- sgl_kernel 0.3.6.post2 只有 nvfp4 (dense+group) 和 blockwise-128 fp8，没有任何 mx 格式 op。
- scale 布局三方通用：128×4 padding 的 32×4×4 swizzle（sgl_kernel `scaled_fp4_quant` 的输出
  可直接喂给 `torch._scaled_mm_v2`，两后端数值逐位量化一致）。
- nvfp4 的 per-tensor global scale 不参与 `_scaled_mm_v2` 调用，需在外面除掉
  （dense probe 里 `out / (gs_a*gs_b)`）；cutlass_scaled_fp4_mm 则通过 alpha 传入。
- cuBLASLt grouped 限制（见 sglang cublaslt_group_gemm 注释）：alpha 是每次调用单个标量，
  per-expert alpha 不同时需按 per-layer global scale 量化。
