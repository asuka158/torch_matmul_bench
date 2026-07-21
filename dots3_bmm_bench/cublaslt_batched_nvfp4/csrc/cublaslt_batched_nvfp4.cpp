// cuBLASLt STRIDED-BATCHED NVFP4 (nvfp4 x nvfp4 -> bf16) bmm, callable from torch.
//
// Target op (dots3 attention absorbed bmm, w_kc / w_vc):
//   D[b] = alpha * dequant(X[b] : [M,K]) @ dequant(W[b] : [N,K])^T      b = 0..B-1
// i.e. a UNIFORM batched GEMM: every batch has the same M/N/K, each batch has its
// own weight. Block scaling is VEC16_UE4M3 (fp8-e4m3 per 16 elems along K, swizzled
// 32x4x4 == sgl_kernel scaled_fp4_quant layout), fp32 accumulate, bf16 out.
//
// Mapping onto cublasLt column-major convention (per batch, same as the grouped ext):
//   cublas_m = N (weight rows), cublas_n = M (tokens), cublas_k = K
//   A(cublas) = weight     [K, N] col-major (ld=K), transa=T   <- row-major [N,K]
//   B(cublas) = activation [K, M] col-major (ld=K), transb=N   <- row-major [M,K]
//   D(cublas) = [N, M] col-major (ld=N)  == row-major [M, N] contiguous
// Batching via CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT + STRIDED_BATCH_OFFSET.
//
// NOTE: cublasLt exposes NO per-batch stride attribute for the block-scale tensors
// (there is A/B_SCALE_POINTER and A/B_SCALE_MODE only), so whether strided-batch is
// supported at all together with VEC16_UE4M3 -- and if so what scale-buffer layout it
// assumes -- is undocumented and must be established empirically. That is exactly what
// probe_bmm.py checks (numeric relerr vs an fp32 reference).

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <cublasLt.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace {

const char* st_name(cublasStatus_t s) {
  switch (s) {
    case CUBLAS_STATUS_SUCCESS: return "SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED: return "NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED: return "ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE: return "INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH: return "ARCH_MISMATCH";
    case CUBLAS_STATUS_EXECUTION_FAILED: return "EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR: return "INTERNAL_ERROR";
    case CUBLAS_STATUS_NOT_SUPPORTED: return "NOT_SUPPORTED";
    default: return "OTHER";
  }
}

#define CHECK_CUBLAS(expr)                                                                     \
  do {                                                                                         \
    cublasStatus_t s__ = (expr);                                                               \
    TORCH_CHECK(s__ == CUBLAS_STATUS_SUCCESS, "cublasLt error ", st_name(s__), " at ", #expr); \
  } while (0)

cublasLtHandle_t get_handle() {
  static cublasLtHandle_t handle = [] {
    cublasLtHandle_t h;
    TORCH_CHECK(cublasLtCreate(&h) == CUBLAS_STATUS_SUCCESS, "cublasLtCreate failed");
    return h;
  }();
  return handle;
}

struct Plan {
  cublasLtMatmulDesc_t op = nullptr;
  cublasLtMatrixLayout_t A = nullptr, B = nullptr, C = nullptr, D = nullptr;
  std::vector<cublasLtMatmulAlgo_t> algos;
  int active_algo = 0;
  float alpha = 1.0f, beta = 0.0f;
  int scale_mode = 0;  // see create_plan
  torch::Tensor w, x, out, w_sc, x_sc, alpha_vec, workspace;  // keep-alive
  size_t ws_size = 0;
};

std::unordered_map<int64_t, Plan> g_plans;
int64_t g_next_id = 1;

void set_batch(cublasLtMatrixLayout_t l, int32_t batch, int64_t stride) {
  CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(
      l, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch, sizeof(batch)));
  CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(
      l, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride, sizeof(stride)));
}

void destroy_plan_objs(Plan& p) {
  if (p.D) cublasLtMatrixLayoutDestroy(p.D);
  if (p.C) cublasLtMatrixLayoutDestroy(p.C);
  if (p.B) cublasLtMatrixLayoutDestroy(p.B);
  if (p.A) cublasLtMatrixLayoutDestroy(p.A);
  if (p.op) cublasLtMatmulDescDestroy(p.op);
  p.op = nullptr; p.A = p.B = p.C = p.D = nullptr;
}

}  // namespace

int64_t cublaslt_version() { return static_cast<int64_t>(cublasLtGetVersion()); }

// w     : [B, N, K/2] uint8  (nvfp4 packed weight, row-major per batch)
// x     : [B, M, K/2] uint8  (nvfp4 packed activation)
// out   : [B, M, N]   bf16
// w_sc  : swizzled e4m3 weight scales, B consecutive blocks of N*(K/16) bytes
// x_sc  : swizzled e4m3 act    scales, B consecutive blocks of ceil(M/128)*128*(K/16) bytes
// scale_mode selects how the per-tensor nvfp4 GLOBAL scale is applied (the per-16
// block scales always use VEC16_UE4M3 and are per-batch regardless):
//   0 = single host scalar alpha for the whole batched call   (per-TENSOR global scale)
//   1 = per-batch alpha via POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_HOST +
//       ALPHA_VECTOR_BATCH_STRIDE; alpha_vec is [B, N] (each batch's row = its alpha)
//   2 = per-batch D scale via D_SCALE_MODE = PER_BATCH_SCALAR_32F; alpha_vec is [B] fp32
// Modes 1/2 are the two documented routes to a per-BATCH global scale; whether either
// actually works together with VEC16_UE4M3 block scaling is what probe_per_batch_scale.py
// settles. (A_SCALE_MODE/B_SCALE_MODE = PER_BATCH_SCALAR_32F is NOT usable here: one
// mode per operand, and nvfp4 needs VEC16_UE4M3 there for the block scales.)
int64_t create_plan(torch::Tensor w, torch::Tensor x, torch::Tensor out,
                    torch::Tensor w_sc, torch::Tensor x_sc,
                    int64_t B, int64_t M, int64_t N, int64_t K,
                    double alpha, torch::Tensor alpha_vec, int64_t scale_mode,
                    torch::Tensor workspace) {
  TORCH_CHECK(w.is_cuda() && x.is_cuda() && out.is_cuda(), "tensors must be CUDA");
  TORCH_CHECK(w.is_contiguous() && x.is_contiguous() && out.is_contiguous(), "tensors must be contiguous");
  TORCH_CHECK(w_sc.is_contiguous() && x_sc.is_contiguous(), "scale tensors must be contiguous");
  TORCH_CHECK(out.dtype() == torch::kBFloat16, "out must be bf16");
  TORCH_CHECK(K % 16 == 0, "K must be a multiple of 16 for nvfp4 block scaling");

  Plan p;
  p.alpha = static_cast<float>(alpha);
  p.beta = 0.0f;
  p.scale_mode = static_cast<int>(scale_mode);
  p.w = w; p.x = x; p.out = out; p.w_sc = w_sc; p.x_sc = x_sc;
  p.alpha_vec = alpha_vec;
  p.workspace = workspace;
  p.ws_size = static_cast<size_t>(workspace.numel());

  cublasOperation_t transa = CUBLAS_OP_T, transb = CUBLAS_OP_N;
  cublasLtMatmulMatrixScale_t vec16 = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;

  CHECK_CUBLAS(cublasLtMatmulDescCreate(&p.op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
  CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));
  CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &vec16, sizeof(vec16)));
  CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &vec16, sizeof(vec16)));
  const void* a_sc = reinterpret_cast<const void*>(w_sc.data_ptr());
  const void* b_sc = reinterpret_cast<const void*>(x_sc.data_ptr());
  CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &a_sc, sizeof(a_sc)));
  CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &b_sc, sizeof(b_sc)));

  if (p.scale_mode == 1) {
    // per-batch alpha vector: length must equal D's row count (= N), strided per batch
    TORCH_CHECK(alpha_vec.is_cuda() && alpha_vec.dtype() == torch::kFloat32 &&
                    alpha_vec.is_contiguous() && alpha_vec.numel() == B * N,
                "scale_mode=1 needs a contiguous fp32 CUDA alpha_vec of [B, N]");
    cublasLtPointerMode_t pm = CUBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_HOST;
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_POINTER_MODE, &pm, sizeof(pm)));
    int64_t astride = N;
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        p.op, CUBLASLT_MATMUL_DESC_ALPHA_VECTOR_BATCH_STRIDE, &astride, sizeof(astride)));
  } else if (p.scale_mode == 2) {
    // per-batch scalar applied to D
    TORCH_CHECK(alpha_vec.is_cuda() && alpha_vec.dtype() == torch::kFloat32 &&
                    alpha_vec.is_contiguous() && alpha_vec.numel() == B,
                "scale_mode=2 needs a contiguous fp32 CUDA alpha_vec of [B]");
    cublasLtMatmulMatrixScale_t pbs = CUBLASLT_MATMUL_MATRIX_SCALE_PER_BATCH_SCALAR_32F;
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_D_SCALE_MODE, &pbs, sizeof(pbs)));
    const void* d_sc = reinterpret_cast<const void*>(alpha_vec.data_ptr());
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_D_SCALE_POINTER, &d_sc, sizeof(d_sc)));
  }

  // layouts: A=[K,N] ld=K (transa=T), B=[K,M] ld=K, C/D=[N,M] ld=N
  CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&p.A, CUDA_R_4F_E2M1, K, N, K));
  CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&p.B, CUDA_R_4F_E2M1, K, M, K));
  CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&p.C, CUDA_R_16BF, N, M, N));
  CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&p.D, CUDA_R_16BF, N, M, N));
  if (B > 1) {
    int32_t b32 = static_cast<int32_t>(B);
    set_batch(p.A, b32, K * N);  // strides in ELEMENTS
    set_batch(p.B, b32, K * M);
    set_batch(p.C, b32, N * M);
    set_batch(p.D, b32, N * M);
  }

  cublasLtMatmulPreference_t pref = nullptr;
  CHECK_CUBLAS(cublasLtMatmulPreferenceCreate(&pref));
  CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(
      pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &p.ws_size, sizeof(p.ws_size)));

  constexpr int kMaxAlgos = 8;
  int returned = 0;
  cublasLtMatmulHeuristicResult_t heur[kMaxAlgos] = {};
  cublasStatus_t hs = cublasLtMatmulAlgoGetHeuristic(
      get_handle(), p.op, p.A, p.B, p.C, p.D, pref, kMaxAlgos, heur, &returned);
  cublasLtMatmulPreferenceDestroy(pref);
  if (hs != CUBLAS_STATUS_SUCCESS || returned == 0) {
    destroy_plan_objs(p);
    TORCH_CHECK(false, "cublasLt batched nvfp4 heuristic failed: ", st_name(hs),
                ", returned=", returned, " (B=", B, " M=", M, " N=", N, " K=", K, ")");
  }
  for (int i = 0; i < returned; ++i) p.algos.push_back(heur[i].algo);
  p.active_algo = 0;

  int64_t id = g_next_id++;
  g_plans.emplace(id, std::move(p));
  return id;
}

int64_t num_algos(int64_t id) {
  auto it = g_plans.find(id);
  TORCH_CHECK(it != g_plans.end(), "unknown plan id ", id);
  return static_cast<int64_t>(it->second.algos.size());
}

void set_algo(int64_t id, int64_t idx) {
  auto it = g_plans.find(id);
  TORCH_CHECK(it != g_plans.end(), "unknown plan id ", id);
  TORCH_CHECK(idx >= 0 && idx < (int64_t)it->second.algos.size(), "algo idx out of range");
  it->second.active_algo = static_cast<int>(idx);
}

void run_plan(int64_t id) {
  auto it = g_plans.find(id);
  TORCH_CHECK(it != g_plans.end(), "unknown plan id ", id);
  Plan& p = it->second;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  // mode 1 passes a DEVICE alpha vector; modes 0/2 pass the host scalar
  const void* alpha_ptr = (p.scale_mode == 1)
                              ? reinterpret_cast<const void*>(p.alpha_vec.data_ptr())
                              : reinterpret_cast<const void*>(&p.alpha);
  CHECK_CUBLAS(cublasLtMatmul(
      get_handle(), p.op, alpha_ptr,
      p.w.data_ptr(), p.A,
      p.x.data_ptr(), p.B,
      &p.beta,
      p.out.data_ptr(), p.C,
      p.out.data_ptr(), p.D,
      &p.algos[p.active_algo], p.workspace.data_ptr(), p.ws_size, stream));
}

void destroy_plan(int64_t id) {
  auto it = g_plans.find(id);
  if (it == g_plans.end()) return;
  destroy_plan_objs(it->second);
  g_plans.erase(it);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("cublaslt_version", &cublaslt_version, "cublasLt library version");
  m.def("create_plan", &create_plan, "create strided-batched nvfp4 matmul plan");
  m.def("run_plan", &run_plan, "enqueue batched matmul on current stream");
  m.def("num_algos", &num_algos, "number of heuristic algos cached in plan");
  m.def("set_algo", &set_algo, "select active algo index");
  m.def("destroy_plan", &destroy_plan, "destroy plan");
}
