// cuBLASLt grouped MXFP8 (mxfp8 x mxfp8 -> bf16) group GEMM, callable from torch.
// Clone of the NVFP4 extension in dots3_gemm_nvfp4_bench/cublaslt_group_gemm
// (itself from sglang commit dc9627509) with the narrow-precision type swapped:
//   data   CUDA_R_4F_E2M1              -> CUDA_R_8F_E4M3
//   scales VEC16_UE4M3 (e4m3 per 16)   -> VEC32_UE8M0 (e8m0 per 32)
// and alpha fixed semantics unchanged (single host scalar per call; for mxfp8
// there is no per-tensor global scale, callers pass alpha = 1.0).
//
// Semantics per group e:
//   D_e[m_e, N] = alpha * dequant(A_e[m_e, K]) @ dequant(W_e[N, K])^T
// with e8m0 scales per 32 elems along K in the swizzled 128x4 (32x4x4) layout,
// fp32 accumulate, bf16 out.
//
// Mapping onto cublasLt column-major convention (per group):
//   cublas_m = N (weight rows), cublas_n = m_e (tokens), cublas_k = K
//   A(cublas) = weight  [K, N]  col-major (ld=K), transa=T
//   B(cublas) = activation [K, m_e] col-major (ld=K), transb=N
//   D(cublas) = [N, m_e] col-major (ld=N)  == row-major [m_e, N] contiguous
//
// Dynamic-m contract (verified for the fp4 twin by backend_test/verify_dynamic_m.py):
// dims (karr/nwarr/marr) and pointer arrays live in DEVICE memory and are read at
// kernel EXECUTION time; create_plan runs once (heuristic included), each step only
// rewrites array CONTENTS device-side, run_plan just enqueues. m_e = 0 groups skip.

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

#define CHECK_CUBLAS(expr)                                                                       \
  do {                                                                                           \
    cublasStatus_t s__ = (expr);                                                                 \
    TORCH_CHECK(s__ == CUBLAS_STATUS_SUCCESS, "cublasLt error ", st_name(s__), " at ", #expr);   \
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
  // keep-alive device tensors (contents may be rewritten between runs)
  torch::Tensor w_ptrs, x_ptrs, out_ptrs, w_sc_ptrs, x_sc_ptrs;
  torch::Tensor karr, nwarr, marr;
  torch::Tensor workspace;  // shared across plans; gemms on one stream are serial
  size_t ws_size = 0;
};

std::unordered_map<int64_t, Plan> g_plans;
int64_t g_next_id = 1;

void check_i64_cuda(const torch::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_cuda() && t.dtype() == torch::kInt64 && t.is_contiguous(), name, " must be contiguous int64 CUDA tensor");
}
void check_i32_cuda(const torch::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_cuda() && t.dtype() == torch::kInt32 && t.is_contiguous(), name, " must be contiguous int32 CUDA tensor");
}

void build_grouped_descs(
    Plan& p,
    int batch,
    int64_t avg_n_tokens,
    int64_t n_val,
    int64_t k_val,
    cublasLtMatmulPreference_t& pref) {
  cublasOperation_t transa = CUBLAS_OP_T, transb = CUBLAS_OP_N;
  cublasLtMatmulMatrixScale_t vec32 = CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;

  CHECK_CUBLAS(cublasLtMatmulDescCreate(&p.op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
  CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));
  CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &vec32, sizeof(vec32)));
  CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &vec32, sizeof(vec32)));
  const void* a_sc = reinterpret_cast<const void*>(p.w_sc_ptrs.data_ptr());
  const void* b_sc = reinterpret_cast<const void*>(p.x_sc_ptrs.data_ptr());
  CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &a_sc, sizeof(a_sc)));
  CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &b_sc, sizeof(b_sc)));

  const void* karr = p.karr.data_ptr();
  const void* nwarr = p.nwarr.data_ptr();
  const void* marr = p.marr.data_ptr();
  // A: weight, transa=T -> stored rows=k, cols=cublas_m(N)
  CHECK_CUBLAS(cublasLtGroupedMatrixLayoutCreate(&p.A, CUDA_R_8F_E4M3, batch, karr, nwarr, karr));
  // B: activation, transb=N -> rows=k, cols=cublas_n(m_e)
  CHECK_CUBLAS(cublasLtGroupedMatrixLayoutCreate(&p.B, CUDA_R_8F_E4M3, batch, karr, marr, karr));
  CHECK_CUBLAS(cublasLtGroupedMatrixLayoutCreate(&p.C, CUDA_R_16BF, batch, nwarr, marr, nwarr));
  CHECK_CUBLAS(cublasLtGroupedMatrixLayoutCreate(&p.D, CUDA_R_16BF, batch, nwarr, marr, nwarr));

  CHECK_CUBLAS(cublasLtMatmulPreferenceCreate(&pref));
  int64_t avg_rows = n_val;          // D rows = N (constant across groups)
  int64_t avg_cols = avg_n_tokens;   // D cols = avg m_e
  int64_t avg_k = k_val;
  CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_GROUPED_DESC_D_AVERAGE_ROWS, &avg_rows, sizeof(avg_rows)));
  CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_GROUPED_DESC_D_AVERAGE_COLS, &avg_cols, sizeof(avg_cols)));
  CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_GROUPED_AVERAGE_REDUCTION_DIM, &avg_k, sizeof(avg_k)));
  CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &p.ws_size, sizeof(p.ws_size)));
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

int64_t cublaslt_version() {
  return static_cast<int64_t>(cublasLtGetVersion());
}

// Create a grouped mxfp8 matmul plan. Heuristic runs here (once); run_plan only enqueues.
// All *_ptrs are int64 device tensors [G] holding raw device addresses; karr/nwarr/marr
// are int32 device tensors [G]. Their CONTENTS may be rewritten between runs (device-side).
// `workspace` is a uint8 CUDA tensor and may be shared by every plan on the stream.
int64_t create_plan(
    torch::Tensor w_ptrs, torch::Tensor x_ptrs, torch::Tensor out_ptrs,
    torch::Tensor w_sc_ptrs, torch::Tensor x_sc_ptrs,
    torch::Tensor karr, torch::Tensor nwarr, torch::Tensor marr,
    int64_t n_val, int64_t k_val, int64_t avg_m_tokens,
    double alpha, torch::Tensor workspace) {
  check_i64_cuda(w_ptrs, "w_ptrs"); check_i64_cuda(x_ptrs, "x_ptrs"); check_i64_cuda(out_ptrs, "out_ptrs");
  check_i64_cuda(w_sc_ptrs, "w_sc_ptrs"); check_i64_cuda(x_sc_ptrs, "x_sc_ptrs");
  check_i32_cuda(karr, "karr"); check_i32_cuda(nwarr, "nwarr"); check_i32_cuda(marr, "marr");
  TORCH_CHECK(workspace.is_cuda() && workspace.dtype() == torch::kUInt8, "workspace must be uint8 CUDA tensor");
  int batch = static_cast<int>(w_ptrs.size(0));
  TORCH_CHECK(x_ptrs.size(0) == batch && out_ptrs.size(0) == batch && marr.size(0) == batch, "group count mismatch");

  Plan p;
  p.alpha = static_cast<float>(alpha);
  p.beta = 0.0f;
  p.w_ptrs = w_ptrs; p.x_ptrs = x_ptrs; p.out_ptrs = out_ptrs;
  p.w_sc_ptrs = w_sc_ptrs; p.x_sc_ptrs = x_sc_ptrs;
  p.karr = karr; p.nwarr = nwarr; p.marr = marr;
  p.workspace = workspace;
  p.ws_size = static_cast<size_t>(workspace.numel());

  cublasLtMatmulPreference_t pref = nullptr;
  build_grouped_descs(p, batch, avg_m_tokens, n_val, k_val, pref);

  constexpr int kMaxAlgos = 8;
  int returned = 0;
  cublasLtMatmulHeuristicResult_t heur[kMaxAlgos] = {};
  cublasStatus_t hs =
      cublasLtMatmulAlgoGetHeuristic(get_handle(), p.op, p.A, p.B, p.C, p.D, pref, kMaxAlgos, heur, &returned);
  cublasLtMatmulPreferenceDestroy(pref);
  if (hs != CUBLAS_STATUS_SUCCESS || returned == 0) {
    destroy_plan_objs(p);
    TORCH_CHECK(false, "cublasLt grouped mxfp8 heuristic failed: ", st_name(hs), ", returned=", returned);
  }
  for (int i = 0; i < returned; ++i) {
    p.algos.push_back(heur[i].algo);
  }
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
  CHECK_CUBLAS(cublasLtMatmul(
      get_handle(), p.op, &p.alpha,
      reinterpret_cast<const void*>(p.w_ptrs.data_ptr()), p.A,
      reinterpret_cast<const void*>(p.x_ptrs.data_ptr()), p.B,
      &p.beta,
      reinterpret_cast<const void*>(p.out_ptrs.data_ptr()), p.C,
      reinterpret_cast<void*>(p.out_ptrs.data_ptr()), p.D,
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
  m.def("create_plan", &create_plan, "create grouped mxfp8 matmul plan (heuristic runs here)");
  m.def("run_plan", &run_plan, "enqueue grouped matmul on current stream (cuda-graph capturable)");
  m.def("num_algos", &num_algos, "number of heuristic algos cached in plan");
  m.def("set_algo", &set_algo, "select active algo index for run_plan");
  m.def("destroy_plan", &destroy_plan, "destroy plan");
}
