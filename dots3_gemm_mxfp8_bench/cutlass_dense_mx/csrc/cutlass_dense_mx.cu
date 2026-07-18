// CUTLASS dense MXFP8 (mxfp8 x mxfp8 -> bf16) GEMM for SM100a, callable from torch.
// Structure cloned from sgl_kernel's nvfp4_scaled_mm_kernels.cu (which has no mx op);
// types per cutlass examples/72_blackwell_narrow_precision_gemm.
//
// D[M, N] = alpha * dequant(A[M, K]) @ dequant(B[N, K])^T, fp32 accumulate, bf16 out.
// A/B: e4m3 row-major [M,K] / [N,K]; SFA/SFB: e8m0 per 32 elems along K, swizzled
// 128x4 (32x4x4) layout (torchao to_blocked); alpha: fp32 scalar tensor (mxfp8 has
// no global scale -> callers pass 1.0).
//
// Two compiled configs, both exposed (the benchmark picks ONE per the fixed-config
// convention after a one-time measured comparison; there is no runtime heuristic):
//   mm_2sm: MmaTileShape 256x256x128, TmaWarpSpecialized2Sm, preferred cluster (4,4)
//           fallback (2,1)   [mirrors the nvfp4 sgl_kernel default config]
//   mm_1sm: MmaTileShape 128x128x128, TmaWarpSpecialized1Sm, preferred cluster (1,1)

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

// clang-format off
#include "cutlass/cutlass.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"
// clang-format on

#define CUTLASS_CHECK(status)                                                       \
  {                                                                                 \
    cutlass::Status error = status;                                                 \
    TORCH_CHECK(error == cutlass::Status::kSuccess, cutlassGetStatusString(error)); \
  }

using namespace cute;

namespace {

struct Traits2Sm {
  using MmaTileShape = Shape<_256, _256, _128>;
  using EpilogueTile = Shape<_128, _64>;
  using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized2Sm;
  using MainloopSchedule = cutlass::gemm::KernelTmaWarpSpecialized2SmMxf8f6f4Sm100;
  static dim3 cluster() { return dim3(4, 4, 1); }
  static dim3 cluster_fallback() { return dim3(2, 1, 1); }
};

struct Traits1Sm {
  using MmaTileShape = Shape<_128, _128, _128>;
  using EpilogueTile = cutlass::epilogue::collective::EpilogueTileAuto;
  using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized1Sm;
  using MainloopSchedule = cutlass::gemm::KernelTmaWarpSpecialized1SmMxf8f6f4Sm100;
  static dim3 cluster() { return dim3(1, 1, 1); }
  static dim3 cluster_fallback() { return dim3(1, 1, 1); }
};

template <typename Traits>
struct Mxf8GemmSm100 {
  using ElementA = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
  using LayoutATag = cutlass::layout::RowMajor;
  static constexpr int AlignmentA = 16;

  using ElementB = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
  using LayoutBTag = cutlass::layout::ColumnMajor;
  static constexpr int AlignmentB = 16;

  using ElementD = cutlass::bfloat16_t;
  using LayoutDTag = cutlass::layout::RowMajor;
  static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

  using ElementAccumulator = float;
  using ArchTag = cutlass::arch::Sm100;
  using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

  using MmaTileShape = typename Traits::MmaTileShape;
  using ClusterShape = Shape<int, int, _1>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag,
      cutlass::arch::OpClassTensorOp,
      MmaTileShape,
      ClusterShape,
      typename Traits::EpilogueTile,
      ElementAccumulator,
      ElementAccumulator,
      void,
      LayoutDTag,
      AlignmentD,
      ElementD,
      LayoutDTag,
      AlignmentD,
      typename Traits::EpilogueSchedule,
      cutlass::epilogue::fusion::LinearCombination<ElementD, float, void, float>>::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag,
      OperatorClass,
      ElementA,
      LayoutATag,
      AlignmentA,
      ElementB,
      LayoutBTag,
      AlignmentB,
      ElementAccumulator,
      MmaTileShape,
      ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename CollectiveEpilogue::SharedStorage))>,
      typename Traits::MainloopSchedule>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::
      GemmUniversal<Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, void>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideD = typename Gemm::GemmKernel::StrideD;
};

template <typename Traits>
void run_gemm(
    torch::Tensor& D,
    torch::Tensor const& A,
    torch::Tensor const& B,
    torch::Tensor const& A_sf,
    torch::Tensor const& B_sf,
    torch::Tensor const& alpha,
    int m,
    int n,
    int k,
    cudaStream_t stream) {
  using GemmT = Mxf8GemmSm100<Traits>;
  using ElementSF = cutlass::float_ue8m0_t;
  using Sm1xxBlkScaledConfig =
      typename GemmT::Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

  auto stride_A = cutlass::make_cute_packed_stride(typename GemmT::StrideA{}, {m, k, 1});
  auto stride_B = cutlass::make_cute_packed_stride(typename GemmT::StrideB{}, {n, k, 1});
  auto stride_D = cutlass::make_cute_packed_stride(typename GemmT::StrideD{}, {m, n, 1});
  auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(m, n, k, 1));
  auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(m, n, k, 1));

  typename GemmT::Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {m, n, k, 1},
      {static_cast<typename GemmT::Gemm::ElementA const*>(A.data_ptr()),
       stride_A,
       static_cast<typename GemmT::Gemm::ElementB const*>(B.data_ptr()),
       stride_B,
       static_cast<ElementSF const*>(A_sf.data_ptr()),
       layout_SFA,
       static_cast<ElementSF const*>(B_sf.data_ptr()),
       layout_SFB},
      {{},
       static_cast<typename GemmT::ElementD const*>(D.data_ptr()),
       stride_D,
       static_cast<typename GemmT::ElementD*>(D.data_ptr()),
       stride_D}};
  arguments.epilogue.thread.alpha_ptr = static_cast<float const*>(alpha.data_ptr());
  arguments.hw_info.cluster_shape = Traits::cluster();
  arguments.hw_info.cluster_shape_fallback = Traits::cluster_fallback();

  typename GemmT::Gemm gemm;
  size_t workspace_size = GemmT::Gemm::get_workspace_size(arguments);
  auto workspace = torch::empty(
      static_cast<int64_t>(workspace_size),
      torch::TensorOptions().dtype(torch::kUInt8).device(A.device()));
  CUTLASS_CHECK(gemm.can_implement(arguments));
  CUTLASS_CHECK(gemm.initialize(arguments, workspace.data_ptr(), stream));
  CUTLASS_CHECK(gemm.run(arguments, workspace.data_ptr(), stream));
}

int round_up_i(int x, int y) {
  return (x + y - 1) / y * y;
}

template <typename Traits>
torch::Tensor mxfp8_mm(
    torch::Tensor const& A,
    torch::Tensor const& B,
    torch::Tensor const& A_sf,
    torch::Tensor const& B_sf,
    torch::Tensor const& alpha) {
  TORCH_CHECK(A.is_cuda() && A.is_contiguous() && A.scalar_type() == at::ScalarType::Float8_e4m3fn, "a: e4m3 contiguous");
  TORCH_CHECK(B.is_cuda() && B.is_contiguous() && B.scalar_type() == at::ScalarType::Float8_e4m3fn, "b: e4m3 contiguous");
  TORCH_CHECK(A_sf.is_cuda() && A_sf.is_contiguous() && A_sf.scalar_type() == at::ScalarType::Float8_e8m0fnu, "scale_a: e8m0 contiguous");
  TORCH_CHECK(B_sf.is_cuda() && B_sf.is_contiguous() && B_sf.scalar_type() == at::ScalarType::Float8_e8m0fnu, "scale_b: e8m0 contiguous");
  TORCH_CHECK(alpha.scalar_type() == at::ScalarType::Float, "alpha: fp32");
  TORCH_CHECK(A.dim() == 2 && B.dim() == 2 && A.size(1) == B.size(1), "shape mismatch");
  int m = static_cast<int>(A.size(0));
  int n = static_cast<int>(B.size(0));
  int k = static_cast<int>(A.size(1));
  TORCH_CHECK(k % 32 == 0 && n % 16 == 0, "k must be /32, n /16");
  int rounded_k = round_up_i(k / 32, 4);
  TORCH_CHECK(A_sf.numel() == (int64_t)round_up_i(m, 128) * rounded_k, "scale_a swizzled numel mismatch");
  TORCH_CHECK(B_sf.numel() == (int64_t)round_up_i(n, 128) * rounded_k, "scale_b swizzled numel mismatch");

  at::cuda::CUDAGuard device_guard{(char)A.get_device()};
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(A.get_device());
  auto D = torch::empty({m, n}, torch::TensorOptions().dtype(torch::kBFloat16).device(A.device()));
  run_gemm<Traits>(D, A, B, A_sf, B_sf, alpha, m, n, k, stream);
  return D;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mm_2sm", &mxfp8_mm<Traits2Sm>, "mxfp8 dense gemm, 256x256x128 / 2SM / cluster(4,4)");
  m.def("mm_1sm", &mxfp8_mm<Traits1Sm>, "mxfp8 dense gemm, 128x128x128 / 1SM / cluster(1,1)");
}
