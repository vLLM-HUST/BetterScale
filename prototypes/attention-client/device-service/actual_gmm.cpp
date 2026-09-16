// Thin Ascend-only adapter to the installed CANN CATLASS grouped-GEMM kernel.
// CATLASS remains an external dependency under its CANN Open Software license.
// CANN headers require this order; they are not individually self-contained.
// clang-format off
#include "kernel_operator.h"
#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm/kernel/grouped_matmul_slice_m.hpp"
#if ACTUAL_GMM_DFC
#include "utils/const_args.hpp"
#include "utils/copy_gm_to_l1_custom.hpp"
#include "utils/block_mmad_preload_async_fixpipe_quant.hpp"
#endif
// clang-format on

#ifndef ACTUAL_GMM_TILE_M
#define ACTUAL_GMM_TILE_M 128
#endif
#ifndef ACTUAL_GMM_TILE_N
#define ACTUAL_GMM_TILE_N 256
#endif
#ifndef ACTUAL_GMM_TILE_K
#define ACTUAL_GMM_TILE_K 256
#endif

#ifndef ACTUAL_GMM_L0_K
#define ACTUAL_GMM_L0_K 64
#endif

using namespace Catlass;
using namespace AscendC;

#if ACTUAL_GMM_DFC
// Supply the BF16-unused quant-scale arguments to the upstream DFC tile engine.
// Scheduling, loads, MMAD and FIXPIPE stay entirely in the upstream template.
template <class Base> struct DfcBf16Tile : Base {
  __aicore__ inline DfcBf16Tile(
      Arch::Resource<typename Base::ArchTag> &resource)
      : Base(resource) {}
  __aicore__ inline void
  operator()(GlobalTensor<bfloat16_t> const &a, layout::RowMajor const &la,
             GlobalTensor<bfloat16_t> const &b, layout::zN const &lb,
             GlobalTensor<bfloat16_t> const &c, layout::RowMajor const &lc,
             GemmCoord const &shape) {
    GlobalTensor<uint64_t> unusedScale;
    Base::operator()(a, la, b, lb, c, lc, unusedScale,
                     layout::VectorLayout{shape.n()}, shape);
  }
};
#endif

// cfg: K,N,group_count,weight_address,group_list_address,capacity.
// The producer guarantees monotone ends in [0,capacity]. Only live rows are
// computed; inactive output rows are untouched, not synthetic expert work.
__aicore__ inline void RunActualGmm(GM_ADDR config, GM_ADDR input,
                                    GM_ADDR output) {
  auto cfg = (__gm__ int64_t *)config;
  uint32_t k = cfg[0], n = cfg[1], groups = cfg[2], capacity = cfg[5];
  GlobalTensor<int64_t> ends;
  ends.SetGlobalBuffer((__gm__ int64_t *)cfg[4]);
  uint32_t first = 0, last = groups;
  while (first < groups && ends.GetValue(first) == 0)
    ++first;
  if (first == groups)
    return;
  while (last > first + 1 && ends.GetValue(last - 1) == ends.GetValue(last - 2))
    --last;
  // Leading groups have zero prefix, so this is a zero-copy live catalog view;
  // group ends remain relative to the packed input. NZ expert stride is K*N
  // because this adapter admits only fully16-aligned Qwen matrices.
#if ACTUAL_GMM_DFC
  using Policy =
      Gemm::MmadAtlasA2PreloadAsyncFixpipe<1, 2, 2, 2, 1, false, true>;
#else
  using Policy = Gemm::MmadAtlasA2PreloadAsync<1, 2, 2, 2, 1, false, true>;
#endif
  using A = Gemm::GemmType<bfloat16_t, layout::RowMajor>;
  using B = Gemm::GemmType<bfloat16_t, layout::zN>;
  using C = Gemm::GemmType<bfloat16_t, layout::RowMajor>;
  using BaseMmad = Gemm::Block::BlockMmad<
      Policy,
      GemmShape<ACTUAL_GMM_TILE_M, ACTUAL_GMM_TILE_N, ACTUAL_GMM_TILE_K>,
      GemmShape<ACTUAL_GMM_TILE_M, ACTUAL_GMM_TILE_N, ACTUAL_GMM_L0_K>, A, B,
      C>;
#if ACTUAL_GMM_DFC
  using Mmad = DfcBf16Tile<BaseMmad>;
#else
  using Mmad = BaseMmad;
#endif
  using Scheduler = Gemm::Block::GemmIdentityBlockSwizzle<9, 1>;
  using Kernel =
      Gemm::Kernel::GroupedMatmulSliceM<Mmad, void, Scheduler, int64_t>;
  typename Kernel::Params params{
      GemmCoord{capacity, n, k},
      last - first,
      (GM_ADDR)(cfg[4] + first * 8),
      input,
      layout::RowMajor{capacity, k},
      (GM_ADDR)(cfg[3] + uint64_t(first) * k * n * 2),
      layout::zN::MakeLayout<bfloat16_t>(k, n),
      output,
      layout::RowMajor{capacity, n}};
  Kernel kernel;
  kernel(params);
}
extern "C" __global__ __aicore__ void actual_gmm(GM_ADDR config, GM_ADDR input,
                                                 GM_ADDR output) {
  RunActualGmm(config, input, output);
}
static const struct FunLevelKType actual_gmm_meta
    __attribute__((used, section(".ascend.meta.actual_gmm"))) = {
        {F_TYPE_KTYPE, sizeof(unsigned int), K_TYPE_AIC}};
