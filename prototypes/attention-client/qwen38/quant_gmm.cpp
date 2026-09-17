// Actual-count INT8 expert GEMM. Reuse CANN CATLASS tile/load/MMAD scheduling;
// keep INT32 accumulators explicit for per-token/per-channel dequantization.
// clang-format off
#include "kernel_operator.h"
#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm/kernel/grouped_matmul_slice_m.hpp"
// clang-format on
using namespace Catlass;
using namespace AscendC;

// cfg: K,N,groups,weight address,group-end address,input capacity.
__aicore__ inline void QuantGmm(GM_ADDR config, GM_ADDR input, GM_ADDR output,
                                uint64_t weightAddress = 0) {
  auto cfg = (__gm__ int64_t *)config;
  uint32_t k = cfg[0], n = cfg[1], groups = cfg[2], capacity = cfg[5];
  using Policy = Gemm::MmadAtlasA2PreloadAsync<1, 2, 2, 2, 1, false, true>;
  using A = Gemm::GemmType<int8_t, layout::RowMajor>;
  using B = Gemm::GemmType<int8_t, layout::zN>;
  using C = Gemm::GemmType<int32_t, layout::RowMajor>;
  using Tile = Gemm::Block::BlockMmad<Policy, GemmShape<128, 256, 256>,
                                      GemmShape<128, 256, 64>, A, B, C>;
  using Schedule = Gemm::Block::GemmIdentityBlockSwizzle<9, 1>;
  using Kernel =
      Gemm::Kernel::GroupedMatmulSliceM<Tile, void, Schedule, int64_t>;
  typename Kernel::Params params{
      GemmCoord{capacity, n, k},
      groups,
      (GM_ADDR)cfg[4],
      input,
      layout::RowMajor{capacity, k},
      (GM_ADDR)(weightAddress ? weightAddress : cfg[3]),
      layout::zN::MakeLayout<int8_t>(k, n),
      output,
      layout::RowMajor{capacity, n}};
  Kernel kernel;
  kernel(params);
}
extern "C" __global__ __aicore__ void quant_gmm(GM_ADDR config, GM_ADDR input,
                                                GM_ADDR output) {
  QuantGmm(config, input, output);
}
static const struct FunLevelKType quant_gmm_meta
    __attribute__((used, section(".ascend.meta.quant_gmm"))) = {
        {F_TYPE_KTYPE, sizeof(unsigned int), K_TYPE_AIC}};
