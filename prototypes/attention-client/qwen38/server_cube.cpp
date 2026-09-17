#include "actual_gmm.cpp"
#include "persistent_protocol.hpp"
#include "quant_gmm.cpp"
using namespace Persistent;
// First Qwen38 gate keeps whole up/down commands, not the BF16-only prefix
// pipeline. Coordinator priorities/two-slot overlap remain unchanged.
extern "C" __global__ __aicore__ void
persistent_cube(GM_ADDR config, GM_ADDR unused, GM_ADDR unused2) {
  auto cfg = (__gm__ int64_t *)config;
  if (!cfg[10])
    return;
  auto ctrl = (__gm__ int32_t *)cfg[0];
  int seen = 0, idle = 0;
  while (!Load(ctrl + STOP * LINE)) {
    int next = Load(ctrl + CCMD * LINE);
    if (next == seen) {
      if (!cfg[24] && ++idle >= cfg[9]) {
        Store(ctrl + STOP * LINE, -11);
        break;
      }
      continue;
    }
    idle = 0;
    int kind = ctrl[CCMD * LINE + 1], slot = ctrl[CCMD * LINE + 2];
    if (next != seen + 1 || kind < 1 || kind > 2 || slot < 0 || slot > 1) {
      Store(ctrl + STOP * LINE, -12);
      break;
    }
    auto ptr = (__gm__ int64_t *)cfg[1] + slot * 16;
    auto desc = (__gm__ int32_t *)ptr[5];
    Refresh(desc);
    Refresh(desc + MAP);
    int layer = desc[0] ? desc[2] : desc[MAP + 2];
    auto weights = (__gm__ int64_t *)cfg[25] + layer * 4;
    for (int offset = 0; offset < 256; offset += LINE)
      Refresh((__gm__ int32_t *)ptr[6] + offset);
    uint64_t begin = GetSystemCycle();
    if (layer < 48)
      QuantGmm((GM_ADDR)ptr[kind == 1 ? 7 : 8], (GM_ADDR)ptr[kind == 1 ? 1 : 3],
               (GM_ADDR)ptr[kind == 1 ? 2 : 4], weights[kind - 1]);
    else
      RunActualGmm((GM_ADDR)ptr[kind == 1 ? 7 : 8],
                   (GM_ADDR)ptr[kind == 1 ? 1 : 3],
                   (GM_ADDR)ptr[kind == 1 ? 2 : 4], weights[kind - 1]);
    PipeBarrier<PIPE_ALL>();
    WorkTime(cfg, 1, next, GetBlockIdx(), begin);
    seen = next;
    Store(ctrl + (CDONE + GetBlockIdx()) * LINE, seen);
  }
}
static const struct FunLevelKType persistent_cube_meta
    __attribute__((used, section(".ascend.meta.persistent_cube"))) = {
        {F_TYPE_KTYPE, sizeof(unsigned int), K_TYPE_AIC}};
