#include "actual_gmm.cpp"
#include "persistent_protocol.hpp"
#include "streaming_gmm.hpp"
// cfg: control,slot_pointer_table,out0,out1,source0,source1,tasks,owner,
//      trace,poll_limit,enabled. Each slot table is16 int64 pointers.
extern "C" __global__ __aicore__ void
persistent_cube(GM_ADDR config, GM_ADDR unused, GM_ADDR unused2) {
  using namespace Persistent;
  auto cfg = (__gm__ int64_t *)config;
  if (!cfg[10])
    return;
  auto ctrl = (__gm__ int32_t *)cfg[0];
  auto slots = (__gm__ int64_t *)cfg[1];
  int seen = 0, idle = 0;
  while (!Load(ctrl + STOP * LINE)) {
    int next = Load(ctrl + CCMD * LINE);
    if (next == seen) {
      if (++idle >= cfg[9]) {
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
    auto ptr = slots + slot * 16;
    int part = ctrl[CCMD * LINE + 3];
    if (part < 0 || part > (cfg[14] ? 1 : 0)) {
      Store(ctrl + STOP * LINE, -13);
      break;
    }
    bool streaming = cfg[14] == 2;
    int catalog = cfg[14] && !streaming ? 9 + part : 6;
    // Freeze both catalogs for the complete slot lifetime. Invalidate scalar
    // cache before reading device-authored counts from the other engine.
    for (int offset = 0; offset < 256; offset += LINE)
      Refresh((__gm__ int32_t *)ptr[catalog] + offset);
    int firstRow = 0;
    if (cfg[14] && (part || streaming)) {
      Refresh((__gm__ int32_t *)ptr[9] + 240);
      firstRow = ((__gm__ int64_t *)ptr[9])[127];
    }
    int configIndex = cfg[14] && !streaming ? (kind == 1 ? 11 : 13) + part
                                            : (kind == 1 ? 7 : 8);
    uint64_t begin = GetSystemCycle();
    if (streaming && kind == 1) {
      auto timing = cfg[13] ? (__gm__ int64_t *)cfg[13] +
                                  ((512 + next - 1) * 24 + GetBlockIdx()) * 8
                            : nullptr;
      RunStreamingUp((GM_ADDR)ptr[7], (GM_ADDR)ptr[1], (GM_ADDR)ptr[2],
                     firstRow, ctrl + (UP_PREFIX_DONE + GetBlockIdx()) * LINE,
                     next, timing);
    } else {
      if (streaming)
        firstRow = 0;
      RunActualGmm(
          (GM_ADDR)ptr[configIndex],
          (GM_ADDR)(ptr[kind == 1 ? 1 : 3] +
                    int64_t(firstRow) * (kind == 1 ? 2048 : 768) * 2),
          (GM_ADDR)(ptr[kind == 1 ? 2 : 4] +
                    int64_t(firstRow) * (kind == 1 ? 1536 : 2048) * 2));
    }
    PipeBarrier<PIPE_ALL>();
    WorkTime(cfg, 1, next, GetBlockIdx(), begin);
    seen = next;
    Store(ctrl + (CDONE + GetBlockIdx()) * LINE, seen);
  }
}
static const struct FunLevelKType persistent_cube_meta
    __attribute__((used, section(".ascend.meta.persistent_cube"))) = {
        {F_TYPE_KTYPE, sizeof(unsigned int), K_TYPE_AIC}};
