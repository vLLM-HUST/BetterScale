// Local engine mailbox: one writer per 64-byte line, generation-tagged
// commands.
#pragma once
#include "kernel_operator.h"
namespace Persistent {
using namespace AscendC;
constexpr int LINE = 16, VW = 16, CW = 24, MAP = 264, CAPACITY = 512;
constexpr int STOP = 0, VCMD = 1, CCMD = 2, VDONE = 3, CDONE = 19, STATUS = 43;
constexpr int UP_PREFIX_DONE = 44, URGENT_CMD = 68, URGENT_DONE = 69;
constexpr int ACT_TAIL_READY = 85; // two slot lines, coordinator-only writer
constexpr int PACK_EPOCH = 87;     // immutable pack command generation per slot
struct PackGate {
  __gm__ int32_t *ready;
  __gm__ int32_t *stop;
  int generation;
  int64_t pollLimit;
  __gm__ int64_t *issueTrace;
};
enum Stage {
  EMPTY,
  PULL,
  PACK,
  READY_UP,
  UP,
  READY_ACT,
  ACT,
  READY_DOWN,
  DOWN,
  READY_RETURN,
  RETURN
};
enum VectorTask { FETCH = 1, REPACK = 2, ACTIVATE = 3, SEND = 4 };
__aicore__ inline void Refresh(__gm__ int32_t *address) {
  GlobalTensor<int32_t> tensor;
  tensor.SetGlobalBuffer(address);
  __asm__ __volatile__("");
  DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE,
                           DcciDst::CACHELINE_OUT>(tensor);
  __asm__ __volatile__("");
}
__aicore__ inline int Load(__gm__ int32_t *address) {
  Refresh(address);
  return *reinterpret_cast<volatile __gm__ int32_t *>(address);
}
__aicore__ inline void Store(__gm__ int32_t *address, int value) {
  *reinterpret_cast<volatile __gm__ int32_t *>(address) = value;
  Refresh(address);
}
__aicore__ inline bool Joined(__gm__ int32_t *ctrl, int first, int cores,
                              int gen) {
  for (int i = 0; i < cores; ++i)
    if (Load(ctrl + (first + i) * LINE) != gen)
      return false;
  return true;
}
// Optional per-core work intervals. Each writer owns one entire64-byte line.
__aicore__ inline void WorkTime(__gm__ int64_t *cfg, int engine, int generation,
                                int core, uint64_t begin) {
  if (!cfg[13])
    return;
  auto row = (__gm__ int64_t *)cfg[13] +
             ((engine * 512 + generation - 1) * 24 + core) * 8;
  row[0] = begin;
  row[1] = GetSystemCycle();
  Refresh((__gm__ int32_t *)row);
}
} // namespace Persistent
