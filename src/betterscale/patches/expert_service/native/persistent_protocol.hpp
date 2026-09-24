#define QWEN_NEXT 1
// Local engine mailbox: one writer per 64-byte line, generation-tagged
// commands.
#pragma once
#include "kernel_operator.h"
namespace Persistent {
using namespace AscendC;
constexpr int LINE = 16, VW = 16, CW = 24;
#ifdef QWEN_NEXT
constexpr int HIDDEN = 2048, INNER = 512, TOPK = 8, EXPERTS = 256;
constexpr int LOCAL_EXPERTS = 256, LAYERS = 40, GROUPS = 256;
constexpr bool SINGLE_LAYER = true;
#else
constexpr int HIDDEN = 2048, INNER = 768, TOPK = 8, EXPERTS = 128;
constexpr int LOCAL_EXPERTS = 64, LAYERS = 2, GROUPS = 128;
constexpr bool SINGLE_LAYER = false;
#endif
constexpr int SOURCES = 7;
constexpr int TOKENS = 4096, ROUTES = TOKENS * TOPK;
constexpr int MAP = ROUTES + 8, CAPACITY = ROUTES * SOURCES;
constexpr int STOP = 0, VCMD = 1, CCMD = 2, VDONE = 3, CDONE = 19, STATUS = 43;
constexpr int UP_PREFIX_DONE = 44, URGENT_CMD = 68, URGENT_DONE = 69;
constexpr int ACT_TAIL_READY = 85; // two slot lines, coordinator-only writer
constexpr int PACK_EPOCH = 87;     // immutable pack command generation per slot
constexpr int DOWN_RANGE_READY = 89, DOWN_ALL_READY = 91;
constexpr int DOWN_PREFIX_DONE = 96; // 24 Cube-owned lines
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
  if (!cfg[13] || generation > 512)
    return;
  auto row = (__gm__ int64_t *)cfg[13] +
             ((engine * 512 + generation - 1) * 24 + core) * 8;
  row[0] = begin;
  row[1] = GetSystemCycle();
  Refresh((__gm__ int32_t *)row);
}
} // namespace Persistent

namespace Persistent {
__aicore__ inline int64_t SourcePointer(__gm__ int64_t *cfg, int source) {
  return ((__gm__ int64_t *)cfg[27])[source];
}
__aicore__ inline int64_t OutputPointer(__gm__ int64_t *cfg, int source) {
  return ((__gm__ int64_t *)cfg[28])[source];
}
__aicore__ inline int SumSources(const int *values) {
  int result = 0;
  for (int source = 0; source < SOURCES; ++source) result += values[source];
  return result;
}
__aicore__ inline bool SameLayer(const int *generations, const int *layers, int layer) {
  for (int source = 0; source < SOURCES; ++source)
    if (generations[source] && layers[source] != layer) return false;
  return true;
}
__aicore__ inline int DescriptorLayer(__gm__ int32_t *desc) {
  for (int source = 0; source < SOURCES; ++source) {
    Refresh(desc + source * MAP);
    if (desc[source * MAP]) return desc[source * MAP + 2];
  }
  return -1;
}
}
