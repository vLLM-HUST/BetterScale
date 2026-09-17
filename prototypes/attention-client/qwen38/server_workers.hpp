#pragma once
#include "persistent_protocol.hpp"
#include "quant_vector.hpp"
using namespace AscendC;
using namespace Persistent;
__aicore__ inline int ScalarMin(int a, int b) { return a < b ? a : b; }
__aicore__ inline int ScalarMax(int a, int b) { return a > b ? a : b; }
struct Transfer {
  TPipe pipe;
  TBuf<TPosition::VECCALC> buf;
  LocalTensor<int32_t> words;
  __aicore__ inline void Init() {
    pipe.InitBuffer(buf, 65536);
    words = buf.Get<int32_t>();
  }
  __aicore__ inline void Read(__gm__ int32_t *p, int n) {
    GlobalTensor<int32_t> g;
    g.SetGlobalBuffer(p);
    DataCopy(words, g, n);
    SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
  }
  __aicore__ inline void Write(__gm__ int32_t *p, int n) {
    SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
    GlobalTensor<int32_t> g;
    g.SetGlobalBuffer(p);
    DataCopy(g, words, n);
    SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
  }
  __aicore__ inline int Flag(__gm__ int32_t *p) {
    Read(p, 8);
    return words.GetValue(0);
  }
  __aicore__ inline void Publish(__gm__ int32_t *p, int gen) {
    for (int i = 0; i < 8; ++i)
      words.SetValue(i, i ? 0 : gen);
    Write(p, 8);
  }
  __aicore__ inline void Copy(__gm__ int32_t *src, __gm__ int32_t *dst, int n) {
    Read(src, n);
    Write(dst, n);
  }
};
__aicore__ inline int RowExpert(__gm__ int64_t *ends, int row) {
  int e = 0;
  while (e < 128 && row >= ends[e])
    ++e;
  return e;
}
// Slot+15 points to [fetched scales,packed scales,activation scales]. Scales
// have one32-byte row per token; no two movers write the same cache line.
// Target input was quantized by native DynamicQuant on the attention device.
__aicore__ inline void Worker(__gm__ int64_t *cfg, Transfer &io) {
  auto ctrl = (__gm__ int32_t *)cfg[0];
  int seen = 0, idle = 0, worker = GetBlockIdx() - 1;
  QuantRow quant(io.words);
  while (!Load(ctrl + STOP * LINE)) {
    int next = Load(ctrl + VCMD * LINE);
    if (next == seen) {
      if (!cfg[24] && ++idle >= cfg[9]) {
        Store(ctrl + STOP * LINE, -21);
        break;
      }
      continue;
    }
    idle = 0;
    int kind = ctrl[VCMD * LINE + 1], slot = ctrl[VCMD * LINE + 2],
        extra = ctrl[VCMD * LINE + 3];
    if (next != seen + 1 || slot < 0 || slot > 1 || kind < 1 || kind > 4) {
      Store(ctrl + STOP * LINE, -22);
      break;
    }
    auto ptr = (__gm__ int64_t *)cfg[1] + slot * 16;
    auto auxiliary = (__gm__ int64_t *)ptr[15];
    auto ends = (__gm__ int64_t *)ptr[6];
    auto desc = (__gm__ int32_t *)ptr[5];
    Refresh(desc);
    Refresh(desc + MAP);
    int layer = desc[0] ? desc[2] : desc[MAP + 2];
    bool int8 = layer < 48;
    auto weights = (__gm__ int64_t *)cfg[25] + layer * 4;
    for (int offset = 0; offset < 256; offset += LINE)
      Refresh((__gm__ int32_t *)ends + offset);
    uint64_t begin = GetSystemCycle();
    if (kind == ACTIVATE) {
      for (int row = worker; row < extra; row += VW) {
        if (int8) {
          int expert = RowExpert(ends, row);
          quant.Dequant((__gm__ int32_t *)ptr[2] + row * INNER * 2,
                        (__gm__ float *)weights[2] + expert * INNER * 2,
                        (__gm__ float *)auxiliary[1] + row * 8, INNER * 2);
          quant.Swiglu(INNER);
          quant.Quantize((__gm__ int8_t *)ptr[3] + row * INNER,
                         (__gm__ float *)auxiliary[2] + row * 8, INNER);
        } else {
          quant.FromBf16((__gm__ bfloat16_t *)ptr[2] + row * INNER * 2,
                         INNER * 2);
          quant.Swiglu(INNER);
          quant.ToBf16((__gm__ bfloat16_t *)ptr[3] + row * INNER, INNER);
        }
      }
    } else {
      for (int c = 0; c < 2; ++c) {
        io.Read(desc + c * MAP, MAP);
        int gen = io.words.GetValue(0), n = io.words.GetValue(1), map[ROUTES];
        for (int i = 0; i < ROUTES; ++i)
          map[i] = io.words.GetValue(8 + i);
        if (!gen)
          continue;
        if (kind == FETCH) {
          if (!(extra & (1 << c)))
            continue;
          int width = HIDDEN / (int8 ? 4 : 2);
          for (int row = worker; row < n; row += VW) {
            io.Copy((__gm__ int32_t *)cfg[4 + c] + 1024 + row * width,
                    (__gm__ int32_t *)ptr[0] + (c * 32 + row) * width, width);
            if (int8)
              io.Copy((__gm__ int32_t *)cfg[4 + c] + 512 + row * 8,
                      (__gm__ int32_t *)auxiliary[0] + (c * 32 + row) * 8, 8);
          }
        } else
          for (int route = worker; route < n * TOPK; route += VW) {
            int row = map[route];
            if (row < 0)
              continue;
            if (kind == REPACK) {
              int width = HIDDEN / (int8 ? 4 : 2);
              io.Copy((__gm__ int32_t *)ptr[0] +
                          (c * 32 + route / TOPK) * width,
                      (__gm__ int32_t *)ptr[1] + row * width, width);
              if (int8)
                io.Copy((__gm__ int32_t *)auxiliary[0] +
                            (c * 32 + route / TOPK) * 8,
                        (__gm__ int32_t *)auxiliary[1] + row * 8, 8);
            } else if (int8) {
              int expert = RowExpert(ends, row);
              quant.Dequant((__gm__ int32_t *)ptr[4] + row * HIDDEN,
                            (__gm__ float *)weights[3] + expert * HIDDEN,
                            (__gm__ float *)auxiliary[2] + row * 8, HIDDEN);
              quant.ToBf16(
                  (__gm__ bfloat16_t *)((__gm__ int32_t *)cfg[2 + c] + 64) +
                      route * HIDDEN,
                  HIDDEN);
            } else
              io.Copy((__gm__ int32_t *)ptr[4] + row * HIDDEN / 2,
                      (__gm__ int32_t *)cfg[2 + c] + 64 + route * HIDDEN / 2,
                      HIDDEN / 2);
          }
      }
    }
    PipeBarrier<PIPE_ALL>();
    WorkTime(cfg, 0, next, worker, begin);
    seen = next;
    Store(ctrl + (VDONE + worker) * LINE, seen);
  }
}
