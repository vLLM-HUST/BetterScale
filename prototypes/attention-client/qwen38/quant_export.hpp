#pragma once
#include "kernel_operator.h"
using namespace AscendC;

// SEND-only INT32 -> BF16, with the exact QuantRow Cast/Muls/Mul/Cast order.
// Metadata occupies bytes[0,1024). Two input+scale slots survive map-tile reads.
// Buffer reuse is local; Finish must precede worker DONE / command reuse.
struct QuantExport {
  static constexpr int H = 2560;
  LocalTensor<int32_t> words;
  bool pending = false;
  int pendingSlot = 0, nextSlot = 0, generation = 0;
  __gm__ bfloat16_t *destination = nullptr;
  __gm__ int32_t *ready = nullptr;
  __aicore__ inline QuantExport(LocalTensor<int32_t> w) : words(w) {}
  __aicore__ inline void Init() {
    SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
    SetFlag<HardEvent::V_MTE2>(EVENT_ID2);
    SetFlag<HardEvent::MTE3_V>(EVENT_ID1);
    SetFlag<HardEvent::MTE3_V>(EVENT_ID2);
  }
  __aicore__ inline void Consume() {
    int slot = pendingSlot;
    auto event = slot ? EVENT_ID2 : EVENT_ID1;
    auto f = words.ReinterpretCast<float>();
    auto bf = words.ReinterpretCast<bfloat16_t>();
    WaitFlag<HardEvent::MTE2_V>(event);
    WaitFlag<HardEvent::MTE2_S>(event);
    float factor = f.GetValue(15616 + slot * 8);
    PipeBarrier<PIPE_V>(); // Previous output Cast must finish reading shared FP32 scratch.
    Cast(f[10496], words[256 + slot * H], RoundMode::CAST_RINT, H);
    PipeBarrier<PIPE_V>();
    Muls(f[10496], f[10496], factor, H);
    PipeBarrier<PIPE_V>();
    Mul(f[10496], f[10496], f[5376 + slot * H], H);
    PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_MTE2>(event);
    WaitFlag<HardEvent::MTE3_V>(event);
    Cast(bf[26112 + slot * H], f[10496], RoundMode::CAST_RINT, H);
    SetFlag<HardEvent::V_MTE3>(event);
    WaitFlag<HardEvent::V_MTE3>(event);
    GlobalTensor<bfloat16_t> out;
    out.SetGlobalBuffer(destination);
    DataCopy(out, bf[26112 + slot * H], H);
    SetFlag<HardEvent::MTE3_V>(event);
    if (ready != nullptr) {
      // Online consumers require per-route READY after its payload, even though
      // fixed-order consumers can omit these notifications at build time.
      SetFlag<HardEvent::MTE3_S>(EVENT_ID3);
      WaitFlag<HardEvent::MTE3_S>(EVENT_ID3);
      for (int j = 0; j < 8; ++j) words.SetValue(15632 + j, j ? 0 : generation);
      SetFlag<HardEvent::S_MTE3>(EVENT_ID3);
      WaitFlag<HardEvent::S_MTE3>(EVENT_ID3);
      GlobalTensor<int32_t> flag;
      flag.SetGlobalBuffer(ready);
      DataCopy(flag, words[15632], 8);
      SetFlag<HardEvent::MTE3_S>(EVENT_ID3);
      WaitFlag<HardEvent::MTE3_S>(EVENT_ID3);
    }
    pending = false;
  }
  __aicore__ inline void Submit(__gm__ int32_t *input, __gm__ float *channel,
                               __gm__ float *rowScale, __gm__ bfloat16_t *out,
                               __gm__ int32_t *flag, int gen) {
    int slot = nextSlot;
    auto event = slot ? EVENT_ID2 : EVENT_ID1;
    auto f = words.ReinterpretCast<float>();
    GlobalTensor<int32_t> x;
    GlobalTensor<float> cs, rs;
    x.SetGlobalBuffer(input); cs.SetGlobalBuffer(channel); rs.SetGlobalBuffer(rowScale);
    WaitFlag<HardEvent::V_MTE2>(event);
    DataCopy(words[256 + slot * H], x, H);
    DataCopy(f[5376 + slot * H], cs, H);
    DataCopy(f[15616 + slot * 8], rs, 8);
    SetFlag<HardEvent::MTE2_V>(event);
    SetFlag<HardEvent::MTE2_S>(event);
    if (pending) Consume();
    pending = true; pendingSlot = slot; nextSlot ^= 1;
    destination = out; ready = flag; generation = gen;
  }
  __aicore__ inline void Finish() {
    if (pending) Consume();
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID2);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID1);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID2);
  }
};
