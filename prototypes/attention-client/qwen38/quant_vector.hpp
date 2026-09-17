#pragma once
#include "kernel_operator.h"
using namespace AscendC;

// A row-local adapter around the upstream GMM-SwiGLU quantization recipe.
// Caller owns one 64KiB UB scratch region. No global barrier or shared writer.
struct QuantRow {
  LocalTensor<int32_t> scratch;
  __aicore__ inline QuantRow(LocalTensor<int32_t> s) : scratch(s) {}
  __aicore__ inline LocalTensor<float> Values() {
    return scratch.ReinterpretCast<float>()[4096];
  }
  __aicore__ inline LocalTensor<float> Temp() {
    return scratch.ReinterpretCast<float>()[11264];
  }
  template <typename T>
  __aicore__ inline void Read(LocalTensor<T> dst, __gm__ T *src, int count) {
    GlobalTensor<T> gm;
    gm.SetGlobalBuffer(src);
    DataCopy(dst, gm, count);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
  }
  __aicore__ inline void FromBf16(__gm__ bfloat16_t *src, int width) {
    auto input = scratch.ReinterpretCast<bfloat16_t>();
    Read(input, src, width);
    Cast(Values(), input, RoundMode::CAST_NONE, width);
    PipeBarrier<PIPE_V>();
  }
  __aicore__ inline void Dequant(__gm__ int32_t *src, __gm__ float *channel,
                                 __gm__ float *rowScale, int width) {
    auto scale = scratch.ReinterpretCast<float>()[8192];
    auto row = scratch.ReinterpretCast<float>()[14592];
    Read(scratch, src, width);
    Cast(Values(), scratch, RoundMode::CAST_RINT, width);
    PipeBarrier<PIPE_V>();
    Read(scale, channel, width);
    Read(row, rowScale, 8);
    SetFlag<HardEvent::V_S>(EVENT_ID0);
    WaitFlag<HardEvent::V_S>(EVENT_ID0);
    float factor = row.GetValue(0);
    Muls(Values(), Values(), factor, width);
    PipeBarrier<PIPE_V>();
    Mul(Values(), Values(), scale, width);
    PipeBarrier<PIPE_V>();
  }
  __aicore__ inline void Swiglu(int inner) {
    auto gate = Values(), tmp = Temp();
    Muls(tmp, gate, -1.0f, inner);
    PipeBarrier<PIPE_V>();
    Exp(tmp, tmp, inner);
    PipeBarrier<PIPE_V>();
    Adds(tmp, tmp, 1.0f, inner);
    PipeBarrier<PIPE_V>();
    Div(gate, gate, tmp, inner);
    PipeBarrier<PIPE_V>();
    Mul(gate, gate, gate[inner], inner);
    PipeBarrier<PIPE_V>();
  }
  __aicore__ inline void Quantize(__gm__ int8_t *dst, __gm__ float *rowScale,
                                  int width, bool integerFirst = false) {
    auto values = Values(), tmp = Temp();
    auto reduction = scratch.ReinterpretCast<float>()[14336];
    auto row = scratch.ReinterpretCast<float>()[14592];
    Abs(tmp, values, width);
    PipeBarrier<PIPE_V>();
    ReduceMax(reduction, tmp, reduction, width, false);
    SetFlag<HardEvent::V_S>(EVENT_ID0);
    WaitFlag<HardEvent::V_S>(EVENT_ID0);
    float maximum = reduction.GetValue(0);
    float scale = maximum > 0.0f ? maximum / 127.0f : 1.0f;
    for (int i = 0; i < 8; ++i)
      row.SetValue(i, i == 0 ? scale : 0.0f);
    Muls(values, values, 1.0f / scale, width);
    PipeBarrier<PIPE_V>();
    // Same two-stage RINT cast as the owned upstream GMM-SwiGLU-quant.
    auto halfValues = values.ReinterpretCast<half>();
    if (integerFirst) {
      // Input DynamicQuant rounds FP32 directly to integer; an intermediate
      // FP16 before rounding would introduce double-rounding near half integers.
      Cast(scratch, values, RoundMode::CAST_RINT, width);
      PipeBarrier<PIPE_V>();
      Cast(halfValues, scratch, RoundMode::CAST_RINT, width);
    } else
      Cast(halfValues, values, RoundMode::CAST_RINT, width);
    PipeBarrier<PIPE_V>();
    auto bytes = scratch.ReinterpretCast<int8_t>();
    Cast(bytes, halfValues, RoundMode::CAST_RINT, width);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
    GlobalTensor<int8_t> out;
    GlobalTensor<float> scales;
    out.SetGlobalBuffer(dst);
    scales.SetGlobalBuffer(rowScale);
    DataCopy(out, bytes, width);
    DataCopy(scales, row, 8);
    SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
  }
  __aicore__ inline void ToBf16(__gm__ bfloat16_t *dst, int width) {
    auto bf = scratch.ReinterpretCast<bfloat16_t>();
    Cast(bf, Values(), RoundMode::CAST_RINT, width);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    GlobalTensor<bfloat16_t> out;
    out.SetGlobalBuffer(dst);
    DataCopy(out, bf, width);
    SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
  }
};
