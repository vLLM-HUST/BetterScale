#pragma once
#include "quant_vector.hpp"

// Target-only ACTIVATE, four contiguous rows of one expert per UB batch.
// Keeps QuantRow's arithmetic order/rounding and padded eight-float scale ABI.
// All storage fits the existing worker's 64KiB scratch; no persistent GM
// buffer, new communication, or cross-core synchronization. BF16 MTP remains
// QuantRow.
struct QuantBatch {
  static constexpr int INNER = 640, WIDTH = 1280, ROWS = 4;
  LocalTensor<int32_t> scratch;
  __aicore__ inline QuantBatch(LocalTensor<int32_t> s) : scratch(s) {}
  __aicore__ inline void Run(__gm__ int32_t *input, __gm__ float *channel,
                             __gm__ float *inputScale, __gm__ int8_t *output,
                             __gm__ float *outputScale, int rows,
                             bool loadChannel) {
    auto f = scratch.ReinterpretCast<float>();
    auto values = f[5120], scales = f[10240], tmp = f[11520];
    auto reduction = f[14080], rowScales = f[14144];
    GlobalTensor<int32_t> x;
    GlobalTensor<float> cs, rs, os;
    GlobalTensor<int8_t> out;
    x.SetGlobalBuffer(input);
    cs.SetGlobalBuffer(channel);
    rs.SetGlobalBuffer(inputScale);
    os.SetGlobalBuffer(outputScale);
    out.SetGlobalBuffer(output);
    DataCopy(scratch, x, rows * WIDTH);
    DataCopy(rowScales, rs, rows * 8);
    if (loadChannel)
      DataCopy(scales, cs, WIDTH);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
    Cast(values, scratch, RoundMode::CAST_RINT, rows * WIDTH);
    PipeBarrier<PIPE_V>();
    for (int r = 0; r < rows; ++r)
      Muls(values[r * WIDTH], values[r * WIDTH], rowScales.GetValue(r * 8),
           WIDTH);
    PipeBarrier<PIPE_V>();
    for (int r = 0; r < rows; ++r)
      Mul(values[r * WIDTH], values[r * WIDTH], scales, WIDTH);
    PipeBarrier<PIPE_V>();
    for (int r = 0; r < rows; ++r)
      Muls(tmp[r * INNER], values[r * WIDTH], -1.0f, INNER);
    PipeBarrier<PIPE_V>();
    Exp(tmp, tmp, rows * INNER);
    PipeBarrier<PIPE_V>();
    Adds(tmp, tmp, 1.0f, rows * INNER);
    PipeBarrier<PIPE_V>();
    for (int r = 0; r < rows; ++r)
      Div(values[r * WIDTH], values[r * WIDTH], tmp[r * INNER], INNER);
    PipeBarrier<PIPE_V>();
    for (int r = 0; r < rows; ++r)
      Mul(values[r * WIDTH], values[r * WIDTH], values[r * WIDTH + INNER],
          INNER);
    PipeBarrier<PIPE_V>();
    for (int r = 0; r < rows; ++r)
      Abs(tmp[r * INNER], values[r * WIDTH], INNER);
    PipeBarrier<PIPE_V>();
    // ReduceMax uses its output as scratch just as QuantRow does. Consume the
    // scalar before reusing that reduction scratch for the next row.
    for (int r = 0; r < rows; ++r) {
      ReduceMax(reduction, tmp[r * INNER], reduction, INNER, false);
      SetFlag<HardEvent::V_S>(EVENT_ID0);
      WaitFlag<HardEvent::V_S>(EVENT_ID0);
      float maximum = reduction.GetValue(0);
      float scale = maximum > 0.0f ? maximum / 127.0f : 1.0f;
      for (int i = 0; i < 8; ++i)
        rowScales.SetValue(r * 8 + i, i == 0 ? scale : 0.0f);
      Muls(values[r * WIDTH], values[r * WIDTH], 1.0f / scale, INNER);
    }
    PipeBarrier<PIPE_V>();
    for (int r = 0; r < rows; ++r)
      Cast(values[r * WIDTH].ReinterpretCast<half>(), values[r * WIDTH],
           RoundMode::CAST_RINT, INNER);
    PipeBarrier<PIPE_V>();
    auto bytes = scratch.ReinterpretCast<int8_t>();
    for (int r = 0; r < rows; ++r)
      Cast(bytes[r * INNER], values[r * WIDTH].ReinterpretCast<half>(),
           RoundMode::CAST_RINT, INNER);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
    DataCopy(out, bytes, rows * INNER);
    DataCopy(os, rowScales, rows * 8);
    // Neither UB payload nor its row-scale cache may be reused before export.
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
  }
};

// Give each worker one contiguous range, clipping each batch at expert ends.
// Scales survive between batches of the same expert. Empty/tail workers still
// return to the caller's usual command-completion protocol.
__aicore__ inline void
ActivateBatched(LocalTensor<int32_t> scratch, int worker, int workers, int rows,
                __gm__ int64_t *ends, int groups, __gm__ int32_t *input,
                __gm__ float *channel, __gm__ float *inputScale,
                __gm__ int8_t *output, __gm__ float *outputScale) {
  int perWorker = (rows + workers - 1) / workers;
  int begin = worker * perWorker, end = begin + perWorker;
  if (end > rows)
    end = rows;
  int expert = 0, cached = -1;
  QuantBatch batch(scratch);
  while (begin < end) {
    while (expert < groups && begin >= ends[expert])
      ++expert;
    // The producer contract requires ends[groups-1] == rows.
    if (expert == groups)
      return;
    int count = end - begin;
    if (count > QuantBatch::ROWS)
      count = QuantBatch::ROWS;
    if (count > ends[expert] - begin)
      count = ends[expert] - begin;
    batch.Run(input + begin * QuantBatch::WIDTH,
              channel + expert * QuantBatch::WIDTH, inputScale + begin * 8,
              output + begin * QuantBatch::INNER, outputScale + begin * 8,
              count, cached != expert);
    cached = expert;
    begin += count;
  }
}
