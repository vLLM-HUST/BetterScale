#pragma once
#include "kernel_operator.h"

// A2 ReduceMin/Max accepts float, not int32. Conversion preserves membership in
// [0,256): every valid ID and both boundaries are exactly representable; rounded
// out-of-range int32 values remain out of range. No model arithmetic is changed.
// Disjoint196608-byte UB: IDs[0,32768), float chunk[32768,36864),
// reduction scratch[36864,40960), min/max lines[40960,40976).
__aicore__ inline bool PlannedRouteIdsValid(
    const AscendC::LocalTensor<int32_t>& ids,
    const AscendC::LocalTensor<float>& fp, int count) {
  using namespace AscendC;
  SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
  WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
  for (int first=0;first<count;first+=4096) {
    int n=count-first<4096?count-first:4096;
    Cast(fp[32768],ids[first],RoundMode::CAST_RINT,n);
    PipeBarrier<PIPE_V>();
    ReduceMin(fp[40960],fp[32768],fp[36864],n,false);
    PipeBarrier<PIPE_V>();
    ReduceMax(fp[40968],fp[32768],fp[36864],n,false);
    SetFlag<HardEvent::V_S>(EVENT_ID0);
    WaitFlag<HardEvent::V_S>(EVENT_ID0);
    if(fp.GetValue(40960)<0.0f || fp.GetValue(40968)>=256.0f)
      return false;
  }
  return true;
}
