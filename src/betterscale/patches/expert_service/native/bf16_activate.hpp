#pragma once
#include "kernel_operator.h"
using namespace AscendC;

// BF16 specialization of the existing Qwen38 multi-row activation pattern.
// Preserve Cast -> -gate -> Exp -> +1 -> Div -> Mul -> BF16 rounding exactly.
// Four interleaved [gate,value] rows; disjoint byte ranges in128KiB Transfer UB:
// input[0,22528), gate[24576,47104), value[47104,69632), tmp[69632,92160).
// No expert-specific scales, cross-core flags, or change to prefix ownership.
__aicore__ inline void V2LiteActivateRange(LocalTensor<int32_t> scratch,
    __gm__ bfloat16_t *input, __gm__ bfloat16_t *output,
    int begin, int end, int worker, int workers) {
  constexpr int M=512, BATCH=4;
  auto bf=scratch.ReinterpretCast<bfloat16_t>();
  auto fp=scratch.ReinterpretCast<float>();
  auto gate=fp[6144], value=fp[11776], tmp=fp[17408];
  int count=(end-begin+workers-1)/workers;
  int first=begin+worker*count, last=first+count;
  if(last>end)last=end;
  for(int row=first;row<last;row+=BATCH) {
    int n=last-row<BATCH?last-row:BATCH;
    GlobalTensor<bfloat16_t> in,out;
    in.SetGlobalBuffer(input+row*2*M);out.SetGlobalBuffer(output+row*M);
    DataCopy(bf,in,n*2*M);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    for(int r=0;r<n;++r) {
      Cast(gate[r*M],bf[r*2*M],RoundMode::CAST_NONE,M);
      Cast(value[r*M],bf[r*2*M+M],RoundMode::CAST_NONE,M);
    }
    PipeBarrier<PIPE_V>();
    Muls(tmp,gate,-1.0f,n*M);PipeBarrier<PIPE_V>();
    Exp(tmp,tmp,n*M);PipeBarrier<PIPE_V>();
    Adds(tmp,tmp,1.0f,n*M);PipeBarrier<PIPE_V>();
    Div(gate,gate,tmp,n*M);PipeBarrier<PIPE_V>();
    Mul(gate,gate,value,n*M);PipeBarrier<PIPE_V>();
    Cast(bf,gate,RoundMode::CAST_RINT,n*M);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    DataCopy(out,bf,n*M);
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
  }
}
