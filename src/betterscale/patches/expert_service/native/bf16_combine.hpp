#pragma once
// Whole-layer owner only: all K experts are local. Preserve the client's
// canonical FP32 Cast/Muls/Add order, but transfer one reduced BF16 vector.
// Coordinator admits SEND only after full down completion (early-return off).
// Metadata is consumed before reusing UB; per-worker maps/probs survive in
// scalar storage. Producer input and slot catalog remain immutable until DONE.
__aicore__ inline void V2LiteCombine(__gm__ int64_t *cfg,Transfer &io,
                                    int slot,int worker) {
  constexpr int PROB_FLOAT=(TOKENS*TOPK*2+255)/256*64;
  constexpr int MAX_LOCAL=(TOKENS+VW-1)/VW*TOPK;
  auto ptr=(__gm__ int64_t*)cfg[1]+slot*16;
  auto sources=(__gm__ int64_t*)cfg[27];
  auto outputs=(__gm__ int64_t*)cfg[28];
  for(int c=0;c<SOURCES;++c) {
    io.Read((__gm__ int32_t*)ptr[5]+c*MAP,8);
    int gen=io.words.GetValue(0),n=io.words.GetValue(1);
    if(!gen)continue;
    int per=(n+VW-1)/VW,first=worker*per,end=ScalarMin(n,first+per);
    if(first>=end)continue;
    int map[MAX_LOCAL];float weights[MAX_LOCAL];
    io.Read((__gm__ int32_t*)ptr[5]+c*MAP,(8+n*TOPK+7)/8*8);
    for(int t=first;t<end;++t)
      for(int k=0;k<TOPK;++k)
        map[(t-first)*TOPK+k]=io.words.GetValue(8+t*TOPK+k);
    // Every worker bulk-loads the small BF16 probability metadata once/source,
    // not one remote scalar load per token. Source READY already covers it.
    auto bf=io.buf.Get<bfloat16_t>();auto fp=io.buf.Get<float>();
    GlobalTensor<bfloat16_t> probabilities;
    probabilities.SetGlobalBuffer((__gm__ bfloat16_t*)(sources[c]+V2LITE_PROB_WORDS*4));
    int padded=(n*TOPK+15)/16*16;
    DataCopy(bf,probabilities,padded);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    Cast(fp[PROB_FLOAT],bf,RoundMode::CAST_NONE,padded);
    SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
    for(int t=first;t<end;++t)
      for(int k=0;k<TOPK;++k)
        weights[(t-first)*TOPK+k]=fp.GetValue(PROB_FLOAT+t*TOPK+k);
    SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
    SetFlag<HardEvent::V_MTE2>(EVENT_ID2);
    SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
    for(int t=first;t<end;++t) {
      int base=(t-first)*TOPK;
      Duplicate(fp[5376],0.0f,HIDDEN);PipeBarrier<PIPE_V>();
      GlobalTensor<bfloat16_t> in;
      in.SetGlobalBuffer((__gm__ bfloat16_t*)ptr[4]+map[base]*HIDDEN);
      WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
      DataCopy(bf[256],in,HIDDEN);
      SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
      for(int k=0;k<TOPK;++k) {
        if(k+1<TOPK) {
          auto ev=((k+1)&1)?EVENT_ID2:EVENT_ID1;
          in.SetGlobalBuffer((__gm__ bfloat16_t*)ptr[4]+map[base+k+1]*HIDDEN);
          WaitFlag<HardEvent::V_MTE2>(ev);
          DataCopy(bf[256+((k+1)&1)*HIDDEN],in,HIDDEN);
          SetFlag<HardEvent::MTE2_V>(ev);
        }
        auto ev=(k&1)?EVENT_ID2:EVENT_ID1;
        WaitFlag<HardEvent::MTE2_V>(ev);
        Cast(fp[2816],bf[256+(k&1)*HIDDEN],RoundMode::CAST_NONE,HIDDEN);
        SetFlag<HardEvent::V_MTE2>(ev);PipeBarrier<PIPE_V>();
        Muls(fp[2816],fp[2816],weights[base+k],HIDDEN);PipeBarrier<PIPE_V>();
        Add(fp[5376],fp[5376],fp[2816],HIDDEN);PipeBarrier<PIPE_V>();
      }
      WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
      Cast(bf[16384],fp[5376],RoundMode::CAST_RINT,HIDDEN);
      SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
      GlobalTensor<bfloat16_t> out;
      out.SetGlobalBuffer((__gm__ bfloat16_t*)outputs[c]+128+t*HIDDEN);
      DataCopy(out,bf[16384],HIDDEN);
      SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
      SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
    }
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID2);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    // The next source's metadata can cover the output UB range. MTE3_V only
    // gates vector reuse, not its MTE2 read: fence the actual next consumer.
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
  }
}
