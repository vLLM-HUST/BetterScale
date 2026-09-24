#pragma once
// DFC-style slot-owned MTE2/MTE3 pipeline. No vector/scalar fence per row.
// Whole-layer placement: FETCH transfers token-major input once; REPACK loads
// each token once and fans it out to its K expert-major destinations locally.
// Native-planned cap1 FETCH directly fans out peer chunks after maps freeze.
// Fine-pack/urgent preemption are disabled for this full-command completion path.
struct V2LiteMovePipeline {
  LocalTensor<int32_t> scratch;
  int sequence=0;
  __aicore__ inline void Init(LocalTensor<int32_t> words) {
    scratch=words;
    SetFlag<HardEvent::S_MTE2>(EVENT_ID0);WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID2);
  }
  __aicore__ inline void Copy(__gm__ int32_t *src,__gm__ int32_t *dst,int n) {
    auto event=(sequence&1)?EVENT_ID2:EVENT_ID1;
    auto local=scratch[(sequence&1)*8192];
    GlobalTensor<int32_t> in,out;in.SetGlobalBuffer(src);out.SetGlobalBuffer(dst);
    WaitFlag<HardEvent::MTE3_MTE2>(event);
    DataCopy(local,in,n);
    SetFlag<HardEvent::MTE2_MTE3>(event);WaitFlag<HardEvent::MTE2_MTE3>(event);
    DataCopy(out,local,n);
    SetFlag<HardEvent::MTE3_MTE2>(event);++sequence;
  }
  __aicore__ inline void Fanout(__gm__ int32_t *src,__gm__ int32_t *dst,int *map) {
    auto event=(sequence&1)?EVENT_ID2:EVENT_ID1;
    auto local=scratch[(sequence&1)*8192];
    GlobalTensor<int32_t> in,out;in.SetGlobalBuffer(src);
    WaitFlag<HardEvent::MTE3_MTE2>(event);
    DataCopy(local,in,HIDDEN/2);
    SetFlag<HardEvent::MTE2_MTE3>(event);WaitFlag<HardEvent::MTE2_MTE3>(event);
    for(int k=0;k<TOPK;++k) {
      if(map[k]<0)continue; // EP: unowned routes have no local destination.
      out.SetGlobalBuffer(dst+map[k]*HIDDEN/2);
      DataCopy(out,local,HIDDEN/2);
    }
    SetFlag<HardEvent::MTE3_MTE2>(event);++sequence;
  }

  __aicore__ inline void RemoteFanout(__gm__ int32_t *src,
                                     __gm__ int32_t *dst,int *map,int tokens) {
    auto event=(sequence&1)?EVENT_ID2:EVENT_ID1;
    auto local=scratch[(sequence&1)*8192];
    GlobalTensor<int32_t> in,out;in.SetGlobalBuffer(src);
    WaitFlag<HardEvent::MTE3_MTE2>(event);
    DataCopy(local,in,tokens*HIDDEN/2);
    SetFlag<HardEvent::MTE2_MTE3>(event);WaitFlag<HardEvent::MTE2_MTE3>(event);
    for(int t=0;t<tokens;++t)for(int k=0;k<TOPK;++k) {
      out.SetGlobalBuffer(dst+map[t*TOPK+k]*HIDDEN/2);
      DataCopy(out,local[t*HIDDEN/2],HIDDEN/2);
    }
    SetFlag<HardEvent::MTE3_MTE2>(event);++sequence;
  }
  __aicore__ inline void Finish() {
    // Also gates the next source's metadata read, not merely buffer reuse.
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID2);
  }
};
__aicore__ inline void V2LiteMove(__gm__ int64_t *cfg,Transfer &io,
                                 int kind,int slot,int worker,int mask) {
  auto ptr=(__gm__ int64_t*)cfg[1]+slot*16;
  auto sources=(__gm__ int64_t*)cfg[27];
  for(int c=0;c<SOURCES;++c) {
    io.Read((__gm__ int32_t*)ptr[5]+c*MAP,8);
    int gen=io.words.GetValue(0),n=io.words.GetValue(1);
    if(!gen || (kind==FETCH && !(mask&(1<<c))))continue;

    if(kind==FETCH && ExpertRoutePlan::MIN_ROWS &&
       n>=ExpertRoutePlan::MIN_ROWS && !cfg[16] && !cfg[17]) {
      int per=(n+VW-1)/VW,first=worker*per,last=ScalarMin(n,first+per);
      if(first>=last)continue;
      int map[(TOKENS+VW-1)/VW*TOPK];
      io.Read((__gm__ int32_t*)ptr[5]+c*MAP+8+first*TOPK,(last-first)*TOPK);
      for(int i=0;i<(last-first)*TOPK;++i)map[i]=io.words.GetValue(i);
      V2LiteMovePipeline copy;copy.Init(io.words);
      for(int t=first;t<last;t+=8)
        copy.RemoteFanout((__gm__ int32_t*)sources[c]+V2LITE_PAYLOAD_WORDS+t*HIDDEN/2,
                          (__gm__ int32_t*)ptr[1],map+(t-first)*TOPK,
                          ScalarMin(8,last-t));
      copy.Finish();
    } else if(kind==FETCH) {

      V2LiteMovePipeline copy;copy.Init(io.words);
      int words=n*HIDDEN/2;
      for(int off=worker*8192;off<words;off+=VW*8192)
        copy.Copy((__gm__ int32_t*)sources[c]+V2LITE_PAYLOAD_WORDS+off,
                  (__gm__ int32_t*)ptr[0]+c*TOKENS*HIDDEN/2+off,
                  ScalarMin(8192,words-off));
      copy.Finish();
    } else {
      int per=(n+VW-1)/VW,first=worker*per,last=ScalarMin(n,first+per);
      if(first>=last)continue;
      int map[(TOKENS+VW-1)/VW*TOPK];
      int start=(8+first*TOPK)/8*8,end=(8+last*TOPK+7)/8*8;
      io.Read((__gm__ int32_t*)ptr[5]+c*MAP+start,end-start);
      for(int t=first;t<last;++t)
        for(int k=0;k<TOPK;++k)
          map[(t-first)*TOPK+k]=io.words.GetValue(8+t*TOPK+k-start);
      V2LiteMovePipeline copy;copy.Init(io.words);
      for(int t=first;t<last;++t)
        copy.Fanout((__gm__ int32_t*)ptr[0]+(c*TOKENS+t)*HIDDEN/2,
                    (__gm__ int32_t*)ptr[1],map+(t-first)*TOPK);
      copy.Finish();
    }
  }
}
