// Transport-only diagnostic. The production client TU is the matched control.
#include "../../src/betterscale/patches/expert_service/native/client_kernel.cpp"

// Dedicated server device: all <=16 blocks fit simultaneously. Each worker has
// its own padded completion word. READY advances only after client retirement.
// This is an echo lower bound, not an imitation of the expert compute scheduler.
extern "C" __global__ __aicore__ void
transport_echo(GM_ADDR config, GM_ADDR unused, GM_ADDR unused2) {
  auto cfg=(__gm__ int64_t*)config;
  auto src=(__gm__ int32_t*)cfg[0], out=(__gm__ int32_t*)cfg[1];
  auto ack=(__gm__ int32_t*)cfg[2];
  int b=GetBlockIdx(), blocks=GetBlockNum(), last=0;
  IO io;io.Init();
  for (;;) {
    int gen;
    do {gen=io.Flag(src);} while(gen==last || gen==0);
    if(gen<0)break;
    io.Read(src+8);int rows=io.ub.GetValue(2),words=rows*H/2;
    for(int off=b*8192;off<words;off+=blocks*8192) {
      int count=words-off<8192?words-off:8192;
      io.Read(src+PACK_PAYLOAD+off,count);io.Write(out+64+off,count);
    }
    io.Publish(ack+b*16,gen);
    if(b==0) {
      for(int w=1;w<blocks;++w)while(io.Flag(ack+w*16)!=gen){}
      io.Publish(out,gen);
    }
    last=gen;
  }
  io.Publish(ack+b*16,last);
}
META(transport_echo)

extern "C" __global__ __aicore__ void
transport_wait(GM_ADDR config, GM_ADDR unused, GM_ADDR unused2) {
  auto cfg=(__gm__ int64_t*)config;IO io;io.Init();
  int gen=io.Flag((__gm__ int32_t*)cfg[0]);
  while(io.Flag((__gm__ int32_t*)cfg[1])!=gen){}
}
META(transport_wait)

extern "C" __global__ __aicore__ void
transport_copy(GM_ADDR config, GM_ADDR unused, GM_ADDR unused2) {
  auto cfg=(__gm__ int64_t*)config;IO io;io.Init();int words=cfg[6]*H/2;
  for(int off=GetBlockIdx()*8192;off<words;off+=GetBlockNum()*8192) {
    int count=words-off<8192?words-off:8192;
    io.Read((__gm__ int32_t*)cfg[1]+64+off,count);
    io.Write((__gm__ int32_t*)cfg[12]+off,count);
  }
}
META(transport_copy)

// DFC CopyGMToGM-style ping-pong: preserve MTE2/MTE3 dependencies per UB slot,
// not a PIPE_ALL and complete read/write drain after every 32-KiB tile.
__aicore__ inline void PipelinedCopy(IO &io,__gm__ int32_t *src,__gm__ int32_t *dst,int words) {
  SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
  SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID2);
  int iteration=0;
  for(int off=GetBlockIdx()*8192;off<words;off+=GetBlockNum()*8192,++iteration) {
    int count=words-off<8192?words-off:8192,slot=(iteration&1)*8192;
    auto event=(iteration&1)?EVENT_ID2:EVENT_ID1;
    WaitFlag<HardEvent::MTE3_MTE2>(event);
    GlobalTensor<int32_t> in,out;in.SetGlobalBuffer(src+off);out.SetGlobalBuffer(dst+off);
    DataCopy(io.ub[slot],in,count);
    SetFlag<HardEvent::MTE2_MTE3>(event);WaitFlag<HardEvent::MTE2_MTE3>(event);
    DataCopy(out,io.ub[slot],count);
    SetFlag<HardEvent::MTE3_MTE2>(event);
  }
  WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID2);
}
extern "C" __global__ __aicore__ void
transport_copy_pipeline(GM_ADDR config, GM_ADDR unused, GM_ADDR unused2) {
  auto cfg=(__gm__ int64_t*)config;IO io;io.Init();
  PipelinedCopy(io,(__gm__ int32_t*)cfg[1]+64,(__gm__ int32_t*)cfg[12],cfg[6]*H/2);
}
META(transport_copy_pipeline)
extern "C" __global__ __aicore__ void
transport_collect_pipeline(GM_ADDR config, GM_ADDR unused, GM_ADDR unused2) {
  auto cfg=(__gm__ int64_t*)config;IO io;io.Init();
  int gen=io.Flag((__gm__ int32_t*)cfg[0]);
  while(io.Flag((__gm__ int32_t*)cfg[1])!=gen){}
  PipelinedCopy(io,(__gm__ int32_t*)cfg[1]+64,(__gm__ int32_t*)cfg[12],cfg[6]*H/2);
}
META(transport_collect_pipeline)
