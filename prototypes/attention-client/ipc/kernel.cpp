// IO publication primitives reused from workspace commit 3532418.
#include "kernel_operator.h"
using namespace AscendC;
constexpr int DEPTH=8, STRIDE=4096, WIDTH=64, MAX_ROWS=8;
// Metadata = [generation, task, layer, expert, phase (0=D), rows, 0, 0].
// Source READY and result DONE are separate cache-line-sized blocks.
class IO {
public:
 TPipe pipe;
 TBuf<TPosition::VECCALC> buf;
 LocalTensor<int32_t> ub;
 __aicore__ inline void Init() { pipe.InitBuffer(buf,16384); ub=buf.Get<int32_t>(); }
 __aicore__ inline void Read(__gm__ int32_t* p,int n=8,int offset=0) {
  GlobalTensor<int32_t> g;g.SetGlobalBuffer(p);DataCopy(ub[offset],g,n);
  SetFlag<HardEvent::MTE2_S>(EVENT_ID0);WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
 }
 __aicore__ inline void Write(__gm__ int32_t* p,int n=8,int offset=0) {
  SetFlag<HardEvent::S_MTE3>(EVENT_ID0);WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
  GlobalTensor<int32_t> g;g.SetGlobalBuffer(p);DataCopy(g,ub[offset],n);
  SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
  PipeBarrier<PIPE_ALL>();
 }
 __aicore__ inline int Flag(__gm__ int32_t* p) {Read(p);return ub.GetValue(0);}
 __aicore__ inline void Publish(__gm__ int32_t* p,int gen) {
  for(int i=0;i<8;++i)ub.SetValue(i,i==0?gen:0);Write(p);
 }
};

// cfg = [source, server_result, client_index, task_count, max_polls].
// plan[task] = descriptor[8] + payload[512]. Plans are immutable graph inputs.
// This producer accepts external INT32 packets; unlike the server fixture,
// it does not fabricate descriptors or hidden values inside the kernel.
extern "C" __global__ __aicore__ void pull_expert_client(
    GM_ADDR config, GM_ADDR plans, GM_ADDR outputs) {
 if(GetBlockIdx()!=0)return;
 auto cfg=(__gm__ int64_t*)config;
 auto source=(__gm__ int32_t*)cfg[0];
 auto result=(__gm__ int32_t*)cfg[1];
 auto plan=(__gm__ int32_t*)plans;
 auto output=(__gm__ int32_t*)outputs;
 const int owner=cfg[2],tasks=cfg[3],limit=cfg[4];
 IO io;io.Init();
 int finished[DEPTH]={0},active[DEPTH]={0};
 int done=0,polls=0;
 while(done<tasks && polls++<limit) {
  for(int slot=0;slot<DEPTH;++slot) {
   int task=slot+finished[slot]*DEPTH;
   if(task>=tasks)continue;
   int gen=finished[slot]+1;
   auto src=source+slot*STRIDE;
   auto dst=result+(owner*DEPTH+slot)*STRIDE;
   io.Read(plan+task*520,8);
   int rows=io.ub.GetValue(5);
   if(io.ub.GetValue(0)!=gen || io.ub.GetValue(1)!=task || rows<1 || rows>MAX_ROWS) {
    io.Publish(output,-2);return;
   }
   if(!active[slot]) {
    io.Read(plan+task*520+8,rows*WIDTH);
    io.Write(src+64,rows*WIDTH);
    io.Read(plan+task*520,8);io.Write(src+8);
    io.Publish(src,gen);active[slot]=1;
   }
   if(io.Flag(dst)==gen) {
    io.Read(dst+64,rows*WIDTH);
    io.Write(output+64+task*512,rows*WIDTH);
    // Only now has consumption completed, permitting next-generation READY.
    ++finished[slot];active[slot]=0;++done;
   }
  }
 }
 for(int i=0;i<8;++i)io.ub.SetValue(i,i==0?(done==tasks?1:-1):(i==1?done:0));
 io.Write(output,8);
}
static const struct FunLevelKType metadata __attribute__((used,section(".ascend.meta.pull_expert_client")))={{F_TYPE_KTYPE,sizeof(unsigned int),K_TYPE_AIV}};
