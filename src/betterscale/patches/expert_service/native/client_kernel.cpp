// IO ordering reused from the validated 3532418 client/server prototype.
#include "kernel_operator.h"
using namespace AscendC;
constexpr int H = 2048, TOPK = 8, ROUTES = 32768;
// Descriptor at source+8: [generation, layer, rows, class (0=D/1=P), 0...].
// source+16 contains a separate generation-tagged shared-completion signal.
// Source READY and result DONE are separate cache-line-sized blocks.
class IO {
public:
  TPipe pipe;
  TBuf<TPosition::VECCALC> buf;
  LocalTensor<int32_t> ub;
  __aicore__ inline void Init() {
    pipe.InitBuffer(buf, 65536);
    ub = buf.Get<int32_t>();
  }
  __aicore__ inline void Read(__gm__ int32_t *p, int n = 8, int offset = 0) {
    GlobalTensor<int32_t> g;
    g.SetGlobalBuffer(p);
    DataCopy(ub[offset], g, n);
    SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
  }
  __aicore__ inline void Write(__gm__ int32_t *p, int n = 8, int offset = 0) {
    SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
    GlobalTensor<int32_t> g;
    g.SetGlobalBuffer(p);
    DataCopy(g, ub[offset], n);
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    PipeBarrier<PIPE_ALL>();
  }
  __aicore__ inline int Flag(__gm__ int32_t *p) {
    Read(p);
    return ub.GetValue(0);
  }
  __aicore__ inline void Publish(__gm__ int32_t *p, int gen) {
    for (int i = 0; i < 8; ++i)
      ub.SetValue(i, i == 0 ? gen : 0);
    Write(p);
  }
};


#define META(name) static const struct FunLevelKType name##_meta \
__attribute__((used,section(".ascend.meta." #name)))={{F_TYPE_KTYPE,sizeof(unsigned int),K_TYPE_AIV}};

constexpr int PACK_PAYLOAD=50176, PACK_SCALES=0;
// Appended to the frozen client TU: reuse its IO ordering and metadata macro.
// Packing never publishes READY. A following one-block kernel on the same
// stream is the global completion boundary; no unsafe inter-block spin barrier.
extern "C" __global__ __aicore__ void
neural_pack(GM_ADDR cfgaddr, GM_ADDR hidden, GM_ADDR topk) {
  auto cfg = (__gm__ int64_t *)cfgaddr;
  if (!cfg[8]) return;
  IO io;
  io.Init();
  auto src = (__gm__ int32_t *)cfg[0];
  const int n = cfg[6], block = GetBlockIdx(), blocks = GetBlockNum();
  const bool quantized = false;
  const int words = n * H / (quantized ? 4 : 2);
  constexpr int TILE = 8192;
  for (int offset = block * TILE; offset < words; offset += blocks * TILE) {
    int count = words - offset < TILE ? words - offset : TILE;
    io.Read((__gm__ int32_t *)hidden + offset, count);
    io.Write(src + PACK_PAYLOAD + offset, count);
  }
  const int routes = (n * TOPK + 7) / 8 * 8;
  for (int offset = block * TILE; offset < routes; offset += blocks * TILE) {
    int count = routes - offset < TILE ? routes - offset : TILE;
    io.Read((__gm__ int32_t *)topk + offset, count);
    io.Write(src + 64 + offset, count);
  }
  const int probabilityWords=(n*TOPK+15)/16*8;
  for(int off=block*TILE;off<probabilityWords;off+=blocks*TILE) {
    int count=probabilityWords-off<TILE?probabilityWords-off:TILE;
    io.Read((__gm__ int32_t*)cfg[11]+off,count);
    io.Write(src+33792+off,count);
  }
  if (quantized) {
    // Eight compact FP32 scales become eight cache-line-padded wire entries.
    for (int row = block * 8; row < n; row += blocks * 8) {
      io.Read((__gm__ int32_t *)cfg[16] + row, 8);
      int values[8];
      for (int j = 0; j < 8; ++j) values[j] = io.ub.GetValue(j);
      for (int j = 0; j < 64; ++j)
        io.ub.SetValue(j, j % 8 == 0 ? values[j / 8] : 0);
      int count = n - row < 8 ? n - row : 8;
      io.Write(src + PACK_SCALES + row * 8, count * 8);
    }
  }
}
extern "C" __global__ __aicore__ void
neural_publish(GM_ADDR cfgaddr, GM_ADDR unused, GM_ADDR unused2) {
  if (GetBlockIdx() != 0) return;
  auto cfg = (__gm__ int64_t *)cfgaddr;
  if (!cfg[8]) return;
  IO io;
  io.Init();
  auto src = (__gm__ int32_t *)cfg[0];
  int gen = io.Flag((__gm__ int32_t *)cfg[7]) + 1;
  for (int j = 0; j < 8; ++j)
    io.ub.SetValue(j, j == 0 ? gen : j == 1 ? cfg[5] : j == 2 ? cfg[6] : j == 3 ? cfg[15] : 0);
  io.Write(src + 8);
  io.Publish(src, gen);
}
META(neural_pack)
META(neural_publish)

constexpr int COLLECT_OWNERS=1,COLLECT_EXPERTS=256;
// Same wire/join and fixed FP32 route order as neural_collect_fused. Only the
// consumer pipeline changes: two input slots let MTE2 fetch k+1 while V reduces k.
// UB bytes: IO metadata [0,512); BF16 slots [512,10752);
// FP32 cast/product [11264,21504), sum [21504,31744); BF16 output [32768,37888).
// IO uses event0; input slots use events1/2. Output has its own lifetime.
extern "C" __global__ __aicore__ void
neural_collect_pipelined(GM_ADDR cfgaddr, GM_ADDR unused, GM_ADDR topk) {
  static_assert(H == 2048 && TOPK == 8, "Replan the UB map for other shapes");
  auto cfg = (__gm__ int64_t *)cfgaddr;
  if (!cfg[8]) return;
  IO io;
  io.Init();
  int gen = io.Flag((__gm__ int32_t *)cfg[0]), n = cfg[6];
  for (int owner = 0; owner < COLLECT_OWNERS; ++owner) {
    int polls = 0;
    while (io.Flag((__gm__ int32_t *)cfg[1 + owner]) != gen && ++polls < cfg[9]) {}
    if (polls >= cfg[9]) { io.Publish((__gm__ int32_t *)cfg[7], -100); return; }
  }
  auto bf = io.buf.Get<bfloat16_t>();
  auto fp = io.buf.Get<float>();
  SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
  SetFlag<HardEvent::V_MTE2>(EVENT_ID2);
  SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
  for (int token = GetBlockIdx(); token < n; token += GetBlockNum()) {
    int ids[TOPK];
    float weights[TOPK];
    int route = token * TOPK;
    io.Read((__gm__ int32_t *)topk + route / 8 * 8, 16);
    for (int k = 0; k < TOPK; ++k) ids[k] = io.ub.GetValue(route % 8 + k);
    io.Read((__gm__ int32_t *)cfg[11] + route / 16 * 8, 16);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    Cast(fp[2816], bf, RoundMode::CAST_NONE, 32);
    SetFlag<HardEvent::V_S>(EVENT_ID0);
    WaitFlag<HardEvent::V_S>(EVENT_ID0);
    for (int k = 0; k < TOPK; ++k) weights[k] = fp.GetValue(2816 + route % 16 + k);
    Duplicate(fp[5376], 0.0f, H);
    PipeBarrier<PIPE_V>();
    // Prime slot0. Subsequent loads alternate; a consumed Cast releases its slot.
    GlobalTensor<bfloat16_t> remote;
    remote.SetGlobalBuffer((__gm__ bfloat16_t *)cfg[1 + ids[0] / COLLECT_EXPERTS] + 128 + route * H);
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
    DataCopy(bf[256], remote, H);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
    for (int k = 0; k < TOPK; ++k) {
      if (k + 1 < TOPK) {
        auto nextEvent = ((k + 1) & 1) ? EVENT_ID2 : EVENT_ID1;
        remote.SetGlobalBuffer((__gm__ bfloat16_t *)cfg[1 + ids[k + 1] / COLLECT_EXPERTS] + 128 + (route + k + 1) * H);
        WaitFlag<HardEvent::V_MTE2>(nextEvent);
        DataCopy(bf[256 + ((k + 1) & 1) * H], remote, H);
        SetFlag<HardEvent::MTE2_V>(nextEvent);
      }
      auto event = (k & 1) ? EVENT_ID2 : EVENT_ID1;
      WaitFlag<HardEvent::MTE2_V>(event);
      Cast(fp[2816], bf[256 + (k & 1) * H], RoundMode::CAST_NONE, H);
      SetFlag<HardEvent::V_MTE2>(event);
      PipeBarrier<PIPE_V>();
      Muls(fp[2816], fp[2816], weights[k], H);
      PipeBarrier<PIPE_V>();
      Add(fp[5376], fp[5376], fp[2816], H);
      PipeBarrier<PIPE_V>();
    }
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    Cast(bf[16384], fp[5376], RoundMode::CAST_RINT, H);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    GlobalTensor<bfloat16_t> output;
    output.SetGlobalBuffer((__gm__ bfloat16_t *)cfg[12] + token * H);
    DataCopy(output, bf[16384], H);
    SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
    // Next token may overwrite the FP32 scratch only after the current V tail.
    SetFlag<HardEvent::V_S>(EVENT_ID0);
    WaitFlag<HardEvent::V_S>(EVENT_ID0);
  }
  WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
  WaitFlag<HardEvent::V_MTE2>(EVENT_ID2);
  WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
}
META(neural_collect_pipelined)
extern "C" __global__ __aicore__ void
neural_promote(GM_ADDR cfgaddr, GM_ADDR unused, GM_ADDR unused2) {
  if (GetBlockIdx() != 0)
    return;
  auto cfg = (__gm__ int64_t *)cfgaddr;
  if (!cfg[8] || cfg[15] != 1)
    return;
  IO io;
  io.Init();
  auto src = (__gm__ int32_t *)cfg[0];
  int generation = io.Flag(src);
  if (generation > 0)
    io.Publish(src + 16, generation);
}

META(neural_promote)
extern "C" __global__ __aicore__ void
neural_retire(GM_ADDR cfgaddr, GM_ADDR unused, GM_ADDR unused2) {
  if (GetBlockIdx() != 0)
    return;
  auto cfg = (__gm__ int64_t *)cfgaddr;
  if (!cfg[8])
    return;
  IO io;
  io.Init();
  if (io.Flag((__gm__ int32_t *)cfg[7]) < 0)
    return;
  int gen = io.Flag((__gm__ int32_t *)cfg[0]);
  io.Publish((__gm__ int32_t *)cfg[7], gen);
}

META(neural_retire)
// Same generation/retirement contract; the whole-layer server already applied
// all K weights in the original fixed FP32 order, so collect is only a copy.
extern "C" __global__ __aicore__ void
neural_collect_reduced(GM_ADDR cfgaddr,GM_ADDR unused,GM_ADDR unused2) {
  auto cfg=(__gm__ int64_t*)cfgaddr;if(!cfg[8])return;
  IO io;io.Init();int gen=io.Flag((__gm__ int32_t*)cfg[0]),polls=0;
  while(io.Flag((__gm__ int32_t*)cfg[1])!=gen && ++polls<cfg[9]){}
  if(polls>=cfg[9]){io.Publish((__gm__ int32_t*)cfg[7],-100);return;}
  int words=cfg[6]*H/2;
  constexpr int TILE=8192;
  for(int off=GetBlockIdx()*TILE;off<words;off+=GetBlockNum()*TILE){
    int n=words-off<TILE?words-off:TILE;
    io.Read((__gm__ int32_t*)cfg[1]+64+off,n);
    io.Write((__gm__ int32_t*)cfg[12]+off,n);
  }
}
META(neural_collect_reduced)
