// IO ordering reused from the validated 3532418 client/server prototype.
#include "kernel_operator.h"
using namespace AscendC;
constexpr int H = 2048, TOPK = 10, ROUTES = 320;
// Metadata = [generation, task, layer, expert, phase (0=D), rows, 0, 0].
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

// cfg: source,peer0..3,layer,rows,counter,enabled,limit,output.
extern "C" __global__ __aicore__ void
neural_client(GM_ADDR cfgaddr, GM_ADDR hidden, GM_ADDR topk) {
  if (GetBlockIdx() != 0)
    return;
  auto cfg = (__gm__ int64_t *)cfgaddr;
  if (!cfg[8])
    return;
  IO io;
  io.Init();
  auto src = (__gm__ int32_t *)cfg[0];
  int gen = io.Flag((__gm__ int32_t *)cfg[7]) + 1, n = cfg[6];
  for (int row = 0; row < n; ++row) {
    io.Read((__gm__ int32_t *)hidden + row * H / 2, H / 2);
    io.Write(src + 1024 + row * H / 2, H / 2);
  }
  int padded = (n * TOPK + 7) / 8 * 8;
  io.Read((__gm__ int32_t *)topk, padded);
  io.Write(src + 64, padded);
  for (int j = 0; j < 8; ++j)
    io.ub.SetValue(j, j == 0 ? gen : (j == 1 ? cfg[5] : (j == 2 ? n : 0)));
  io.Write(src + 8);
  io.Publish(src, gen);
}
extern "C" __global__ __aicore__ void
neural_collect(GM_ADDR cfgaddr, GM_ADDR unused, GM_ADDR topk) {
  auto cfg = (__gm__ int64_t *)cfgaddr;
  if (!cfg[8])
    return;
  IO io;
  io.Init();
  int gen = io.Flag((__gm__ int32_t *)cfg[0]), n = cfg[6];
  for (int owner = 0; owner < 4; ++owner) {
    int polls = 0;
    while (io.Flag((__gm__ int32_t *)cfg[1 + owner]) != gen &&
           ++polls < cfg[9]) {
    }
    if (polls >= cfg[9]) {
      io.Publish((__gm__ int32_t *)cfg[7], -100);
      return;
    }
  }
  io.Read((__gm__ int32_t *)topk, (n * TOPK + 7) / 8 * 8);
  int ids[ROUTES];
  for (int i = 0; i < n * TOPK; ++i)
    ids[i] = io.ub.GetValue(i);
  for (int i = GetBlockIdx(); i < n * TOPK; i += GetBlockNum()) {
    int owner = ids[i] / 128;
    io.Read((__gm__ int32_t *)cfg[1 + owner] + 64 + i * H / 2, H / 2);
    io.Write((__gm__ int32_t *)cfg[10] + i * H / 2, H / 2);
  }
}
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
#define META(name)                                                             \
  static const struct FunLevelKType name##_meta                                \
      __attribute__((used, section(".ascend.meta." #name))) = {                \
          {F_TYPE_KTYPE, sizeof(unsigned int), K_TYPE_AIV}};
META(neural_client)
META(neural_collect)
META(neural_retire)
