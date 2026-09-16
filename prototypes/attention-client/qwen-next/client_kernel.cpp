// IO ordering reused from the validated 3532418 client/server prototype.
#include "kernel_operator.h"
using namespace AscendC;
constexpr int H = 2048, TOPK = 10, ROUTES = 320;
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
    io.ub.SetValue(
        j, j == 0 ? gen
                  : (j == 1 ? cfg[5] : (j == 2 ? n : (j == 3 ? cfg[15] : 0))));
  io.Write(src + 8);
  io.Publish(src, gen);
}
// Enqueued AFTER native shared expert and BEFORE collect, inside the same
// outer graph/stream. It publishes critical-path urgency without a host fence.
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
// Token-owned gather/reduce: exported route generations replace the initial
// all-server join. One mover owns each accumulator; server-local output remains
// immutable until the source publishes its next generation after final drain.
extern "C" __global__ __aicore__ void
neural_collect_reduce(GM_ADDR cfgaddr, GM_ADDR unused, GM_ADDR topk) {
  auto cfg = (__gm__ int64_t *)cfgaddr;
  if (!cfg[8])
    return;
  uint64_t began = GetSystemCycle(), firstRead = 0, reduced = 0;
  int polling = 0;
  IO io;
  io.Init();
  int gen = io.Flag((__gm__ int32_t *)cfg[0]), n = cfg[6];
  int ids[ROUTES];
  float weights[ROUTES];
  io.Read((__gm__ int32_t *)topk, (n * TOPK + 7) / 8 * 8);
  for (int i = 0; i < n * TOPK; ++i)
    ids[i] = io.ub.GetValue(i);
  io.Read((__gm__ int32_t *)cfg[11], ROUTES / 2);
  auto bf = io.buf.Get<bfloat16_t>();
  auto fp = io.buf.Get<float>();
  PipeBarrier<PIPE_ALL>();
  Cast(fp[1024], bf, RoundMode::CAST_NONE, ROUTES);
  PipeBarrier<PIPE_ALL>();
  for (int i = 0; i < n * TOPK; ++i)
    weights[i] = fp.GetValue(1024 + i);
  int masks[2] = {0, 0};
  int tokens[2] = {int(GetBlockIdx()), int(GetBlockIdx() + GetBlockNum())};
  Duplicate(fp[3072], 0.0f, 4096);
  PipeBarrier<PIPE_V>();
  int remaining = (tokens[0] < n) + (tokens[1] < n);
  int idle = 0;
  while (remaining) {
    bool progress = false;
    for (int t = 0; t < 2; ++t) {
      int token = tokens[t];
      if (token >= n || masks[t] == ((1 << TOPK) - 1))
        continue;
      for (int owner = 0; owner < 4; ++owner) {
        bool needed = false;
        for (int k = 0; k < TOPK; ++k)
          needed |=
              !(masks[t] & (1 << k)) && ids[token * TOPK + k] / 128 == owner;
        if (!needed)
          continue;
        // Read all top-k cache-line flags together rather than separate polls.
        ++polling;
        io.Read((__gm__ int32_t *)cfg[1 + owner] + 64 + ROUTES * H / 2 +
                    token * TOPK * 16,
                TOPK * 16);
        int ready = 0;
        for (int k = 0; k < TOPK; ++k)
          if (io.ub.GetValue(k * 16) == gen)
            ready |= 1 << k;
        for (int k = 0; k < TOPK; ++k) {
          int bit = 1 << k, route = token * TOPK + k;
          if ((masks[t] & bit) || !(ready & bit) || ids[route] / 128 != owner)
            continue;
          if (!firstRead)
            firstRead = GetSystemCycle();
          io.Read((__gm__ int32_t *)cfg[1 + owner] + 64 + route * H / 2, H / 2);
          PipeBarrier<PIPE_ALL>();
          Cast(fp[1024], bf, RoundMode::CAST_NONE, H);
          PipeBarrier<PIPE_V>();
          Muls(fp[1024], fp[1024], weights[route], H);
          PipeBarrier<PIPE_V>();
          Add(fp[3072 + t * H], fp[3072 + t * H], fp[1024], H);
          PipeBarrier<PIPE_V>();
          masks[t] |= bit;
          progress = true;
        }
      }
      if (masks[t] == ((1 << TOPK) - 1)) {
        Cast(bf, fp[3072 + t * H], RoundMode::CAST_RINT, H);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        io.Write((__gm__ int32_t *)cfg[12] + token * H / 2, H / 2);
        --remaining;
      }
    }
    idle = progress ? 0 : idle + 1;
    if (!progress && cfg[14]) {
      uint64_t until = GetSystemCycle() + cfg[14];
      while (GetSystemCycle() < until) {
      }
    }
    if (idle >= cfg[9]) {
      io.Publish((__gm__ int32_t *)cfg[7], -101);
      return;
    }
  }
  reduced = GetSystemCycle();
  // Do not authorize input/frame reuse before producer kernels also retire.
  // Empty local expert sets still owe a server-wide completion generation.
  for (int owner = 0; owner < 4; ++owner) {
    int polls = 0;
    while (io.Flag((__gm__ int32_t *)cfg[1 + owner]) != gen)
      if (++polls >= cfg[9]) {
        io.Publish((__gm__ int32_t *)cfg[7], -102);
        return;
      }
  }
  if (cfg[13]) {
    auto times = io.buf.Get<int64_t>();
    times.SetValue(0, began);
    times.SetValue(1, firstRead);
    times.SetValue(2, reduced);
    times.SetValue(3, GetSystemCycle());
    times.SetValue(4, polling);
    io.Write((__gm__ int32_t *)cfg[13] + (gen * 16 + GetBlockIdx()) * 16, 16);
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

META(neural_collect_reduce)

META(neural_promote)
