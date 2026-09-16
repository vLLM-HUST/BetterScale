// IO ordering reused from the validated 3532418 client/server prototype.
#include "kernel_operator.h"
using namespace AscendC;
constexpr int H = 2048, GROUPS = 128, ROWS = 8, CAP = 16, STRIDE = 16384;
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

// Result/source slots are 64KiB; descriptor at32B, payload at256B.
// prepare cfg:
// local,peer0,peer1,clients,tasks,enabled,limit,state,batch,groups,trace.
extern "C" __global__ __aicore__ void
queue_prepare(GM_ADDR cfgaddr, GM_ADDR packed, GM_ADDR unused) {
  if (GetBlockIdx() != 0)
    return;
  auto cfg = (__gm__ int64_t *)cfgaddr;
  auto state = (__gm__ int32_t *)cfg[7];
  auto batch = (__gm__ int32_t *)cfg[8];
  auto groups = (__gm__ int32_t *)cfg[9];
  auto x = (__gm__ int32_t *)packed;
  IO io;
  io.Init();
  int gen[2] = {0}, rows[2] = {0}, group[2] = {0}, offset[2] = {0};
  io.Read(state);
  int finished[2] = {io.ub.GetValue(0), io.ub.GetValue(1)};
  int status = io.ub.GetValue(3), wave = io.ub.GetValue(2);
  int count = 0, total = 0;
  if (cfg[5] && status == 0) {
    int polls = 0;
    while (polls++ < cfg[6]) {
      bool allDone = true;
      for (int c = 0; c < cfg[3]; ++c) {
        if (finished[c] >= cfg[4])
          continue;
        allDone = false;
        auto src = (__gm__ int32_t *)cfg[1 + c];
        if (io.Flag(src) != finished[c] + 1)
          continue;
        io.Read(src + 8);
        int g = io.ub.GetValue(0), task = io.ub.GetValue(1),
            l = io.ub.GetValue(2), e = io.ub.GetValue(3), n = io.ub.GetValue(5);
        if (g != finished[c] + 1 || task != finished[c] || l < 0 || l >= 2 ||
            e < 0 || e >= 64 || n < 1 || n > ROWS) {
          status = -2;
          break;
        }
        gen[c] = g;
        rows[c] = n;
        group[c] = l * 64 + e;
        ++count;
      }
      if (status || count || allDone)
        break;
    }
    if (!count && !status &&
        (finished[0] < cfg[4] || (cfg[3] == 2 && finished[1] < cfg[4])))
      status = -1;
  }
  if (status) {
    count = 0;
    gen[0] = gen[1] = rows[0] = rows[1] = 0;
  }
  // Scatter compact packets into expert-major rows. GMM reads device cumsums.
  for (int g = 0; g < GROUPS; ++g) {
    for (int c = 0; c < cfg[3]; ++c)
      if (gen[c] && group[c] == g) {
        offset[c] = total;
        auto src = (__gm__ int32_t *)cfg[1 + c];
        io.Read(src + 64, rows[c] * H / 2);
        io.Write(x + total * H / 2, rows[c] * H / 2);
        total += rows[c];
      }
    // int64 count table is published in one DMA below; keep it out of IO
    // scratch.
  }
  for (int c = 0; c < 2; ++c) {
    int fields[8] = {gen[c], rows[c], group[c], offset[c], 0, 0, 0, 0};
    for (int j = 0; j < 8; ++j)
      io.ub.SetValue(j, fields[j]);
    io.Write(batch + c * 8);
  }
  int cumulative = 0;
  for (int g = 0; g < GROUPS; ++g) {
    for (int c = 0; c < cfg[3]; ++c)
      if (gen[c] && group[c] == g)
        cumulative += rows[c];
    io.ub.SetValue(2 * g, g == GROUPS - 1 ? CAP : cumulative);
    io.ub.SetValue(2 * g + 1, 0);
  }
  io.Write(groups, GROUPS * 2);
  int fields[8] = {finished[0], finished[1], wave, status, count, total, 0, 0};
  for (int j = 0; j < 8; ++j)
    io.ub.SetValue(j, fields[j]);
  io.Write(state);
}

extern "C" __global__ __aicore__ void
queue_complete(GM_ADDR cfgaddr, GM_ADDR computed, GM_ADDR unused) {
  if (GetBlockIdx() != 0)
    return;
  auto cfg = (__gm__ int64_t *)cfgaddr;
  if (!cfg[5])
    return;
  auto state = (__gm__ int32_t *)cfg[7];
  auto batch = (__gm__ int32_t *)cfg[8];
  auto local = (__gm__ int32_t *)cfg[0];
  auto y = (__gm__ int32_t *)computed;
  IO io;
  io.Init();
  io.Read(state);
  int fields[8];
  for (int j = 0; j < 8; ++j)
    fields[j] = io.ub.GetValue(j);
  int gen[2] = {0}, group[2] = {-1, -1};
  for (int c = 0; c < cfg[3]; ++c) {
    io.Read(batch + c * 8);
    gen[c] = io.ub.GetValue(0);
    int n = io.ub.GetValue(1), offset = io.ub.GetValue(3);
    group[c] = io.ub.GetValue(2);
    if (!gen[c])
      continue;
    io.Read(y + offset * H / 2, n * H / 2);
    io.Write(local + c * STRIDE + 64, n * H / 2);
    fields[c] = gen[c];
  }
  // Finish every payload copy before publishing any DONE (IO scratch is
  // shared).
  for (int c = 0; c < cfg[3]; ++c)
    if (gen[c])
      io.Publish(local + c * STRIDE, gen[c]);
  auto trace = (__gm__ int32_t *)cfg[10] + fields[2] * 8;
  int record[8] = {fields[4], fields[5], gen[0],    gen[1],
                   group[0],  group[1],  fields[3], 0};
  for (int j = 0; j < 8; ++j)
    io.ub.SetValue(j, record[j]);
  io.Write(trace);
  ++fields[2];
  for (int j = 0; j < 8; ++j)
    io.ub.SetValue(j, fields[j]);
  io.Write(state);
}

// client cfg: local,result,client_index,tasks,enabled,limit,status,outputs.
extern "C" __global__ __aicore__ void
queue_client(GM_ADDR cfgaddr, GM_ADDR descaddr, GM_ADDR inputsaddr) {
  if (GetBlockIdx() != 0)
    return;
  auto cfg = (__gm__ int64_t *)cfgaddr;
  auto status = (__gm__ int32_t *)cfg[6];
  IO io;
  io.Init();
  if (!cfg[4]) {
    io.Publish(status, 1);
    return;
  }
  auto src = (__gm__ int32_t *)cfg[0];
  auto dst = (__gm__ int32_t *)cfg[1] + cfg[2] * STRIDE;
  auto desc = (__gm__ int32_t *)descaddr;
  auto inputs = (__gm__ int32_t *)inputsaddr;
  auto out = (__gm__ int32_t *)cfg[7];
  for (int task = 0; task < cfg[3]; ++task) {
    io.Read(desc + task * 8);
    int gen = io.ub.GetValue(0), n = io.ub.GetValue(5);
    if (gen != task + 1 || n < 1 || n > ROWS) {
      io.Publish(status, -2);
      return;
    }
    io.Read(inputs + task * ROWS * H / 2, n * H / 2);
    io.Write(src + 64, n * H / 2);
    io.Read(desc + task * 8);
    io.Write(src + 8);
    io.Publish(src, gen);
    int polls = 0;
    while (io.Flag(dst) != gen && ++polls < cfg[5]) {
    }
    if (polls >= cfg[5]) {
      io.Publish(status, -1);
      return;
    }
    io.Read(dst + 64, n * H / 2);
    io.Write(out + task * ROWS * H / 2, n * H / 2);
  }
  io.Publish(status, 1);
}
static const struct FunLevelKType prep_meta
    __attribute__((used, section(".ascend.meta.queue_prepare"))) = {
        {F_TYPE_KTYPE, sizeof(unsigned int), K_TYPE_AIV}};
static const struct FunLevelKType done_meta
    __attribute__((used, section(".ascend.meta.queue_complete"))) = {
        {F_TYPE_KTYPE, sizeof(unsigned int), K_TYPE_AIV}};
static const struct FunLevelKType client_meta
    __attribute__((used, section(".ascend.meta.queue_client"))) = {
        {F_TYPE_KTYPE, sizeof(unsigned int), K_TYPE_AIV}};

// Neural wire: source descriptor32B, top-k IDs256B, BF16 input4096B.
// Each source has one outstanding layer; both servers must finish before reuse.
constexpr int TOKENS = 32, TOPK = 8, ROUTES = TOKENS * TOPK,
              MAPSTRIDE = 8 + ROUTES;
__aicore__ inline void CopyWords(IO &io, __gm__ int32_t *src,
                                 __gm__ int32_t *dst, int words) {
  for (int offset = 0; offset < words; offset += 8192) {
    int n = words - offset;
    if (n > 8192)
      n = 8192;
    io.Read(src + offset, n);
    io.Write(dst + offset, n);
  }
}
// cfg:
// out0,out1,src0,src1,tasks,enabled,limit,state,batch,groups,trace,expert_owner.
extern "C" __global__ __aicore__ void
neural_prepare(GM_ADDR cfgaddr, GM_ADDR packed, GM_ADDR unused) {
  if (GetBlockIdx() != 0)
    return;
  auto cfg = (__gm__ int64_t *)cfgaddr;
  auto state = (__gm__ int32_t *)cfg[7];
  auto batch = (__gm__ int32_t *)cfg[8];
  auto groups = (__gm__ int32_t *)cfg[9];
  IO io;
  io.Init();
  io.Read(state);
  int finished[2] = {io.ub.GetValue(0), io.ub.GetValue(1)},
      wave = io.ub.GetValue(2), status = io.ub.GetValue(3);
  int gen[2] = {0}, rows[2] = {0}, layer[2] = {0}, ids[2][ROUTES],
      map[2][ROUTES];
  for (int c = 0; c < 2; ++c)
    for (int i = 0; i < ROUTES; ++i) {
      ids[c][i] = -1;
      map[c][i] = -1;
    }
  int selected = 0;
  if (cfg[5] && !status) {
    for (int poll = 0; poll < cfg[6]; ++poll) {
      for (int c = 0; c < 2; ++c) {
        if (finished[c] >= cfg[4])
          continue;
        auto src = (__gm__ int32_t *)cfg[2 + c];
        if (io.Flag(src) != finished[c] + 1)
          continue;
        io.Read(src + 8);
        gen[c] = io.ub.GetValue(0);
        layer[c] = io.ub.GetValue(1);
        rows[c] = io.ub.GetValue(2);
        if (gen[c] != finished[c] + 1 || layer[c] < 0 || layer[c] >= 2 ||
            rows[c] < 1 || rows[c] > TOKENS) {
          status = -2;
          break;
        }
        io.Read(src + 64, rows[c] * TOPK);
        for (int i = 0; i < rows[c] * TOPK; ++i) {
          ids[c][i] = io.ub.GetValue(i);
          if (ids[c][i] < 0 || ids[c][i] >= 128)
            status = -3;
        }
        ++selected;
      }
      if (selected || status ||
          (finished[0] >= cfg[4] && finished[1] >= cfg[4]))
        break;
    }
    if (!selected && !status && (finished[0] < cfg[4] || finished[1] < cfg[4]))
      status = -1;
  }
  if (status) {
    gen[0] = gen[1] = rows[0] = rows[1] = selected = 0;
  }
  int total = 0, ends[GROUPS];
  for (int g = 0; g < GROUPS; ++g) {
    for (int c = 0; c < 2; ++c)
      if (gen[c] && layer[c] == g / 64) {
        for (int i = 0; i < rows[c] * TOPK; ++i)
          if (ids[c][i] / 64 == cfg[11] && ids[c][i] % 64 == g % 64) {
            auto src = (__gm__ int32_t *)cfg[2 + c] + 1024 + (i / TOPK) * H / 2;
            io.Read(src, H / 2);
            io.Write((__gm__ int32_t *)packed + total * H / 2, H / 2);
            map[c][i] = total++;
          }
      }
    ends[g] = total;
  }
  for (int c = 0; c < 2; ++c) {
    for (int j = 0; j < 8; ++j)
      io.ub.SetValue(j, j == 0 ? gen[c] : (j == 1 ? rows[c] : 0));
    for (int j = 0; j < ROUTES; ++j)
      io.ub.SetValue(8 + j, map[c][j]);
    io.Write(batch + c * MAPSTRIDE, MAPSTRIDE);
  }
  for (int g = 0; g < GROUPS; ++g) {
    io.ub.SetValue(2 * g, g == GROUPS - 1 ? 2 * ROUTES : ends[g]);
    io.ub.SetValue(2 * g + 1, 0);
  }
  io.Write(groups, 2 * GROUPS);
  int fields[8] = {finished[0], finished[1], wave, status,
                   selected,    total,       0,    0};
  for (int j = 0; j < 8; ++j)
    io.ub.SetValue(j, fields[j]);
  io.Write(state);
}
extern "C" __global__ __aicore__ void
neural_complete(GM_ADDR cfgaddr, GM_ADDR computed, GM_ADDR unused) {
  if (GetBlockIdx() != 0)
    return;
  auto cfg = (__gm__ int64_t *)cfgaddr;
  if (!cfg[5])
    return;
  auto state = (__gm__ int32_t *)cfg[7];
  auto batch = (__gm__ int32_t *)cfg[8];
  IO io;
  io.Init();
  io.Read(state);
  int fields[8];
  for (int j = 0; j < 8; ++j)
    fields[j] = io.ub.GetValue(j);
  int gen[2] = {0};
  for (int c = 0; c < 2; ++c) {
    io.Read(batch + c * MAPSTRIDE, MAPSTRIDE);
    gen[c] = io.ub.GetValue(0);
    int n = io.ub.GetValue(1);
    int map[ROUTES];
    for (int j = 0; j < ROUTES; ++j)
      map[j] = io.ub.GetValue(8 + j);
    if (!gen[c])
      continue;
    auto dst = (__gm__ int32_t *)cfg[c] + 64;
    for (int i = 0; i < n * TOPK; ++i) {
      if (map[i] >= 0)
        io.Read((__gm__ int32_t *)computed + map[i] * H / 2, H / 2);
      else
        for (int j = 0; j < H / 2; ++j)
          io.ub.SetValue(j, 0);
      io.Write(dst + i * H / 2, H / 2);
    }
    fields[c] = gen[c];
  }
  for (int c = 0; c < 2; ++c)
    if (gen[c])
      io.Publish((__gm__ int32_t *)cfg[c], gen[c]);
  int record[8] = {fields[4], fields[5], gen[0], gen[1], fields[3], 0, 0, 0};
  for (int j = 0; j < 8; ++j)
    io.ub.SetValue(j, record[j]);
  io.Write((__gm__ int32_t *)cfg[10] + fields[2] * 8);
  ++fields[2];
  for (int j = 0; j < 8; ++j)
    io.ub.SetValue(j, fields[j]);
  io.Write(state);
}
// cfg:
// source,out0,out1,layer,rows,generation,enabled,limit,local_out0,local_out1.
extern "C" __global__ __aicore__ void
neural_client(GM_ADDR cfgaddr, GM_ADDR hidden, GM_ADDR topk) {
  if (GetBlockIdx() != 0)
    return;
  auto cfg = (__gm__ int64_t *)cfgaddr;
  if (!cfg[6])
    return;
  auto src = (__gm__ int32_t *)cfg[0];
  auto counter = (__gm__ int32_t *)cfg[5];
  IO io;
  io.Init();
  int gen = io.Flag(counter) + 1, n = cfg[4];
  CopyWords(io, (__gm__ int32_t *)hidden, src + 1024, n * H / 2);
  io.Read((__gm__ int32_t *)topk, n * TOPK);
  io.Write(src + 64, n * TOPK);
  for (int j = 0; j < 8; ++j)
    io.ub.SetValue(j, j == 0 ? gen : (j == 1 ? cfg[3] : (j == 2 ? n : 0)));
  io.Write(src + 8);
  io.Publish(src, gen);
  for (int s = 0; s < 2; ++s) {
    auto remote = (__gm__ int32_t *)cfg[1 + s];
    int polls = 0;
    while (io.Flag(remote) != gen && ++polls < cfg[7]) {
    }
    if (polls >= cfg[7]) {
      io.Publish(counter, -100);
      return;
    }
    CopyWords(io, remote + 64, (__gm__ int32_t *)cfg[8 + s], n * TOPK * H / 2);
  }
  io.Publish(counter,
             gen); // both raw contributions consumed, now source reusable
}
static const struct FunLevelKType np_meta
    __attribute__((used, section(".ascend.meta.neural_prepare"))) = {
        {F_TYPE_KTYPE, sizeof(unsigned int), K_TYPE_AIV}};
static const struct FunLevelKType nc_meta
    __attribute__((used, section(".ascend.meta.neural_complete"))) = {
        {F_TYPE_KTYPE, sizeof(unsigned int), K_TYPE_AIV}};
static const struct FunLevelKType ni_meta
    __attribute__((used, section(".ascend.meta.neural_client"))) = {
        {F_TYPE_KTYPE, sizeof(unsigned int), K_TYPE_AIV}};
