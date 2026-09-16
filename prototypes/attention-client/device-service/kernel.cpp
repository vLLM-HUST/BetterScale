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
  int selected = 0, batchPolls = 0;
  if (cfg[5] && !status) {
    for (int poll = 0; poll < cfg[6]; ++poll) {
      // Bit1 is the forced diagnostic barrier. Bit3 is bounded coalescing:
      // its budget starts only after work exists; an absent source cannot
      // hold a ready source indefinitely. Finished sources need no wait.
      if (cfg[12] & (2 | 8)) {
        int ready = 0, remaining = 0;
        for (int c = 0; c < 2; ++c)
          if (finished[c] < cfg[4]) {
            ++remaining;
            if (io.Flag((__gm__ int32_t *)cfg[2 + c]) == finished[c] + 1)
              ++ready;
          }
        if (ready < remaining) {
          if ((cfg[12] & 2) || ready == 0)
            continue;
          if (batchPolls < cfg[13]) {
            ++batchPolls;
            continue;
          }
        }
      }
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
  if (cfg[12]) {
    // Count -> prefix -> assign, as in grouped dispatch. Do not rescan every
    // route once per expert: topology changes need not make metadata O(E*N*K).
    int counts[GROUPS], cursor[GROUPS];
    for (int g = 0; g < GROUPS; ++g)
      counts[g] = 0;
    for (int c = 0; c < 2; ++c)
      if (gen[c])
        for (int i = 0; i < rows[c] * TOPK; ++i)
          if (ids[c][i] / 64 == cfg[11])
            ++counts[layer[c] * 64 + ids[c][i] % 64];
    for (int g = 0; g < GROUPS; ++g) {
      cursor[g] = total;
      total += counts[g];
      ends[g] = total;
    }
    for (int c = 0; c < 2; ++c)
      if (gen[c])
        for (int i = 0; i < rows[c] * TOPK; ++i)
          if (ids[c][i] / 64 == cfg[11])
            map[c][i] = cursor[layer[c] * 64 + ids[c][i] % 64]++;
  } else {
    for (int g = 0; g < GROUPS; ++g) {
      for (int c = 0; c < 2; ++c)
        if (gen[c] && layer[c] == g / 64) {
          for (int i = 0; i < rows[c] * TOPK; ++i)
            if (ids[c][i] / 64 == cfg[11] && ids[c][i] % 64 == g % 64) {
              auto src =
                  (__gm__ int32_t *)cfg[2 + c] + 1024 + (i / TOPK) * H / 2;
              if (!cfg[12]) {
                io.Read(src, H / 2);
                io.Write((__gm__ int32_t *)packed + total * H / 2, H / 2);
              }
              map[c][i] = total++;
            }
        }
      ends[g] = total;
    }
  }
  for (int c = 0; c < 2; ++c) {
    for (int j = 0; j < 8; ++j)
      io.ub.SetValue(j, j == 0 ? gen[c] : (j == 1 ? rows[c] : 0));
    for (int j = 0; j < ROUTES; ++j)
      io.ub.SetValue(8 + j, map[c][j]);
    io.Write(batch + c * MAPSTRIDE, MAPSTRIDE);
  }
  for (int g = 0; g < GROUPS; ++g) {
    io.ub.SetValue(2 * g,
                   g == GROUPS - 1 && !(cfg[12] & 4) ? 2 * ROUTES : ends[g]);
    io.ub.SetValue(2 * g + 1, 0);
  }
  io.Write(groups, 2 * GROUPS);
  int fields[8] = {finished[0], finished[1], wave,       status,
                   selected,    total,       batchPolls, 0};
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
    for (int i = 0; !cfg[12] && i < n * TOPK; ++i) {
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
  int record[8] = {fields[4], fields[5], gen[0],       gen[1],
                   fields[3], fields[6], int(cfg[12]), 0};
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
  if (cfg[10])
    return; // Parallel collect and retirement follow in the graph.
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

// Topology adapter: DFC-like row ownership, with graph kernel boundaries
// instead of an all-EP wave barrier. The selector writes maps, all AIVs move
// disjoint rows, then the next graph node may consume/publish. No atomics or
// zero placeholders.
extern "C" __global__ __aicore__ void
neural_pack(GM_ADDR cfgaddr, GM_ADDR packed, GM_ADDR unused) {
  auto cfg = (__gm__ int64_t *)cfgaddr;
  if (!cfg[5])
    return;
  IO io;
  io.Init();
  for (int c = 0; c < 2; ++c) {
    io.Read((__gm__ int32_t *)cfg[8] + c * MAPSTRIDE, MAPSTRIDE);
    int generation = io.ub.GetValue(0), n = io.ub.GetValue(1);
    int map[ROUTES];
    for (int i = 0; i < ROUTES; ++i)
      map[i] = io.ub.GetValue(8 + i);
    if (!generation)
      continue;
    for (int i = GetBlockIdx(); i < n * TOPK; i += GetBlockNum())
      if (map[i] >= 0) {
        io.Read((__gm__ int32_t *)cfg[2 + c] + 1024 + (i / TOPK) * H / 2,
                H / 2);
        io.Write((__gm__ int32_t *)packed + map[i] * H / 2, H / 2);
      }
  }
}
extern "C" __global__ __aicore__ void
neural_scatter(GM_ADDR cfgaddr, GM_ADDR computed, GM_ADDR unused) {
  auto cfg = (__gm__ int64_t *)cfgaddr;
  if (!cfg[5])
    return;
  IO io;
  io.Init();
  for (int c = 0; c < 2; ++c) {
    io.Read((__gm__ int32_t *)cfg[8] + c * MAPSTRIDE, MAPSTRIDE);
    int generation = io.ub.GetValue(0), n = io.ub.GetValue(1);
    int map[ROUTES];
    for (int i = 0; i < ROUTES; ++i)
      map[i] = io.ub.GetValue(8 + i);
    if (!generation)
      continue;
    for (int i = GetBlockIdx(); i < n * TOPK; i += GetBlockNum())
      if (map[i] >= 0) {
        io.Read((__gm__ int32_t *)computed + map[i] * H / 2, H / 2);
        io.Write((__gm__ int32_t *)cfg[c] + 64 + i * H / 2, H / 2);
      }
  }
}
// Every core pulls disjoint live routes from their unique owner. Unowned/stale
// slots are never read; graph completion joins these reads before counter
// advance.
extern "C" __global__ __aicore__ void
neural_collect(GM_ADDR cfgaddr, GM_ADDR unused, GM_ADDR topk) {
  auto cfg = (__gm__ int64_t *)cfgaddr;
  if (!cfg[6])
    return;
  IO io;
  io.Init();
  int gen = io.Flag((__gm__ int32_t *)cfg[0]), n = cfg[4];
  for (int s = 0; s < 2; ++s) {
    int polls = 0;
    while (io.Flag((__gm__ int32_t *)cfg[1 + s]) != gen && ++polls < cfg[7]) {
    }
    if (polls >= cfg[7]) {
      io.Publish((__gm__ int32_t *)cfg[5], -100);
      return;
    }
  }
  io.Read((__gm__ int32_t *)topk, n * TOPK);
  int ids[ROUTES];
  for (int i = 0; i < n * TOPK; ++i)
    ids[i] = io.ub.GetValue(i);
  for (int i = GetBlockIdx(); i < n * TOPK; i += GetBlockNum()) {
    int owner = ids[i] / 64;
    io.Read((__gm__ int32_t *)cfg[1 + owner] + 64 + i * H / 2, H / 2);
    io.Write((__gm__ int32_t *)cfg[8] + i * H / 2, H / 2);
  }
}
// Token-owned gather/reduce: exported route generations replace the initial
// all-server join. One mover owns each accumulator; server-local output remains
// immutable until the source publishes its next generation after final drain.
extern "C" __global__ __aicore__ void
neural_collect_reduce(GM_ADDR cfgaddr, GM_ADDR unused, GM_ADDR topk) {
  auto cfg = (__gm__ int64_t *)cfgaddr;
  if (!cfg[6])
    return;
  uint64_t began = GetSystemCycle(), firstRead = 0, reduced = 0;
  int polling = 0;
  IO io;
  io.Init();
  int gen = io.Flag((__gm__ int32_t *)cfg[0]), n = cfg[4];
  int ids[256];
  float weights[256];
  io.Read((__gm__ int32_t *)topk, n * 8);
  for (int i = 0; i < n * 8; ++i)
    ids[i] = io.ub.GetValue(i);
  io.Read((__gm__ int32_t *)cfg[11], 128);
  auto bf = io.buf.Get<bfloat16_t>();
  auto fp = io.buf.Get<float>();
  PipeBarrier<PIPE_ALL>();
  Cast(fp[1024], bf, RoundMode::CAST_NONE, 256);
  PipeBarrier<PIPE_ALL>();
  for (int i = 0; i < n * 8; ++i)
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
      if (token >= n || masks[t] == 255)
        continue;
      for (int owner = 0; owner < 2; ++owner) {
        bool needed = false;
        for (int k = 0; k < 8; ++k)
          needed |= !(masks[t] & (1 << k)) && ids[token * 8 + k] / 64 == owner;
        if (!needed)
          continue;
        // Read all eight cache-line flags together, not eight tiny DMA polls.
        ++polling;
        io.Read((__gm__ int32_t *)cfg[1 + owner] + 64 + 256 * H / 2 +
                    token * 8 * 16,
                128);
        int ready = 0;
        for (int k = 0; k < 8; ++k)
          if (io.ub.GetValue(k * 16) == gen)
            ready |= 1 << k;
        for (int k = 0; k < 8; ++k) {
          int bit = 1 << k, route = token * 8 + k;
          if ((masks[t] & bit) || !(ready & bit) || ids[route] / 64 != owner)
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
      if (masks[t] == 255) {
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
    if (idle >= cfg[7]) {
      io.Publish((__gm__ int32_t *)cfg[5], -101);
      return;
    }
  }
  reduced = GetSystemCycle();
  // Do not authorize input/frame reuse before producer kernels also retire.
  // Empty local expert sets still owe a server-wide completion generation.
  for (int owner = 0; owner < 2; ++owner) {
    int polls = 0;
    while (io.Flag((__gm__ int32_t *)cfg[1 + owner]) != gen)
      if (++polls >= cfg[7]) {
        io.Publish((__gm__ int32_t *)cfg[5], -102);
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
  if (!cfg[6])
    return;
  IO io;
  io.Init();
  if (io.Flag((__gm__ int32_t *)cfg[5]) < 0)
    return;
  int gen = io.Flag((__gm__ int32_t *)cfg[0]);
  io.Publish((__gm__ int32_t *)cfg[5], gen);
}
#define SERVICE_META(name)                                                     \
  static const struct FunLevelKType name##_meta                                \
      __attribute__((used, section(".ascend.meta." #name))) = {                \
          {F_TYPE_KTYPE, sizeof(unsigned int), K_TYPE_AIV}};
SERVICE_META(neural_pack)
SERVICE_META(neural_scatter)
SERVICE_META(neural_collect)
SERVICE_META(neural_retire)
SERVICE_META(neural_collect_reduce)
