#include "persistent_protocol.hpp"
#include "priority_policy.hpp"
using namespace AscendC;
using namespace Persistent;

__aicore__ inline int ScalarMin(int a, int b) { return a < b ? a : b; }
__aicore__ inline int ScalarMax(int a, int b) { return a > b ? a : b; }

// MTE completion precedes every flag publication. Control lines use scalar
// DCCI; payload reads/writes use DMA, not scalar cache.
struct Transfer {
  TPipe pipe;
  TBuf<TPosition::VECCALC> buf;
  LocalTensor<int32_t> words;
  __aicore__ inline void Init() {
    pipe.InitBuffer(buf, 32768);
    words = buf.Get<int32_t>();
  }
  __aicore__ inline void Read(__gm__ int32_t *p, int n) {
    GlobalTensor<int32_t> g;
    g.SetGlobalBuffer(p);
    DataCopy(words, g, n);
    SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
  }
  __aicore__ inline void Write(__gm__ int32_t *p, int n) {
    SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
    GlobalTensor<int32_t> g;
    g.SetGlobalBuffer(p);
    DataCopy(g, words, n);
    SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
  }
  __aicore__ inline int Flag(__gm__ int32_t *p) {
    Read(p, 8);
    return words.GetValue(0);
  }
  __aicore__ inline void Publish(__gm__ int32_t *p, int gen) {
    for (int i = 0; i < 8; ++i)
      words.SetValue(i, i ? 0 : gen);
    Write(p, 8);
  }
  __aicore__ inline void Copy(__gm__ int32_t *src, __gm__ int32_t *dst, int n) {
    Read(src, n);
    Write(dst, n);
  }
  __aicore__ inline void Activation(__gm__ bfloat16_t *src,
                                    __gm__ bfloat16_t *dst) {
    GlobalTensor<bfloat16_t> in, out;
    in.SetGlobalBuffer(src);
    out.SetGlobalBuffer(dst);
    auto b = buf.Get<bfloat16_t>();
    auto f = buf.Get<float>();
    // Input occupies0..3072 bytes; float temporaries start at4096.
    auto gate = f[1024], value = f[1792], tmp = f[2560];
    DataCopy(b, in, INNER * 2);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    Cast(gate, b, RoundMode::CAST_NONE, INNER);
    Cast(value, b[INNER], RoundMode::CAST_NONE, INNER);
    PipeBarrier<PIPE_V>();
    Muls(tmp, gate, -1.0f, INNER);
    PipeBarrier<PIPE_V>();
    Exp(tmp, tmp, INNER);
    PipeBarrier<PIPE_V>();
    Adds(tmp, tmp, 1.0f, INNER);
    PipeBarrier<PIPE_V>();
    Div(gate, gate, tmp, INNER);
    PipeBarrier<PIPE_V>();
    Mul(gate, gate, value, INNER);
    PipeBarrier<PIPE_V>();
    Cast(b, gate, RoundMode::CAST_RINT, INNER);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    DataCopy(out, b, INNER);
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
  }
};

// One SEND command retains both route maps across prefix/tail passes. All
// activation must finish before admission, otherwise waiting for down here
// could prevent the same AIV team from producing its required activation.
__aicore__ inline void ReturnStreaming(__gm__ int64_t *cfg, Transfer &io,
                                       int slot, int worker, int command) {
  auto ctrl = (__gm__ int32_t *)cfg[0];
  auto ptr = (__gm__ int64_t *)cfg[1] + slot * 16;
  int boundary = ctrl[VCMD * LINE + 3], generation = ctrl[VCMD * LINE + 4];
  int maps[2][ROUTES], rows[2], sources[2];
  for (int c = 0; c < 2; ++c) {
    io.Read((__gm__ int32_t *)ptr[5] + c * MAP, MAP);
    sources[c] = io.words.GetValue(0);
    rows[c] = io.words.GetValue(1);
    for (int r = 0; r < ROUTES; ++r)
      maps[c][r] = io.words.GetValue(8 + r);
  }
  for (int pass = 0; pass < 2; ++pass) {
    int64_t polls = 0;
    int line = pass ? DOWN_ALL_READY : DOWN_RANGE_READY;
    while (Load(ctrl + (line + slot) * LINE) != generation) {
      if (Load(ctrl + STOP * LINE) || ++polls >= cfg[9]) {
        Store(ctrl + STOP * LINE, -25);
        return;
      }
    }
    bool marked = false;
    for (int c = 0; c < 2; ++c) {
      if (!sources[c])
        continue;
      for (int route = worker; route < rows[c] * TOPK; route += VW) {
        int row = maps[c][route];
        if (row < 0 || (row >= boundary) != bool(pass))
          continue;
        if (cfg[13] && !marked) {
          auto times =
              (__gm__ int64_t *)cfg[13] + ((command - 1) * 24 + worker) * 8;
          times[3 + pass] = GetSystemCycle();
          marked = true;
        }
        io.Copy((__gm__ int32_t *)ptr[4] + row * HIDDEN / 2,
                (__gm__ int32_t *)cfg[2 + c] + 64 + route * HIDDEN / 2,
                HIDDEN / 2);
        if (cfg[23]) {
          // Copy has retired MTE3. The flag belongs to this route's sole mover;
          // never publish a generation before the exported payload is visible.
          Store((__gm__ int32_t *)cfg[2 + c] + 64 + ROUTES * HIDDEN / 2 +
                    route * LINE,
                sources[c]);
        }
      }
    }
  }
}

// A separate urgent mailbox lets a mover suspend its local copy loop without
// retiring/reissuing that command or rereading its already-local route map.
// Poll only after completed DMA; Activation may overwrite UB, not local maps.
__aicore__ inline void Urgent(__gm__ int64_t *cfg, Transfer &io, int worker,
                              int &seen) {
  if (!cfg[17])
    return;
  auto ctrl = (__gm__ int32_t *)cfg[0];
  int gen = Load(ctrl + URGENT_CMD * LINE);
  if (gen == seen)
    return;
  int slot = ctrl[URGENT_CMD * LINE + 2];
  if (gen != seen + 1 || slot < 0 || slot > 1 ||
      ctrl[URGENT_CMD * LINE + 1] != ACTIVATE) {
    Store(ctrl + STOP * LINE, -24);
    return;
  }
  int end = ctrl[URGENT_CMD * LINE + 3], beginRow = ctrl[URGENT_CMD * LINE + 4];
  auto ptr = (__gm__ int64_t *)cfg[1] + slot * 16;
  uint64_t begin = GetSystemCycle();
  for (int row = beginRow + worker; row < end; row += VW)
    io.Activation((__gm__ bfloat16_t *)ptr[2] + row * INNER * 2,
                  (__gm__ bfloat16_t *)ptr[3] + row * INNER);
  PipeBarrier<PIPE_ALL>();
  WorkTime(cfg, 2, gen, worker, begin);
  seen = gen;
  Store(ctrl + (URGENT_DONE + worker) * LINE, gen);
}

__aicore__ inline void Worker(__gm__ int64_t *cfg, Transfer &io) {
  auto ctrl = (__gm__ int32_t *)cfg[0];
  int seen = 0, idle = 0, worker = GetBlockIdx() - 1, urgentSeen = 0;
  while (!Load(ctrl + STOP * LINE)) {
    Urgent(cfg, io, worker, urgentSeen);
    int next = Load(ctrl + VCMD * LINE);
    if (next == seen) {
      if (!cfg[24] && ++idle >= cfg[9]) {
        Store(ctrl + STOP * LINE, -21);
        break;
      }
      continue;
    }
    idle = 0;
    int kind = ctrl[VCMD * LINE + 1], slot = ctrl[VCMD * LINE + 2];
    int extra = ctrl[VCMD * LINE + 3];
    if (next != seen + 1 || slot < 0 || slot > 1 || kind < 1 || kind > 4) {
      Store(ctrl + STOP * LINE, -22);
      break;
    }
    auto ptr = (__gm__ int64_t *)cfg[1] + slot * 16;
    uint64_t begin = GetSystemCycle();
    if (kind == SEND && cfg[22]) {
      ReturnStreaming(cfg, io, slot, worker, next);
    } else if (kind == ACTIVATE) {
      for (int row = ctrl[VCMD * LINE + 4] + worker; row < extra; row += VW)
        io.Activation((__gm__ bfloat16_t *)ptr[2] + row * INNER * 2,
                      (__gm__ bfloat16_t *)ptr[3] + row * INNER);
    } else {
      int sourceBase = 0;
      for (int c = 0; c < 2; ++c) {
        io.Read((__gm__ int32_t *)ptr[5] + c * MAP, MAP);
        int gen = io.words.GetValue(0), n = io.words.GetValue(1),
            layer = io.words.GetValue(2);
        int map[ROUTES];
        for (int i = 0; i < ROUTES; ++i)
          map[i] = io.words.GetValue(8 + i);
        int base = sourceBase;
        sourceBase += kind == FETCH ? n : n * TOPK;
        int first = 0, limit = kind == FETCH ? n : n * TOPK;
        if (cfg[16] && (kind == FETCH || kind == REPACK)) {
          first = ScalarMax(0, ctrl[VCMD * LINE + 4] - base);
          limit = ScalarMin(limit, ctrl[VCMD * LINE + 5] - base);
        }
        if (!gen)
          continue;
        if (kind == FETCH) {
          if (!(extra & (1 << c)))
            continue;
          for (int row = first + worker; row < limit; row += VW) {
            Urgent(cfg, io, worker, urgentSeen);
            io.Copy((__gm__ int32_t *)cfg[4 + c] + 1024 + row * HIDDEN / 2,
                    (__gm__ int32_t *)ptr[0] + (c * 32 + row) * HIDDEN / 2,
                    HIDDEN / 2);
          }
        } else {
          for (int route = first + worker; route < limit; route += VW)
            if (map[route] >= 0) {
              if (kind == REPACK)
                Urgent(cfg, io, worker, urgentSeen);
              if (kind == REPACK) {
                io.Copy((__gm__ int32_t *)ptr[0] +
                            (c * 32 + route / TOPK) * HIDDEN / 2,
                        (__gm__ int32_t *)ptr[1] + map[route] * HIDDEN / 2,
                        HIDDEN / 2);
                if (cfg[19]) {
                  // Copy already waits on MTE3_S. This is an observation after
                  // existing output completion, not a new payload fence.
                  // One cache line per destination row: movers never share it.
                  auto mark = (__gm__ int64_t *)cfg[19] +
                              ((next - 1) * CAPACITY + map[route]) * 8;
                  mark[0] = GetSystemCycle();
                  mark[1] = c;
                  mark[2] = gen;
                  mark[3] = route;
                  mark[4] = layer;
                  mark[5] = worker;
                  Refresh((__gm__ int32_t *)mark);
                }
                if (cfg[20])
                  Store((__gm__ int32_t *)cfg[20] +
                            (slot * CAPACITY + map[route]) * LINE,
                        next);
              } else
                io.Copy((__gm__ int32_t *)ptr[4] + map[route] * HIDDEN / 2,
                        (__gm__ int32_t *)cfg[2 + c] + 64 + route * HIDDEN / 2,
                        HIDDEN / 2);
            }
        }
      }
    }
    PipeBarrier<PIPE_ALL>();
    WorkTime(cfg, 0, next, worker, begin);
    seen = next;
    Store(ctrl + (VDONE + worker) * LINE, seen);
  }
}

struct Slot {
  int stage = EMPTY, gen[2] = {0, 0}, rows[2] = {0, 0}, layer[2] = {0, 0};
  int ids[2][ROUTES], live = 0, boundary = 0;
  int upDone = 0, actDone = 0, downDone = 0, downGen = 0, downPrefix = 0;
  int moveCursor = 0, moveEnd = 0, fetchMask = 0;
  int priority = 0, serviceRank = 0, ticket = 0;
};
__aicore__ inline void Descriptor(Transfer &io, __gm__ int64_t *ptr, Slot &s) {
  for (int c = 0; c < 2; ++c) {
    for (int j = 0; j < MAP; ++j)
      io.words.SetValue(
          j, j == 0 ? s.gen[c]
                    : (j == 1 ? s.rows[c] : (j == 2 ? s.layer[c] : -1)));
    io.Write((__gm__ int32_t *)ptr[5] + c * MAP, MAP);
  }
}
__aicore__ inline int Accept(Transfer &io, __gm__ int64_t *cfg, Slot &s,
                             int *claimed, int *finished, int *closed,
                             PriorityPolicy &policy) {
  int mask = 0;
  bool pending[2] = {false, false};
  int generations[2] = {0, 0}, layers[2] = {0, 0}, rows[2] = {0, 0};
  int priorities[2] = {0, 0}, urgency[2] = {0, 0};
  bool occupied = s.gen[0] || s.gen[1];
  for (int c = 0; c < 2; ++c) {
    if (claimed[c] || closed[c] || (!cfg[24] && finished[c] >= cfg[6]))
      continue;
    auto src = (__gm__ int32_t *)cfg[4 + c];
    int gen = io.Flag(src);
    if (cfg[24] && gen < 0) {
      if (gen != -(finished[c] + 1))
        return -1;
      closed[c] = 1;
      continue;
    }
    if (gen != finished[c] + 1)
      continue;
    io.Read(src + 8, 8);
    int desc = io.words.GetValue(0), layer = io.words.GetValue(1),
        n = io.words.GetValue(2), priority = io.words.GetValue(3);
    if (desc != gen || layer < 0 || layer >= LAYERS ||
        (SINGLE_LAYER && layer >= cfg[26]) || n < 1 || n > 32 || priority < 0 ||
        priority > 1)
      return -1;
    if (occupied && (priority != s.priority ||
                     (SINGLE_LAYER && ((s.gen[0] && s.layer[0] != layer) ||
                                       (s.gen[1] && s.layer[1] != layer)))))
      continue;
    pending[c] = true;
    generations[c] = gen;
    layers[c] = layer;
    rows[c] = n;
    priorities[c] = priority;
    urgency[c] =
        priority == 1 && Promoted(gen, io.Flag(src + 16)) ? 0 : priority;
  }
  int first = policy.Choose(pending, urgency);
  if (first < 0)
    return 0;
  if (!occupied) {
    s.priority = priorities[first];
    s.serviceRank = policy.Grant(first, s.priority) ? -1 : urgency[first];
    s.ticket = policy.nextTicket++;
  }
  for (int j = 0; j < 2; ++j) {
    int c = (first + j) % 2;
    if (!pending[c] || priorities[c] != s.priority ||
        (SINGLE_LAYER && layers[c] != layers[first]))
      continue;
    auto src = (__gm__ int32_t *)cfg[4 + c];
    int gen = generations[c], n = rows[c], layer = layers[c];
    io.Read(src + 64, (n * TOPK + 7) / 8 * 8);
    for (int i = 0; i < n * TOPK; ++i) {
      s.ids[c][i] = io.words.GetValue(i);
      if (s.ids[c][i] < 0 || s.ids[c][i] >= EXPERTS)
        return -1;
    }
    claimed[c] = gen;
    s.gen[c] = gen;
    s.rows[c] = n;
    s.layer[c] = layer;
    mask |= 1 << c;
  }
  return mask;
}
__aicore__ inline void Group(Transfer &io, __gm__ int64_t *ptr, Slot &s,
                             int owner, int mode, int tailExperts) {
  bool segmented = mode != 0;
  int count[GROUPS], cursor[GROUPS];
  for (int g = 0; g < GROUPS; ++g)
    count[g] = 0;
  for (int c = 0; c < 2; ++c)
    if (s.gen[c])
      for (int i = 0; i < s.rows[c] * TOPK; ++i)
        if (s.ids[c][i] / LOCAL_EXPERTS == owner)
          ++count[(SINGLE_LAYER ? 0 : s.layer[c] * LOCAL_EXPERTS) +
                  s.ids[c][i] % LOCAL_EXPERTS];
  s.live = 0;
  for (int g = 0; g < GROUPS; ++g) {
    cursor[g] = s.live;
    s.live += count[g];
    io.words.SetValue(g * 2, s.live);
    io.words.SetValue(g * 2 + 1, 0);
  }
  io.Write((__gm__ int32_t *)ptr[6], GROUPS * 2);
  s.boundary = s.live;
  if (segmented) {
    // Split only at an expert boundary. A single hot expert stays intact;
    // empty segments remain legal no-ops. Catalogs are frozen before PACK.
    int cut = 0, sum = 0;
    while (cut < GROUPS && sum < (s.live + 1) / 2)
      sum += count[cut++];
    if (tailExperts) {
      cut = GROUPS;
      int remaining = tailExperts;
      while (cut > 0 && remaining > 0)
        if (count[--cut])
          --remaining;
      sum = 0;
      for (int g = 0; g < cut; ++g)
        sum += count[g];
    }
    s.boundary = sum;
    if (mode == 2) {
      // Continuous GEMMs consume only the full catalog and this boundary.
      // Preserve the existing aligned last-line address read by every Cube.
      for (int j = 0; j < 8; ++j)
        io.words.SetValue(j, j == 6 ? sum : 0);
      io.Write((__gm__ int32_t *)ptr[9] + 248, 8);
    } else
      for (int part = 0; part < 2; ++part) {
        int prefix = 0;
        for (int g = 0; g < GROUPS; ++g) {
          if ((part == 0 && g < cut) || (part == 1 && g >= cut))
            prefix += count[g];
          io.words.SetValue(g * 2, prefix);
          io.words.SetValue(g * 2 + 1, 0);
        }
        io.Write((__gm__ int32_t *)ptr[9 + part], GROUPS * 2);
      }
  }
  for (int c = 0; c < 2; ++c) {
    for (int i = 0; i < MAP; ++i)
      io.words.SetValue(i, -1);
    io.words.SetValue(0, s.gen[c]);
    io.words.SetValue(1, s.rows[c]);
    io.words.SetValue(2, s.layer[c]);
    if (s.gen[c])
      for (int i = 0; i < s.rows[c] * TOPK; ++i)
        if (s.ids[c][i] / LOCAL_EXPERTS == owner)
          io.words.SetValue(
              8 + i, cursor[(SINGLE_LAYER ? 0 : s.layer[c] * LOCAL_EXPERTS) +
                            s.ids[c][i] % LOCAL_EXPERTS]++);
    io.Write((__gm__ int32_t *)ptr[5] + c * MAP, MAP);
  }
}
__aicore__ inline void Command(__gm__ int32_t *ctrl, int line, int gen,
                               int kind, int slot, int extra = 0,
                               int firstRow = 0, int lastRow = 0) {
  ctrl[line * LINE + 1] = kind;
  ctrl[line * LINE + 2] = slot;
  ctrl[line * LINE + 3] = extra;
  ctrl[line * LINE + 4] = firstRow;
  ctrl[line * LINE + 5] = lastRow;
  Store(ctrl + line * LINE, gen);
}
__aicore__ inline void StartFetch(Slot &slot, int mask) {
  slot.fetchMask = mask;
  slot.moveCursor = mask & 1 ? 0 : slot.rows[0];
  slot.moveEnd = mask & 2 ? slot.rows[0] + slot.rows[1] : slot.rows[0];
}
__aicore__ inline void MoveCommand(__gm__ int32_t *ctrl, int gen, int id,
                                   Slot &slot, int quantum) {
  bool fetch = slot.stage == PULL;
  int limit = ScalarMin(slot.moveEnd,
                        slot.moveCursor + (fetch ? quantum / 8 : quantum));
  Command(ctrl, VCMD, gen, fetch ? FETCH : REPACK, id,
          fetch ? slot.fetchMask : 0, slot.moveCursor, limit);
}
// Coordinator-observed command intervals, not exact instruction timestamps.
__aicore__ inline void Record(__gm__ int64_t *cfg, int &count, int engine,
                              int kind, int slot, uint64_t begin, int rows,
                              int part = 0) {
  auto record = (__gm__ int64_t *)cfg[11] + (count % 512) * 8;
  record[0] = engine;
  record[1] = kind;
  record[2] = slot;
  record[3] = begin;
  record[4] = GetSystemCycle();
  record[5] = rows;
  record[6] = count;
  record[7] = part;
  Refresh((__gm__ int32_t *)record);
  ++count;
}
__aicore__ inline void Coordinator(__gm__ int64_t *cfg, Transfer &io) {
  auto ctrl = (__gm__ int32_t *)cfg[0];
  auto slots = (__gm__ int64_t *)cfg[1];
  Slot s[2];
  PriorityPolicy policy;
  int promotions = 0;
  int parts = cfg[14] ? 2 : 1, vpart = 0, cpart = 0;
  int claimed[2] = {0, 0}, finished[2] = {0, 0}, closed[2] = {0, 0};
  int vgen = 0, cgen = 0, vs = -1, cs = -1, waves = 0, idle = 0;
  int pullsDuringCube = 0, eventCount = 0, vkind = 0, ckind = 0;
  uint64_t vbegin = 0, cbegin = 0;
  int us = -1, ugen = 0, upart = 0;
  uint64_t ubegin = 0;
  bool streaming = cfg[14] == 2;
  while (!Load(ctrl + STOP * LINE)) {
    bool progress = false;
    if (us >= 0 && Joined(ctrl, URGENT_DONE, VW, ugen)) {
      Record(cfg, eventCount, 2, ACTIVATE, us, ubegin, s[us].live, upart);
      ++s[us].actDone;
      if (cfg[18] && s[us].actDone == parts && s[us].downGen)
        Store(ctrl + (ACT_TAIL_READY + us) * LINE, s[us].downGen);
      us = -1;
      progress = true;
    }
    if (streaming && cs >= 0 && ckind == 1 && s[cs].upDone == 0 &&
        Joined(ctrl, UP_PREFIX_DONE, CW, cgen)) {
      s[cs].upDone = 1;
      progress = true;
    }
    if (cfg[22] && cs >= 0 && ckind == 2 && !s[cs].downPrefix &&
        Joined(ctrl, DOWN_PREFIX_DONE, CW, cgen)) {
      s[cs].downPrefix = 1;
      Store(ctrl + (DOWN_RANGE_READY + cs) * LINE, cgen);
      progress = true;
    }
    if (cs >= 0 && Joined(ctrl, CDONE, CW, cgen)) {
      Record(cfg, eventCount, 1, ckind, cs, cbegin, s[cs].live, cpart);
      if (ckind == 1)
        s[cs].upDone = streaming ? parts : s[cs].upDone + 1;
      else {
        s[cs].downDone = streaming ? parts : s[cs].downDone + 1;
        if (cfg[22]) {
          // A prefix poll can miss a core that finishes before the following
          // full-completion poll. Full completion implies prefix completion;
          // publish both before clearing cs, otherwise SEND waits forever for
          // a prefix generation that the coordinator will never revisit.
          s[cs].downPrefix = 1;
          Store(ctrl + (DOWN_RANGE_READY + cs) * LINE, cgen);
          Store(ctrl + (DOWN_ALL_READY + cs) * LINE, cgen);
        }
      }
      cs = -1;
      progress = true;
    }
    if (vs >= 0 && Joined(ctrl, VDONE, VW, vgen)) {
      auto &slot = s[vs];
      auto ptr = slots + vs * 16;
      Record(cfg, eventCount, 0, vkind, vs, vbegin, slot.live, vpart);
      if ((cfg[16] || cfg[17]) && (slot.stage == PULL || slot.stage == PACK)) {
        bool fetch = slot.stage == PULL;
        slot.moveCursor =
            cfg[17] ? slot.moveEnd
                    : ScalarMin(slot.moveEnd,
                                slot.moveCursor +
                                    int(fetch ? cfg[16] / 8 : cfg[16]));
        if (slot.moveCursor == slot.moveEnd) {
          if (fetch) {
            int added =
                Accept(io, cfg, slot, claimed, finished, closed, policy);
            if (added < 0) {
              Store(ctrl + STOP * LINE, -31);
              break;
            }
            if (added) {
              Descriptor(io, ptr, slot);
              StartFetch(slot, added);
            } else {
              Group(io, ptr, slot, cfg[7], cfg[14], cfg[15]);
              slot.stage = PACK;
              slot.moveCursor = 0;
              slot.moveEnd = (slot.rows[0] + slot.rows[1]) * TOPK;
            }
          } else {
            slot.stage = READY_UP;
          }
        }
        // Rejoin the ready selection instead of chaining the next DMA chunk.
        vs = -1;
      } else if (slot.stage == PULL) {
        // No wait-to-coalesce: one snapshot after useful DMA completes.
        int added = Accept(io, cfg, slot, claimed, finished, closed, policy);
        if (added < 0) {
          Store(ctrl + STOP * LINE, -31);
          break;
        }
        if (added) {
          Descriptor(io, ptr, slot);
          vkind = FETCH;
          vbegin = GetSystemCycle();
          Command(ctrl, VCMD, ++vgen, FETCH, vs, added);
          if (cs >= 0)
            ++pullsDuringCube;
        } else {
          Group(io, ptr, slot, cfg[7], cfg[14], cfg[15]);
          slot.stage = PACK;
          vkind = REPACK;
          vbegin = GetSystemCycle();
          if (cfg[20])
            Store(ctrl + (PACK_EPOCH + vs) * LINE, vgen + 1);
          Command(ctrl, VCMD, ++vgen, REPACK, vs);
        }
      } else {
        if (vkind == REPACK) {
          if (slot.stage == PACK)
            slot.stage = READY_UP;
        } else if (vkind == ACTIVATE) {
          ++slot.actDone;
          if (cfg[18] && slot.actDone == parts && slot.downGen)
            Store(ctrl + (ACT_TAIL_READY + vs) * LINE, slot.downGen);
        } else if (slot.stage == RETURN) {
          for (int c = 0; c < 2; ++c)
            if (slot.gen[c]) {
              io.Publish((__gm__ int32_t *)cfg[2 + c], slot.gen[c]);
              finished[c] = slot.gen[c];
              claimed[c] = 0;
            }
          int fields[16] = {slot.gen[0],
                            slot.gen[1],
                            slot.live,
                            vs,
                            pullsDuringCube,
                            vgen,
                            cgen,
                            slot.gen[0] ? slot.layer[0] : -1,
                            slot.gen[1] ? slot.layer[1] : -1,
                            slot.rows[0],
                            slot.rows[1],
                            slot.priority,
                            slot.ticket,
                            slot.serviceRank,
                            0,
                            0};
          for (int i = 0; i < 16; ++i)
            io.words.SetValue(i, fields[i]);
          io.Write((__gm__ int32_t *)cfg[8] + (waves % (cfg[6] * 2)) * 16, 16);
          ++waves;
          slot.gen[0] = slot.gen[1] = slot.rows[0] = slot.rows[1] = 0;
          slot.stage = EMPTY;
          slot.upDone = slot.actDone = slot.downDone = slot.downGen = 0;
          slot.downPrefix = 0;
        }
        vs = -1;
      }
      progress = true;
    }
    if (cfg[24] ? (closed[0] && closed[1])
                : (finished[0] == cfg[6] && finished[1] == cfg[6])) {
      ctrl[STATUS * LINE + 1] = waves;
      ctrl[STATUS * LINE + 2] = pullsDuringCube;
      ctrl[STATUS * LINE + 3] = eventCount;
      ctrl[STATUS * LINE + 4] = finished[0];
      ctrl[STATUS * LINE + 5] = finished[1];
      ctrl[STATUS * LINE + 6] = promotions;
      Store(ctrl + STATUS * LINE, 1);
      Store(ctrl + STOP * LINE, 1);
      break;
    }
    // Shared completion promotes a still-owned generation, including work
    // already admitted into staging. Never let a previous generation promote
    // a new task. Promotions are monotone until this slot retires.
    for (int i = 0; i < 2; ++i)
      if (s[i].stage != EMPTY && s[i].serviceRank > 0)
        for (int c = 0; c < 2; ++c)
          if (s[i].gen[c] &&
              Promoted(s[i].gen[c],
                       io.Flag((__gm__ int32_t *)cfg[4 + c] + 16))) {
            s[i].serviceRank = 0;
            ++promotions;
            break;
          }
    int oldest = s[0].ticket <= s[1].ticket ? 0 : 1;
    for (int k = 0; k < 2; ++k) {
      int i = (oldest + k) % 2;
      if (s[i].stage != EMPTY &&
          policy.ProtectWaitingPrefill(s[i].priority, s[i].serviceRank))
        break;
    }
    int order[2] = {0, 1};
    if (PriorityBefore(s[1].serviceRank, s[1].ticket, s[0].serviceRank,
                       s[0].ticket)) {
      order[0] = 1;
      order[1] = 0;
    }
    if (cs < 0) {
      // Down may consume only a joined activation segment. Otherwise continue
      // up while AIV consumes its earlier, disjoint segment. UP denotes the
      // entire compute lifetime; engine progress is tracked independently.
      for (int k = 0; k < 2 && cs < 0; ++k)
        for (int phase = 0; phase < 2 && cs < 0; ++phase) {
          int i = order[k];
          auto &slot = s[i];
          if (slot.stage != READY_UP && slot.stage != UP &&
              !(cfg[20] && slot.stage == PACK))
            continue;
          bool ready =
              phase == 0
                  ? (streaming ? slot.actDone >= (cfg[18] ? 1 : parts) &&
                                     slot.upDone == parts && slot.downDone == 0
                               : slot.downDone < slot.actDone)
                  : slot.upDone < parts;
          if (ready) {
            cs = i;
            slot.stage = UP;
            ckind = phase == 0 ? 2 : 1;
            cpart = phase == 0 ? slot.downDone : slot.upDone;
            cbegin = GetSystemCycle();
            ++cgen;
            if (cfg[18] && ckind == 2) {
              // A globally unique command generation prevents old slot flags
              // satisfying a new invocation. Publish only after the AIV join.
              slot.downGen = cgen;
              if (slot.actDone == parts)
                Store(ctrl + (ACT_TAIL_READY + i) * LINE, cgen);
            }
            Command(ctrl, CCMD, cgen, ckind, i, cpart);
            progress = true;
          }
        }
    }
    if (cfg[17] && us < 0 && vs >= 0 && (vkind == FETCH || vkind == REPACK)) {
      for (int i = 0; i < 2 && us < 0; ++i) {
        if (i == vs || s[i].stage != UP || s[i].actDone >= s[i].upDone)
          continue;
        us = i;
        upart = s[i].actDone;
        ubegin = GetSystemCycle();
        int start = upart ? s[i].boundary : 0;
        int end = upart ? s[i].live : s[i].boundary;
        Command(ctrl, URGENT_CMD, ++ugen, ACTIVATE, i, end, start);
        progress = true;
      }
    }
    if (vs < 0 && us < 0) {
      for (int k = 0; k < 2 && vs < 0; ++k)
        for (int phase = 0; phase < 2 && vs < 0; ++phase) {
          int i = order[k];
          auto &slot = s[i];
          if (slot.stage != UP)
            continue;
          bool ready =
              phase == 0
                  ? (slot.downDone == parts ||
                     (cfg[22] && slot.downPrefix && slot.actDone == parts))
                  : slot.actDone < slot.upDone;
          if (ready) {
            vs = i;
            vkind = phase == 0 ? SEND : ACTIVATE;
            vpart = phase == 0 ? 0 : slot.actDone;
            if (phase == 0)
              slot.stage = RETURN;
            int start = vpart ? slot.boundary : 0;
            int end = parts == 2 && !vpart ? slot.boundary : slot.live;
            vbegin = GetSystemCycle();
            if (phase == 0 && cfg[22]) {
              end = slot.boundary;
              start = slot.downGen;
            }
            Command(ctrl, VCMD, ++vgen, vkind, i, end, start);
            progress = true;
          }
        }
      if (cfg[16] || cfg[17]) {
        for (int i = 0; i < 2 && vs < 0; ++i) {
          if (s[i].stage != PULL && s[i].stage != PACK)
            continue;
          vs = i;
          vkind = s[i].stage == PULL ? FETCH : REPACK;
          vpart = s[i].moveCursor;
          vbegin = GetSystemCycle();
          MoveCommand(ctrl, ++vgen, i, s[i], cfg[17] ? 512 : cfg[16]);
          if (cs >= 0 && vkind == FETCH)
            ++pullsDuringCube;
          progress = true;
        }
      }
      for (int i = 0; i < 2 && vs < 0; ++i)
        if (s[i].stage == EMPTY) {
          int mask = Accept(io, cfg, s[i], claimed, finished, closed, policy);
          if (mask < 0) {
            Store(ctrl + STOP * LINE, -32);
            break;
          }
          if (mask) {
            vs = i;
            s[i].stage = PULL;
            Descriptor(io, slots + i * 16, s[i]);
            vkind = FETCH;
            vpart = 0;
            vbegin = GetSystemCycle();
            if (cfg[16] || cfg[17]) {
              StartFetch(s[i], mask);
              MoveCommand(ctrl, ++vgen, i, s[i], cfg[17] ? 512 : cfg[16]);
            } else {
              Command(ctrl, VCMD, ++vgen, FETCH, i, mask);
            }
            if (cs >= 0)
              ++pullsDuringCube;
            progress = true;
          }
        }
    }
    idle = (progress || cfg[24]) ? 0 : idle + 1;
    if (!cfg[24] && idle >= cfg[9]) {
      Store(ctrl + STOP * LINE, -33);
      break;
    }
  }
}
extern "C" __global__ __aicore__ void
persistent_vector(GM_ADDR config, GM_ADDR unused, GM_ADDR unused2) {
  auto cfg = (__gm__ int64_t *)config;
  if (!cfg[10])
    return;
  Transfer io;
  io.Init();
  if (GetBlockIdx() == 0)
    Coordinator(cfg, io);
  else
    Worker(cfg, io);
}
static const struct FunLevelKType persistent_vector_meta
    __attribute__((used, section(".ascend.meta.persistent_vector"))) = {
        {F_TYPE_KTYPE, sizeof(unsigned int), K_TYPE_AIV}};
