// Two independently progressing tokens per AIV. Each ready contribution is
// pulled and accumulated immediately; a bitmask prevents double consumption.
// Arrival-order FP32 addition may differ from fixed top-k order.
extern "C" __global__ __aicore__ void
neural_collect_online(GM_ADDR cfgaddr, GM_ADDR unused, GM_ADDR topk) {
  auto cfg = (__gm__ int64_t *)cfgaddr;
  if (!cfg[8]) return;
  IO io;
  io.Init();
  auto bf = io.buf.Get<bfloat16_t>();
  auto fp = io.buf.Get<float>();
  int gen = io.Flag((__gm__ int32_t *)cfg[0]), n = cfg[6];
  uint64_t began = GetSystemCycle(), first = 0;
  int polls = 0;
  for (int base = GetBlockIdx(); base < n; base += GetBlockNum() * 2) {
    int tokens[2] = {base, base + int(GetBlockNum())};
    int ids[2][TOPK], masks[2] = {0,0};
    float weights[2][TOPK];
    int remaining = 0;
    for (int t = 0; t < 2; ++t) {
      if (tokens[t] >= n) continue;
      ++remaining;
      int route = tokens[t] * TOPK;
      io.Read((__gm__ int32_t *)topk + route / 8 * 8, 16);
      for (int k = 0; k < TOPK; ++k) ids[t][k] = io.ub.GetValue(route % 8 + k);
      io.Read((__gm__ int32_t *)cfg[11] + route / 16 * 8, 16);
      PipeBarrier<PIPE_ALL>();
      Cast(fp[4096], bf, RoundMode::CAST_NONE, 32);
      PipeBarrier<PIPE_ALL>();
      for (int k = 0; k < TOPK; ++k) weights[t][k] = fp.GetValue(4096 + route % 16 + k);
      Duplicate(fp[8192 + t * H], 0.0f, H);
      PipeBarrier<PIPE_ALL>();
    }
    int idle = 0;
    while (remaining) {
      bool progress = false;
      for (int t = 0; t < 2; ++t) {
        if (tokens[t] >= n || masks[t] == (1 << TOPK) - 1) continue;
        int route = tokens[t] * TOPK;
        for (int owner = 0; owner < COLLECT_OWNERS; ++owner) {
          bool needed = false;
          for (int k = 0; k < TOPK; ++k)
            needed |= !(masks[t] & (1 << k)) && ids[t][k] / COLLECT_EXPERTS == owner;
          if (!needed) continue;
          ++polls;
          io.Read((__gm__ int32_t *)cfg[1 + owner] + 64 + ROUTES * H / 2 + route * 16, TOPK * 16);
          int ready = 0;
          for (int k = 0; k < TOPK; ++k)
            if (io.ub.GetValue(k * 16) == gen) ready |= 1 << k;
          for (int k = 0; k < TOPK; ++k) {
            int bit = 1 << k;
            if ((masks[t] & bit) || !(ready & bit) || ids[t][k] / COLLECT_EXPERTS != owner) continue;
            if (!first) first = GetSystemCycle();
            io.Read((__gm__ int32_t *)cfg[1 + owner] + 64 + (route+k)*H/2, H/2);
            PipeBarrier<PIPE_ALL>();
            Cast(fp[4096], bf, RoundMode::CAST_NONE, H);
            PipeBarrier<PIPE_V>();
            Muls(fp[4096], fp[4096], weights[t][k], H);
            PipeBarrier<PIPE_V>();
            Add(fp[8192+t*H], fp[8192+t*H], fp[4096], H);
            PipeBarrier<PIPE_ALL>();
            masks[t] |= bit;
            progress = true;
          }
        }
        if (masks[t] == (1 << TOPK) - 1) {
          Cast(bf, fp[8192+t*H], RoundMode::CAST_RINT, H);
          SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
          WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
          io.Write((__gm__ int32_t *)cfg[12] + tokens[t]*H/2, H/2);
          --remaining;
        }
      }
      idle = progress ? 0 : idle + 1;
      if (idle >= cfg[9]) { io.Publish((__gm__ int32_t *)cfg[7], -101); return; }
      if (!progress && cfg[14]) {
        uint64_t until = GetSystemCycle() + cfg[14];
        while (GetSystemCycle() < until) {}
      }
    }
  }
  uint64_t reduced = GetSystemCycle();
  // Early consumption is not permission to reuse source or output storage.
  for (int owner = 0; owner < COLLECT_OWNERS; ++owner) {
    int idle = 0;
    while (io.Flag((__gm__ int32_t *)cfg[1 + owner]) != gen)
      if (++idle >= cfg[9]) { io.Publish((__gm__ int32_t *)cfg[7], -102); return; }
  }
  uint64_t drained = GetSystemCycle();
  if (cfg[13]) {
    auto times = io.buf.Get<int64_t>();
    times.SetValue(0,began); times.SetValue(1,first);
    times.SetValue(2,reduced); times.SetValue(3,drained);
    times.SetValue(4,polls); times.SetValue(5,gen);
    times.SetValue(6,0); times.SetValue(7,0);
    io.Write((__gm__ int32_t *)cfg[13] + GetBlockIdx()*16,16);
  }
}
META(neural_collect_online)
