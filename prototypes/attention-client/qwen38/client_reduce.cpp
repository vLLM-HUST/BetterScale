// Token-owned pull and FP32 reduction, after the unchanged all-owner join.
// UB: [0,16KiB) wire read, [16,32KiB) converted input, [32,48KiB) sum.
extern "C" __global__ __aicore__ void
neural_collect_fused(GM_ADDR cfgaddr, GM_ADDR unused, GM_ADDR topk) {
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
  for (int token = GetBlockIdx(); token < n; token += GetBlockNum()) {
    int ids[TOPK];
    float weights[TOPK];
    int route = token * TOPK;
    // TOPK=10 and even row starts: 16 int32 words span this token.
    io.Read((__gm__ int32_t *)topk + route / 8 * 8, 16);
    for (int k = 0; k < TOPK; ++k) ids[k] = io.ub.GetValue(route % 8 + k);
    io.Read((__gm__ int32_t *)cfg[11] + route / 16 * 8, 16);
    PipeBarrier<PIPE_ALL>();
    Cast(fp[4096], bf, RoundMode::CAST_NONE, 32);
    PipeBarrier<PIPE_ALL>();
    for (int k = 0; k < TOPK; ++k) weights[k] = fp.GetValue(4096 + route % 16 + k);
    Duplicate(fp[8192], 0.0f, H);
    PipeBarrier<PIPE_V>();
    for (int k = 0; k < TOPK; ++k) {
      int owner = ids[k] / COLLECT_EXPERTS;
      io.Read((__gm__ int32_t *)cfg[1 + owner] + 64 + (route + k) * H / 2, H / 2);
      PipeBarrier<PIPE_ALL>();
      Cast(fp[4096], bf, RoundMode::CAST_NONE, H);
      PipeBarrier<PIPE_V>();
      Muls(fp[4096], fp[4096], weights[k], H);
      PipeBarrier<PIPE_V>();
      Add(fp[8192], fp[8192], fp[4096], H);
      PipeBarrier<PIPE_ALL>();
    }
    Cast(bf, fp[8192], RoundMode::CAST_RINT, H);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    io.Write((__gm__ int32_t *)cfg[12] + token * H / 2, H / 2);
  }
}
META(neural_collect_fused)
