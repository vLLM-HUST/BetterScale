// Same wire/join and fixed FP32 route order as neural_collect_fused. Only the
// consumer pipeline changes: two input slots let MTE2 fetch k+1 while V reduces k.
// UB bytes: IO metadata [0,512); BF16 slots [512,10752);
// FP32 cast/product [11264,21504), sum [21504,31744); BF16 output [32768,37888).
// IO uses event0; input slots use events1/2. Output has its own lifetime.
extern "C" __global__ __aicore__ void
neural_collect_pipelined(GM_ADDR cfgaddr, GM_ADDR unused, GM_ADDR topk) {
  static_assert(H == 2560 && TOPK == 10, "Replan the UB map for other shapes");
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
