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
  const bool quantized = cfg[5] < 48;
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
