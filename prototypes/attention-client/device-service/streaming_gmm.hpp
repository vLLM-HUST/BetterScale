#pragma once
// Uses the installed CATLASS tile engine, layouts and swizzler unchanged.
// This local group traversal supplies the progress publication hook absent from
// GroupedMatmulSliceM. Keep math/tile policy shared with actual_gmm.cpp.
__aicore__ inline void RunStreamingUp(GM_ADDR config, GM_ADDR input,
                                      GM_ADDR output, int boundary,
                                      __gm__ int32_t *progress, int generation,
                                      __gm__ int64_t *timing) {
  auto cfg = (__gm__ int64_t *)config;
  const uint32_t k = cfg[0], n = cfg[1], groups = cfg[2];
  using Tile = ActualGmmTypes::Mmad;
  using Schedule = Gemm::Block::GemmIdentityBlockSwizzle<9, 1>;
  Arch::Resource<typename Tile::ArchTag> resource;
  Tile tile(resource);
  Schedule schedule;
  GlobalTensor<bfloat16_t> x, y;
  x.SetGlobalBuffer((__gm__ bfloat16_t *)input);
  y.SetGlobalBuffer((__gm__ bfloat16_t *)output);
  GlobalTensor<int64_t> ends;
  ends.SetGlobalBuffer((__gm__ int64_t *)cfg[4]);
  const auto weightLayout = layout::zN::MakeLayout<bfloat16_t>(k, n);
  uint32_t row = 0, nextCore = 0;
  bool published = false;
  for (uint32_t expert = 0; expert < groups; ++expert) {
    if (!published && row == boundary) {
      // Drain outstanding tile work, including FIX writes, before publishing.
      // The tile object and its L1/L0 buffers remain alive across this
      // boundary.
      tile.SynchronizeBlock();
      PipeBarrier<PIPE_ALL>();
      if (timing) {
        timing[2] = GetSystemCycle();
        Persistent::Refresh((__gm__ int32_t *)timing);
      }
      Persistent::Store(progress, generation);
      published = true;
    }
    uint32_t end = ends.GetValue(expert), rows = end - row;
    if (rows) {
      GemmCoord shape{rows, n, k};
      schedule.Update(shape, MatrixCoord{uint32_t(ACTUAL_GMM_TILE_M),
                                         uint32_t(ACTUAL_GMM_TILE_N)});
      const uint32_t tiles = schedule.GetCoreLoops();
      GlobalTensor<bfloat16_t> weight;
      weight.SetGlobalBuffer((__gm__ bfloat16_t *)cfg[3] +
                             uint64_t(expert) * k * n);
      if (rows <= ACTUAL_GMM_TILE_M)
        weight.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
      const layout::RowMajor lx{rows, k}, ly{rows, n};
      for (uint32_t t =
               (GetBlockIdx() + GetBlockNum() - nextCore) % GetBlockNum();
           t < tiles; t += GetBlockNum()) {
        auto coord = schedule.GetBlockCoord(t);
        uint64_t m = coord.m() * ACTUAL_GMM_TILE_M;
        uint64_t col = coord.n() * ACTUAL_GMM_TILE_N;
        tile(x[uint64_t(row) * k + m * k], lx,
             weight[weightLayout.GetOffset(
                 MatrixCoord{uint32_t(0), uint32_t(col)})],
             weightLayout, y[uint64_t(row) * n + m * n + col], ly,
             schedule.GetActualBlockShape(coord));
      }
      nextCore = (nextCore + tiles) % GetBlockNum();
    }
    row = end;
  }
  tile.SynchronizeBlock();
  PipeBarrier<PIPE_ALL>();
  if (!published) {
    if (timing) {
      timing[2] = GetSystemCycle();
      Persistent::Refresh((__gm__ int32_t *)timing);
    }
    Persistent::Store(progress, generation);
  }
}
