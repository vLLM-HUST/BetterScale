#pragma once
// Uses the installed CATLASS tile engine, layouts and swizzler unchanged.
// Up publishes a prefix; down waits at its first tail input issue. A non-null
// stop selects down mode. timing[2] is up publication; [3:5] is down tail wait.
// This local group traversal supplies the progress publication hook absent from
// GroupedMatmulSliceM. Keep math/tile policy shared with actual_gmm.cpp.
__aicore__ inline void
RunStreamingGmm(GM_ADDR config, GM_ADDR input, GM_ADDR output, int boundary,
                __gm__ int32_t *progress, int generation,
                __gm__ int64_t *timing, __gm__ int32_t *stop = nullptr,
                int64_t pollLimit = 0, Persistent::PackGate *pack = nullptr,
                __gm__ int32_t *downPrefix = nullptr,
                uint64_t weightAddress = 0) {
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
    if (stop && !published && row >= boundary) {
      if (downPrefix) {
        // One bounded prefix drain, not a drain per expert. Retain the same
        // tile object/buffers for the tail. FIX output must precede readiness.
        tile.SynchronizeBlock();
        PipeBarrier<PIPE_ALL>();
        if (timing)
          timing[5] = GetSystemCycle();
        Persistent::Store(downPrefix, generation);
      }
      // Check before issuing any tail GM->L1 copy. Keep the same tile object:
      // prefix MMAD/FIX may continue while its scalar issuer waits.
      if (timing)
        timing[3] = GetSystemCycle();
      int64_t polls = 0;
      while (Persistent::Load(progress) != generation) {
        if (Persistent::Load(stop) || ++polls >= pollLimit) {
          Persistent::Store(stop, -14);
          return;
        }
      }
      if (timing)
        timing[4] = GetSystemCycle();
      published = true;
    }
    if (!stop && !published && row == boundary) {
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
      weight.SetGlobalBuffer(
          (__gm__ bfloat16_t *)(weightAddress ? weightAddress : cfg[3]) +
          uint64_t(expert) * k * n);
      if (rows <= ACTUAL_GMM_TILE_M)
        weight.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
      const layout::RowMajor lx{rows, k}, ly{rows, n};
      uint32_t firstTile =
          (GetBlockIdx() + GetBlockNum() - nextCore) % GetBlockNum();
      if (firstTile < tiles && pack && pack->ready) {
        // All rows of this expert are required, including contributions from
        // both sources. Cores with no tile for it have no input dependency.
        // This guard precedes tile() and hence any pending GM->L1 prefetch.
        int64_t polls = 0;
        for (uint32_t r = row; r < end; ++r)
          while (Persistent::Load(pack->ready + r * Persistent::LINE) !=
                 pack->generation) {
            if (Persistent::Load(pack->stop) || ++polls >= pack->pollLimit) {
              Persistent::Store(pack->stop, -15);
              return;
            }
          }
      }
      for (uint32_t t = firstTile; t < tiles; t += GetBlockNum()) {
        if (t == firstTile && pack && pack->issueTrace) {
          auto mark = pack->issueTrace + expert * 2;
          mark[0] = GetSystemCycle();
          mark[1] = row;
          // Four expert records per line; one AIC owns the entire region.
          Persistent::Refresh(
              (__gm__ int32_t *)(pack->issueTrace + (expert / 4) * 8));
        }
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
  if (stop && !published && downPrefix) {
    if (timing)
      timing[5] = GetSystemCycle();
    Persistent::Store(downPrefix, generation);
  }
  if (!stop && !published) {
    if (timing) {
      timing[2] = GetSystemCycle();
      Persistent::Refresh((__gm__ int32_t *)timing);
    }
    Persistent::Store(progress, generation);
  }
}
