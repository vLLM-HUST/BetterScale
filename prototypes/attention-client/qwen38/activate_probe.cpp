#include "quant_batch.hpp"
// cfg:
// rows,group_count,group_ends,channel_scales,input_scales,output_scales,mode.
extern "C" __global__ __aicore__ void
activate_probe(GM_ADDR config, GM_ADDR input, GM_ADDR output) {
  auto cfg = (__gm__ int64_t *)config;
  TPipe pipe;
  TBuf<TPosition::VECCALC> buf;
  pipe.InitBuffer(buf, 65536);
  auto scratch = buf.Get<int32_t>();
  int rows = cfg[0], groups = cfg[1], worker = GetBlockIdx();
  auto ends = (__gm__ int64_t *)cfg[2];
  auto channel = (__gm__ float *)cfg[3];
  auto scales = (__gm__ float *)cfg[4];
  auto outScale = (__gm__ float *)cfg[5];
  if (cfg[6]) {
    ActivateBatched(scratch, worker, GetBlockNum(), rows, ends, groups,
                    (__gm__ int32_t *)input, channel, scales,
                    (__gm__ int8_t *)output, outScale);
  } else {
    QuantRow q(scratch);
    for (int row = worker; row < rows; row += GetBlockNum()) {
      int expert = 0;
      while (expert < groups && row >= ends[expert])
        ++expert;
      q.Dequant((__gm__ int32_t *)input + row * 1280, channel + expert * 1280,
                scales + row * 8, 1280);
      q.Swiglu(640);
      q.Quantize((__gm__ int8_t *)output + row * 640, outScale + row * 8, 640);
    }
  }
}
static const struct FunLevelKType activate_probe_meta
    __attribute__((used, section(".ascend.meta.activate_probe"))) = {
        {F_TYPE_KTYPE, sizeof(unsigned int), K_TYPE_AIV}};
