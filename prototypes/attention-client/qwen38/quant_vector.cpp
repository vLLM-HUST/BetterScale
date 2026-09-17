#include "quant_vector.hpp"
// cfg: mode,width,capacity,input-scales,weight-scales,group-ends,groups,output-scales.
// mode0 BF16->INT8; mode1 INT32->FP32 SwiGLU->INT8; mode2 INT32->BF16.
extern "C" __global__ __aicore__ void quant_vector(GM_ADDR config, GM_ADDR input,
                                                   GM_ADDR output) {
  auto cfg = (__gm__ int64_t *)config;
  int mode = cfg[0], width = cfg[1], capacity = cfg[2], groups = cfg[6];
  auto ends = (__gm__ int64_t *)cfg[5];
  TPipe pipe;
  TBuf<TPosition::VECCALC> buffer;
  pipe.InitBuffer(buffer, 65536);
  QuantRow op(buffer.Get<int32_t>());
  for (int row = GetBlockIdx(); row < capacity; row += GetBlockNum()) {
    int expert = 0;
    if (mode) {
      while (expert < groups && row >= ends[expert])
        ++expert;
      if (expert == groups)
        continue;
      op.Dequant((__gm__ int32_t *)input + row * width,
                 (__gm__ float *)cfg[4] + expert * width,
                 (__gm__ float *)cfg[3] + row * 8, width);
    } else
      op.FromBf16((__gm__ bfloat16_t *)input + row * width, width);
    if (mode == 2)
      op.ToBf16((__gm__ bfloat16_t *)output + row * width, width);
    else {
      int outWidth = mode ? width / 2 : width;
      if (mode)
        op.Swiglu(outWidth);
      op.Quantize((__gm__ int8_t *)output + row * outWidth,
                  (__gm__ float *)cfg[7] + row * 8, outWidth, mode == 0);
    }
  }
}
static const struct FunLevelKType quant_vector_meta
    __attribute__((used, section(".ascend.meta.quant_vector"))) = {
        {F_TYPE_KTYPE, sizeof(unsigned int), K_TYPE_AIV}};
