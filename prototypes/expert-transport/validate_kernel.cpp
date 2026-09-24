#include "route_validate.hpp"
using namespace AscendC;
extern "C" __global__ __aicore__ void
route_validate_probe(GM_ADDR config, GM_ADDR unused, GM_ADDR unused2) {
  auto cfg=(__gm__ int64_t*)config;
  TPipe pipe;TBuf<TPosition::VECCALC> buf;pipe.InitBuffer(buf,196608);
  auto ids=buf.Get<int32_t>();auto fp=buf.Get<float>();
  GlobalTensor<int32_t> in,out;
  in.SetGlobalBuffer((__gm__ int32_t*)cfg[0]);out.SetGlobalBuffer((__gm__ int32_t*)cfg[2]);
  DataCopy(ids,in,int(cfg[1]));
  bool valid=PlannedRouteIdsValid(ids,fp,int(cfg[1]));
  for(int i=0;i<8;++i)ids.SetValue(i,i?0:int(valid));
  SetFlag<HardEvent::S_MTE3>(EVENT_ID0);WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
  DataCopy(out,ids,8);
  SetFlag<HardEvent::MTE3_S>(EVENT_ID0);WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
}
static const struct FunLevelKType route_validate_probe_meta
__attribute__((used,section(".ascend.meta.route_validate_probe")))={{F_TYPE_KTYPE,sizeof(unsigned int),K_TYPE_AIV}};
