#include "tiling.hpp"
#include "kernel_operator.h"
#include "attn_infra/detail/alignment.hpp"
using NpuArch::Detail::Alignment::CeilDiv;
#include "vendor/flash_attention_interface.cpp"
#include <acl/acl.h>
extern "C" __global__ __aicore__ void quota_head_lanes(
 GM_ADDR q,GM_ADDR k,GM_ADDR v,GM_ADDR mask,GM_ADDR table,GM_ADDR out,
 GM_ADDR qlen,GM_ADDR kvlen,GM_ADDR workspace,GM_ADDR tiling) {
 KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
 AscendC::SetSysWorkspace(workspace);
 auto* plan=reinterpret_cast<__gm__ FAInferTilingData*>(tiling);
 uint32_t group=AscendC::GetBlockIdx();
#ifdef __DAV_C220_VEC__
 group/=AscendC::GetSubBlockNum();
#endif
 // Metadata guarantees split nodes <= active producer groups. An inactive
 // group owns neither a producer nor a merge node; it only owes the barrier.
 if (group >= plan->needCoreNum) {
  if (plan->totalSplitNodeNum != 0) { AscendC::SyncAll(); }
  return;
 }

 SplitFuse::FAInfer<bfloat16_t,bfloat16_t,float,true,true,
 // Q1 bottom-right causal makes every valid KV position visible, including
 // every context shard. Tail masking still follows each actual KV length.
 KernelCommon::FaiKernel::MaskType::NO_MASK,KernelCommon::FaiKernel::inputLayout::TND>(
 q,k,v,nullptr,mask,table,out,nullptr,qlen,kvlen,workspace+(16ULL<<20),tiling,nullptr);
}
extern "C" void lane_launch(void* stream,uint64_t* p) {
 quota_head_lanes<<<24,nullptr,stream>>>((GM_ADDR)p[0],(GM_ADDR)p[1],(GM_ADDR)p[2],
 (GM_ADDR)p[3],(GM_ADDR)p[4],(GM_ADDR)p[5],(GM_ADDR)p[6],(GM_ADDR)p[7],(GM_ADDR)p[8],(GM_ADDR)p[9]);
}
