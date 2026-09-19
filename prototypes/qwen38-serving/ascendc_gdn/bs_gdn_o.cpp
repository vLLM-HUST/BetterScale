#include "o_tiling.h"
#include "o/gemm/kernel/gdn_fwd_o_kernel.hpp"
extern "C" __global__ __aicore__ void bs_gdn_o(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR h, GM_ADDR g, GM_ADDR cu, GM_ADDR indices, GM_ADDR o, GM_ADDR workspace, GM_ADDR tiling) {
 KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
 Catlass::Gemm::Kernel::GDNFwdOKernel<bfloat16_t,float,float> kernel;
 kernel.Init(q,k,v,h,g,cu,indices,o,tiling,workspace);
 kernel.Process();
}
