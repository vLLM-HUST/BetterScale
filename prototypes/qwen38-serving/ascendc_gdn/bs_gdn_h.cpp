#include "h_tiling.h"
#include "h/gemm/kernel/gdn_fwd_h_kernel.hpp"
extern "C" __global__ __aicore__ void bs_gdn_h(GM_ADDR k, GM_ADDR w, GM_ADDR u, GM_ADDR g, GM_ADDR initial, GM_ADDR cu, GM_ADDR indices, GM_ADDR h, GM_ADDR vnew, GM_ADDR final, GM_ADDR workspace, GM_ADDR tiling) {
 KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
 Catlass::Gemm::Kernel::GDNFwdHKernel<bfloat16_t,float,float,float> kernel;
 kernel.Init(k,w,u,g,initial,cu,indices,h,vnew,final,tiling,workspace);
 kernel.Process();
}
