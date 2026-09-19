// Research launcher for the installed CANN9.0.1 MatMulV3 base kernel.
// CANN headers retain their CANN Open Software License Agreement v2.0.
// Only the externally supplied tiling changes; ND BF16 operands stay unchanged.
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "mat_mul_base_kernel.h"
#include <acl/acl.h>

using namespace AscendC;
using namespace matmul;
static_assert(sizeof(MatmulTilingData) == 280, "native tiling ABI changed");
static_assert(sizeof(TCubeTiling) == 200, "cube prefix ABI changed");
extern "C" __global__ __aicore__ void bs_gate_up_panel(
    GM_ADDR x, GM_ADDR w, GM_ADDR y, GM_ADDR tiling) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    MatmulTilingData data;
    auto dst = reinterpret_cast<uint32_t*>(&data);
    auto src = reinterpret_cast<__gm__ uint32_t*>(tiling);
    for (uint32_t i = 0; i < sizeof(data) / sizeof(uint32_t); ++i) dst[i] = src[i];
    using A = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t, false>;
    using B = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t, true>;
    using C = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t>;
    TPipe pipe;
    MatmulBaseKernel<A, B, C, C, MatmulBaseBlock, MM_CFG_NO_PRELOAD> op;
    op.Init(x, w, y, nullptr, nullptr, nullptr, &data, &pipe);
    op.Process();
}
extern "C" int launch_panel(void* x, void* w, void* y, void* tiling, void* stream) {
    if (!x || !w || !y || !tiling || !stream) return 1;
    bs_gate_up_panel<<<24, nullptr, stream>>>(
        (uint8_t*)x, (uint8_t*)w, (uint8_t*)y, (uint8_t*)tiling);
    return (int)aclrtGetLastError(ACL_RT_THREAD_LEVEL);
}
