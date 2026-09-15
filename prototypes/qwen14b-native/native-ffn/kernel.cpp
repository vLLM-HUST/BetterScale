/*
 * Derived from Huawei grouped_matmul_swiglu_quant (2025), pinned in upstream/.
 * Modifications (2026): dense BF16 operands and outputs; remove quantization;
 * consume paired channel tiles rather than full rows. Native MatmulImpl and
 * the two-slot producer/consumer barrier protocol are retained.
 * CANN Open Software License Agreement Version 2.0; see upstream/CANN-LICENSE.
 */
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include <acl/acl.h>
#include <cstdint>
using namespace AscendC;
using namespace matmul;

#ifndef NATIVE_CM
#define NATIVE_CM 128
#endif
#ifndef NATIVE_CN
#define NATIVE_CN 256
#endif
namespace native_bf16 {
constexpr uint32_t CM=NATIVE_CM, CN=NATIVE_CN;
constexpr uint32_t H = 5120, I = 13824, N = 2 * I, CORES = 24;
// Keep the original native MDL GEMM, not a scalar/Triton K-loop rewrite.
__aicore__ constexpr MatmulConfig MakeConfig() {
    auto c = GetMDLConfig(false, false, 0, true, false, false, true);
    c.singleCoreM = CM; c.singleCoreN = CN; c.singleCoreK = H;
    c.basicM = CM; c.basicN = CN; c.basicK = 128;
    return c;
}
constexpr MatmulConfig CFG = MakeConfig();
using A = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t, false>;
using B = MatmulType<TPosition::GM, CubeFormat::NZ, bfloat16_t, false>;
using C = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t>;
using Bias = MatmulType<TPosition::GM, CubeFormat::ND, float>;
__aicore__ constexpr MatmulApiStaticTiling MakeTiling() {
    auto t = GetMatmulApiTiling<A, B, C, Bias>(CFG);
    // BF16 operands need twice the bytes of the original INT8 pipeline.
    t.stepM = 1; t.stepN = 1; t.stepKa = 2; t.stepKb = 2;
    t.depthA1 = 4; t.depthB1 = 4; t.isBias = false;
    return t;
}
constexpr MatmulApiStaticTiling MDL = MakeTiling();
using MM = MatmulImpl<A, B, C, Bias, MDL>;

template<uint32_t VC>
class DenseSwiglu {
    static constexpr uint32_t VR = 8, COUNT = VR * VC;
    MM &mm;
    TPipe &pipe;
    GlobalTensor<bfloat16_t> x, weight, output, scratch;
    TQue<QuePosition::VECIN, 1> inQueue;
    TQue<QuePosition::VECOUT, 1> outQueue;
    TBuf<TPosition::VECCALC> fp32, activation;
    uint32_t rows, slab;
public:
    __aicore__ inline DenseSwiglu(MM &m, TPipe &p): mm(m), pipe(p) {}
    __aicore__ inline void Init(GM_ADDR xp, GM_ADDR wp, GM_ADDR yp, GM_ADDR sp,
                                uint32_t totalRows, uint32_t slabRows) {
        rows = totalRows; slab = slabRows;
        x.SetGlobalBuffer((__gm__ bfloat16_t*)xp);
        weight.SetGlobalBuffer((__gm__ bfloat16_t*)wp);
        output.SetGlobalBuffer((__gm__ bfloat16_t*)yp);
        scratch.SetGlobalBuffer((__gm__ bfloat16_t*)sp);
        if ASCEND_IS_AIV {
            pipe.InitBuffer(inQueue, 2, 2 * COUNT * sizeof(bfloat16_t));
            pipe.InitBuffer(outQueue, 2, COUNT * sizeof(bfloat16_t));
            pipe.InitBuffer(fp32, 2 * COUNT * sizeof(float));
            pipe.InitBuffer(activation, COUNT * sizeof(float));
        }
    }
    __aicore__ inline void Cube(uint32_t start, uint32_t live, uint64_t base) {
        uint32_t nm = (live + CM - 1) / CM, nn = N / CN;
        for (uint32_t tile = GetBlockIdx(); tile < nm * nn; tile += CORES) {
            uint32_t mt = tile / nn, nt = tile % nn;
            uint32_t rm = mt * CM;
            uint32_t take = live - rm < CM ? live - rm : CM;
            mm.SetOrgShape(live, N, H);
            mm.SetSingleShape(take, CN, H);
            mm.SetTensorA(x[(uint64_t)(start + rm) * H], false);
            // BF16 NZ has16x16 inner blocks; H,N and tile offsets are aligned.
            mm.SetTensorB(weight[(uint64_t)nt * CN * H], false);
            mm.IterateAll<false>(scratch[base + (uint64_t)rm * N + nt * CN], 0);
        }
    }
    __aicore__ inline void Vector(uint32_t start, uint32_t live, uint64_t base) {
        uint32_t nc = I / VC, nr = (live + VR - 1) / VR;
        for (uint32_t tile = GetBlockIdx(); tile < nr * nc; tile += CORES * 2) {
            uint32_t r = (tile / nc) * VR, c = (tile % nc) * VC;
            uint32_t take = live - r < VR ? live - r : VR;
            uint32_t count = take * VC;
            auto in = inQueue.AllocTensor<bfloat16_t>();
            DataCopyExtParams load{(uint16_t)take, VC * 2, (N - VC) * 2, 0, 0};
            DataCopyPadExtParams<bfloat16_t> pad{false, 0, 0, 0};
            DataCopyPad(in, scratch[base + (uint64_t)r * N + c], load, pad);
            DataCopyPad(in[COUNT], scratch[base + (uint64_t)r * N + I + c], load, pad);
            inQueue.EnQue(in);
            in = inQueue.DeQue<bfloat16_t>();
            auto f = fp32.Get<float>();
            Cast(f, in, RoundMode::CAST_NONE, count);
            Cast(f[COUNT], in[COUNT], RoundMode::CAST_NONE, count);
            PipeBarrier<PIPE_V>();
            auto act = activation.Get<float>();
            // Native API computes src0 * swish(src1): up then gate.
            SwiGLU<float, false>(act, f[COUNT], f, 1.0f, count);
            PipeBarrier<PIPE_V>();
            auto out = outQueue.AllocTensor<bfloat16_t>();
            Cast(out, act, RoundMode::CAST_RINT, count);
            outQueue.EnQue(out);
            out = outQueue.DeQue<bfloat16_t>();
            DataCopyExtParams store{(uint16_t)take, VC * 2, 0, (I - VC) * 2, 0};
            DataCopyPad(output[(uint64_t)(start + r) * I + c], out, store);
            // Queue free does not by itself order VECCALC reuse by the next tile.
            PipeBarrier<PIPE_ALL>();
            outQueue.FreeTensor(out);
            inQueue.FreeTensor(in);
        }
    }
    __aicore__ inline void Process(bool vectorEnabled = true) {
        uint32_t loops = (rows + slab - 1) / slab;
        for (uint32_t s = 0; s < loops; ++s) {
            uint32_t start = s * slab;
            uint32_t live = rows - start < slab ? rows - start : slab;
            uint64_t base = (uint64_t)(s % 2) * slab * N;
            if ASCEND_IS_AIC {
                if (s >= 2) SyncAll<false>(); // wait for the slot's old reader
                Cube(start, live, base);
                SyncAll<false>();            // publish completed slab
            }
            if ASCEND_IS_AIV {
                SyncAll<false>();            // wait until both half-rows exist
                if (vectorEnabled) Vector(start, live, base);
                if (s + 2 < loops) SyncAll<false>(); // release slot for reuse
            }
        }
    }
};
} // namespace native_bf16

extern "C" __global__ __aicore__ void qwen_bf16_native_swiglu(
    GM_ADDR x, GM_ADDR w, GM_ADDR y, GM_ADDR scratch, uint32_t rows, uint32_t slab, uint32_t vc) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    TPipe pipe;
    native_bf16::MM mm;
    if ASCEND_IS_AIC { mm.SetSubBlockIdx(0); mm.Init(static_cast<const TCubeTiling*>(nullptr)); }
    if (vc == 0) {
        native_bf16::DenseSwiglu<256> op(mm, pipe);
        op.Init(x,w,y,scratch,rows,slab); op.Process(false);
    } else if (vc == 256) {
        native_bf16::DenseSwiglu<256> op(mm, pipe);
        op.Init(x,w,y,scratch,rows,slab); op.Process();
    } else {
        native_bf16::DenseSwiglu<512> op(mm, pipe);
        op.Init(x,w,y,scratch,rows,slab); op.Process();
    }
}

extern "C" int launch_native_ffn(void *x, void *w, void *y, void *scratch,
    uint32_t rows, uint32_t slab, uint32_t vc, void *stream) {
    if (!x || !w || !y || !scratch || rows == 0 || rows > 4096 ||
        slab < 128 || slab > 4096 || slab % 128 || (vc != 0 && vc != 256 && vc != 512)) return 1;
    qwen_bf16_native_swiglu<<<24, nullptr, stream>>>(
        (uint8_t*)x,(uint8_t*)w,(uint8_t*)y,(uint8_t*)scratch,rows,slab,vc);
    return (int)aclrtGetLastError(ACL_RT_THREAD_LEVEL);
}
