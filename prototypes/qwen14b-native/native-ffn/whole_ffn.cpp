/* Native AscendC research extension; CANN v2 attribution and license in kernel.cpp. */
#include "kernel.cpp"
namespace whole_ffn {
using namespace native_bf16;
__aicore__ constexpr MatmulConfig DownConfig() {
    auto c = GetMDLConfig(false, false, 0, false, false, false, true);
    c.singleCoreM = 128; c.singleCoreN = 256; c.singleCoreK = I;
    c.basicM = 128; c.basicN = 256; c.basicK = 64;
    return c;
}
__aicore__ constexpr MatmulApiStaticTiling DownTiling() {
    auto t = GetMatmulApiTiling<A,B,C,Bias>(DownConfig());
    t.stepM = 1; t.stepN = 1; t.stepKa = 4; t.stepKb = 4;
    t.depthA1 = 8; t.depthB1 = 8; t.isBias = false;
    return t;
}
constexpr auto DOWN = DownTiling();
using DownMM = MatmulImpl<A,B,C,Bias,DOWN>;
__aicore__ inline void Gate(GM_ADDR x, GM_ADDR w, GM_ADDR z, GM_ADDR scratch, uint32_t rows) {
    TPipe pipe;
    MM mm;
    if ASCEND_IS_AIC { mm.SetSubBlockIdx(0); mm.Init(static_cast<const TCubeTiling*>(nullptr)); }
    DenseSwiglu<256> op(mm, pipe);
    op.Init(x,w,z,scratch,rows,256); op.Process();
    if ASCEND_IS_AIC { mm.End(); }
    PipeBarrier<PIPE_ALL>();
    SyncAll<false>(); // all row-block Z channels are now published to down
}
__aicore__ inline void Down(GM_ADDR z, GM_ADDR w, GM_ADDR y, uint32_t rows) {
    TPipe pipe;
    if ASCEND_IS_AIC {
        DownMM mm; mm.SetSubBlockIdx(0); mm.Init(static_cast<const TCubeTiling*>(nullptr));
        GlobalTensor<bfloat16_t> zg, wg, yg;
        zg.SetGlobalBuffer((__gm__ bfloat16_t*)z);
        wg.SetGlobalBuffer((__gm__ bfloat16_t*)w);
        yg.SetGlobalBuffer((__gm__ bfloat16_t*)y);
        uint32_t nm = (rows + 127) / 128;
        // Two2560-channel weight panels, each67.5MiB. Keep full K per GEMM.
        for (uint32_t tile = GetBlockIdx(); tile < nm * 20; tile += CORES) {
            uint32_t pn = tile / (nm * 10), local = tile % (nm * 10);
            uint32_t aa=nm,bb=10;
            while(bb) { uint32_t rem=aa%bb;aa=bb;bb=rem; }
            uint32_t mr=local%nm, nr=(local+local/(nm/aa*10))%10+pn*10;
            uint32_t row=mr*128, live=rows-row<128?rows-row:128;
            mm.SetOrgShape(rows,H,I); mm.SetSingleShape(live,256,I);
            mm.SetTensorA(zg[(uint64_t)row*I],false);
            mm.SetTensorB(wg[(uint64_t)nr*256*I],false);
            mm.Iterate();mm.GetTensorC(yg[(uint64_t)row*H+nr*256],0);
        }
        mm.End();
    }
    PipeBarrier<PIPE_ALL>();
    SyncAll<false>(); // all consumers finish before the next row-block stage
}
}
extern "C" __global__ __aicore__ void qwen_whole_ffn(
    GM_ADDR x,GM_ADDR w,GM_ADDR dw,GM_ADDR z,GM_ADDR scratch,GM_ADDR y,
    uint32_t rows,uint32_t chunk) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    for(uint32_t r=0;r<rows;r+=chunk) {
        uint32_t live=rows-r<chunk?rows-r:chunk;
        whole_ffn::Gate(x+(uint64_t)r*native_bf16::H*2,w,
                       z+(uint64_t)r*native_bf16::I*2,scratch,live);
        whole_ffn::Down(z+(uint64_t)r*native_bf16::I*2,dw,
                       y+(uint64_t)r*native_bf16::H*2,live);
    }
}
extern "C" int launch_whole_ffn(void*x,void*w,void*dw,void*z,void*scratch,void*y,
    uint32_t rows,uint32_t chunk,void*stream) {
    if(!x||!w||!dw||!z||!scratch||!y||rows<1||rows>4096||
       (chunk!=512&&chunk!=1024&&chunk!=4096))return 1;
    qwen_whole_ffn<<<24,nullptr,stream>>>((uint8_t*)x,(uint8_t*)w,(uint8_t*)dw,
        (uint8_t*)z,(uint8_t*)scratch,(uint8_t*)y,rows,chunk);
    return (int)aclrtGetLastError(ACL_RT_THREAD_LEVEL);
}
