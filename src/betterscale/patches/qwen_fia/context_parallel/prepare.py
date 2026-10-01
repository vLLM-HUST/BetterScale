"""Materialize the paid partial-outlined producer from installed CANN source.

Derived from ascend-op-prof-single-decode ea92f61. The Qwen port fixes the
dimensionful partial offset and adds device-length endpoint scaling, empty
partial identities and zero-KV padding. Installed vendor licenses are kept.
No full-only diagnostic or flat-scheduler build is exposed here.
"""
import argparse
import re
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--output',type=Path,required=True)
p.add_argument('--cann',type=Path,required=True)
a=p.parse_args()
k=a.cann/'opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/fused_infer_attention_score'
a.output.mkdir(parents=True,exist_ok=True);v=a.output/'vendor';v.mkdir(exist_ok=True)
s=(k/'flash_attention_regular.h').read_text()
# CANN9.1 joins this outer else onto the closing-brace line. Match only
# the established branch boundary; all numerical transformation guards remain.
def outer_else(text, start):
    candidates = ('            } \n            else {', '            } else {')
    matches = [re.search('^' + re.escape(marker), text[start:], re.MULTILINE)
               for marker in candidates]
    matches = [start + match.start() for match in matches if match is not None]
    if not matches:
        raise ValueError('unrecognized vendor outer else boundary')
    return min(matches)

for before,after in (
 ('int32_t stN1IdxNow = (BIdx == startBIdx) ? startN1Idx : 0;','int32_t stN1IdxNow = startN1Idx;'),
 ('int32_t enN1IdxNow = (BIdx == endBIdx) ? endN1Idx : curQNBlockNumTmp - 1;','int32_t enN1IdxNow = endN1Idx;')):
    assert s.count(before)==1
    if True:s=s.replace(before,after)
if True:
    begin=s.index('                uint32_t startBIdx = fATilingData->coreInfo.startBIdx[coreIdx];')
    end=s.index('                for (uint32_t BIdx = startBIdx;',begin)
    s=s[:begin]+'''                __gm__ uint32_t* desc = reinterpret_cast<__gm__ uint32_t*>(params.tiling + 2560 + coreIdx * 32);
                uint32_t startBIdx = desc[0];
                uint32_t endBIdx = desc[1];
                uint32_t startN1Idx = desc[2];
                uint32_t startS2Idx = desc[3];
                uint32_t endS2Idx = desc[4];
                uint64_t gmOffsetLseFD = desc[5];
                uint64_t gmOffsetOFD = gmOffsetLseFD * 128;

'''+s[end:]
    # The admitted operator contract is Q1 with one complete GQA group per
    # lane, not general prefill. Remove generic Q/head traversal, uniformly
    # for every length and every split/no-split plan in that contract.
    begin=s.index('                for (uint32_t BIdx = startBIdx;')
    end=outer_else(s,begin)
    s=s[:begin]+'''                for (uint32_t BIdx = startBIdx; BIdx <= endBIdx; BIdx++) {
                    uint32_t kv = static_cast<uint32_t>(gActualKvseqlen.GetValue(BIdx));
                    uint32_t blocks = (kv + 511) / 512;
                    uint32_t lo = BIdx == startBIdx ? startS2Idx : 0;
                    uint32_t hi = BIdx == endBIdx ? endS2Idx : blocks;
                    bool split = hi - lo < blocks;
                    runMainLoop(coreIdx, BIdx, startN1Idx, 0, split, lo, hi,
                        gmOffsetLseFD, gmOffsetOFD, globalTensors, pseQ, pseKv);
                    if (split) {
                        gmOffsetLseFD += groupSize;
                        gmOffsetOFD += groupSize * embedV;
                    }
                }
'''+s[end:]
    # FD describes the shared scheduler/storage, not every segment's output
    # semantics. A whole segment must use the normal epilogue; only an actual
    # partial needs the FD math protocol. Keep one shared pipeline body rather
    # than duplicating it into full/partial template instantiations.
    begin=s.index('__aicore__ inline void runMainLoop(')
    tail=s[begin:].replace('if constexpr (IS_FD)', 'if (isSplitKV)')
    tail=tail.replace('if constexpr (!IS_FD)', 'if (!isSplitKV)')
    s=s[:begin]+tail
start=s.index('            AscendC::ListTensorDesc keyListTensorDescInit')
end=s.index('            AscendC::GlobalTensor<ElementK> gK;',start)
s=s[:start]+'''            __gm__ uint8_t* currentKey = params.k;
            __gm__ uint8_t* currentValue = params.v;
'''+s[end:]
# Uniform metadata branch after all local pipelines drain. With no partial
# nodes every producer has already written final O; no cross-core consumer
# exists. Keep one kernel/graph for both paths, including inactive producers.
before='''            if constexpr (IS_FD) {
            AscendC::SyncAll();'''
assert s.count(before)==1
s=s.replace(before,'''            if constexpr (IS_FD) {
                if (fATilingData->totalSplitNodeNum == 0) { return; }
            AscendC::SyncAll();''')
# Q1 NO_MASK has one common softmax/rescale pipeline. Pass segment state
# through the vendor APIs instead of duplicating their large inline bodies.
if True:
    def branch_body_and_end(text, start):
        opening=text.index('{',start); depth=1; end=opening+1
        while depth:
            if text[end]=='{': depth+=1
            elif text[end]=='}': depth-=1
            end+=1
        return text[opening+1:end-1],end
    def fuse_branch(text, marker, transform):
        assert text.count(marker)==1
        begin=text.index(marker)
        body,end=branch_body_and_end(text,begin)
        following=end
        while text[following].isspace(): following+=1
        assert text.startswith('else {',following)
        _,finish=branch_body_and_end(text,following)
        return text[:begin]+transform(body)+text[finish:]
    marker='if (isSplitKV) {\n                                epilogueOnlineSoftmax('
    def common_softmax(body):
        assert body.count('                                    false);')==1
        return body.replace('                                    false);',
            '                                    false,\n                                    isSplitKV ? false : startsWithMaskThenNomaskFlag);')
    s=fuse_branch(s,marker,common_softmax)
    marker='if (isSplitKV) {\n                            LayoutLse layoutgmLse'
    # kvEnd equals kvSLoopNumTotal for a complete segment. SplitKVParams is
    # already a runtime mode in the inherited rescale API (default=false).
    s=fuse_branch(s,marker,lambda body:body)
(v/'flash_attention_regular.h').write_text(s)
s=(k/'flash_attention_interface.cpp').read_text().split('    // 新增：')[0]+'}\n'
s=s.replace('#include "flash_attention_regular_decode.h"','')
assert s.count('__global__ __aicore__ void FAInfer(')==1
s=s.replace('__global__ __aicore__ void FAInfer(','__aicore__ inline void FAInfer(')
(v/'flash_attention_interface.cpp').write_text(s)
# Keep the cold cross-core reduction out of the producer instruction body.
# It remains a device function in the same kernel, after the same barrier.
c=(k/'attn_infra/epilogue/block/CombineScale.hpp').read_text()
c=c.replace('../../../attn_infra/', 'attn_infra/')
assert c.count('__aicore__ inline void operator()(')==1
c=c.replace('__aicore__ inline void operator()(',
    '__aicore__ __attribute__((noinline)) void operator()(')
(v/'combine_outlined.hpp').write_text(c)
source=Path(__file__).parent
s=(source/'lane_kernel.cpp').read_text()
if not (k/'attn_infra/detail/alignment.hpp').is_file():
    if not (k/'attn_infra/detail/fused_alignment.hpp').is_file():
        raise FileNotFoundError('vendor alignment header is absent')
    s=s.replace('attn_infra/detail/alignment.hpp', 'attn_infra/detail/fused_alignment.hpp')
    s=s.replace('using NpuArch::Detail::Alignment::CeilDiv;',
                'using NpuArch::Detail::Alignment::CeilDiv;\nconstexpr uint32_t FLOAT_PER_BLOCK = 32 / sizeof(float);')
interface=(k/'flash_attention_interface.cpp').read_text()
if 'class LAYOUT_K = layout::ColumnMajor,' in interface:
    assert 'class LAYOUT_V = layout::RowMajor,' in interface
    old='SplitFuse::FAInfer<bfloat16_t,bfloat16_t,float,true,true,'
    assert s.count(old)==1
    # New vendor template exposes K/V layouts before its existing parameters.
    # Preserve the vendor defaults, not a new KV storage interpretation.
    s=s.replace(old, 'SplitFuse::FAInfer<NpuArch::layout::ColumnMajor,NpuArch::layout::RowMajor,'
                     'bfloat16_t,bfloat16_t,float,true,true,')


s=s.replace('#include "vendor/flash_attention_interface.cpp"',
    '#include "vendor/combine_outlined.hpp"\n#include "vendor/flash_attention_interface.cpp"')

(a.output/'kernel.cpp').write_text(s)
(a.output/'tiling.hpp').write_bytes((source/'tiling.hpp').read_bytes())

# Verification causal/partial-row correction.
h=a.output/'vendor/flash_attention_regular.h';s=h.read_text()
before='''                    if (split) {
                        gmOffsetLseFD += groupSize;
                        gmOffsetOFD += groupSize * embedV;
                    }'''
assert s.count(before)==1
s=s.replace(before,'''                    if (split) {
                        uint32_t qrows = static_cast<uint32_t>(gActualQseqlen.GetValue(BIdx));
                        if (BIdx != 0) { qrows -= static_cast<uint32_t>(gActualQseqlen.GetValue(BIdx - 1)); }
                        gmOffsetLseFD += groupSize * qrows;
                        gmOffsetOFD += groupSize * qrows * embedV;
                    }''')
# Causal masked and nonmasked overloads already accept runtime split state.
# The full path has kvStart==0, so the partial localLastNoMaskStackId formula
# also equals the full path's last-nonmasked index. Share each call body.
def fuse_first_after(text,marker):
    start=text.index('if (isSplitKV) {',text.index(marker))
    def end_block(start):
        opening=text.index('{',start);depth=1;end=opening+1
        while depth:
            depth+=(text[end]=='{')-(text[end]=='}');end+=1
        return opening,end
    opening,end=end_block(start);following=end
    while text[following].isspace():following+=1
    assert text.startswith('else {',following)
    _,finish=end_block(following)
    return text[:start]+text[opening+1:end-1]+text[finish:]
s=fuse_first_after(s,'if constexpr (MASK_TYPE == FaiKernel::MaskType::MASK_CAUSAL)')
s=fuse_first_after(s,'uint32_t noMaskStackSeqNum = (triUp + 1) / MAX_KV_STACK_LEN;')
h.write_text(s)
k=a.output/'kernel.cpp';s=k.read_text();assert s.count('MaskType::NO_MASK')==1
s=s.replace('MaskType::NO_MASK','MaskType::MASK_CAUSAL')
s=s.replace('// Q1 bottom-right causal makes every valid KV position visible, including\n // every context shard. Tail masking still follows each actual KV length.', '// Verification Q1..16 preserves bottom-right causal masking within the query group.')
k.write_text(s)

# Original producer specialized, with only the partial path outlined.
k = a.cann / 'opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/fused_infer_attention_score'
original = (k / 'flash_attention_regular.h').read_text()
h = a.output / 'vendor/flash_attention_regular.h'
current = h.read_text()

def bounds(s):
    begin = s.index('__aicore__ inline void runMainLoop(')
    opening = s.index('{', begin)
    depth, end = 1, opening + 1
    while depth:
        depth += (s[end] == '{') - (s[end] == '}')
        end += 1
    return begin, opening, end

begin, opening, end = bounds(original)
producer = original[begin:end]
assert producer.count('if constexpr (IS_FD)') == 6
assert producer.count('if constexpr (!IS_FD)') == 2
assert producer.count('            bool isSplitKV,\n') == 1
producer = producer.replace('            bool isSplitKV,\n', '')
producer = producer.replace('runMainLoop(', 'runSegment(')
producer = producer.replace('IS_FD', 'SPLIT_SEGMENT').replace('isSplitKV', 'SPLIT_SEGMENT')
producer = 'template<bool SPLIT_SEGMENT>\n        ' + producer
if False:
    producer = producer.replace('__aicore__ inline void runSegment(',
        '__aicore__ __attribute__((noinline)) void runSegment(')
# Selection is uniform across the AIC + two AIVs for each owned segment.
# Full and partial are compiled separately; initialization, drain and merge
# remain outside the producer, exactly as in the baseline scheduler.
args = 'coreIdx, BIdx, qNBlockIdx, qSBlockIdx, stS2IdxNow, enS2IdxNow,\n                    gmOffsetLseFD, gmOffsetOFD, globalTensors, pseQ, pseKv'
partial_call = 'runSegment<true>'
if True:
    signature = original[begin:opening].replace('            bool isSplitKV,\n', '')
    signature = signature.replace('runMainLoop(', 'runPartialSegment(')
    signature = signature.replace('__aicore__ inline', '__aicore__ __attribute__((noinline))')
    # Template wrapper avoids the ASC export issue previously seen with a
    # non-template device helper; the partial producer remains in this binary.
    producer += '\n\n        template<bool PARTIAL>\n        ' + signature + '''{
            static_assert(PARTIAL, "partial wrapper only");
            runSegment<true>(''' + args + ''');
        }'''
    partial_call = 'runPartialSegment<true>'
dispatcher = original[begin:opening] + '''{
            if (isSplitKV) {
                ''' + partial_call + '(' + args + ''');
            } else {
                runSegment<false>(''' + args + ''');
            }
        }'''
begin, _, end = bounds(current)
h.write_text(current[:begin] + producer + '\n\n        ' + dispatcher + current[end:])
actual = h.read_text()
assert actual.startswith(current[:begin]) and actual.endswith(current[end:])
p = a.output / 'kernel.cpp'
s = p.read_text()
assert s.count('quota_head_lanes') == 2
p.write_text(s.replace('quota_head_lanes',
    'quota_specialized_producers_outlined' if False else
    'quota_specialized_partial_outlined' if True else 'quota_specialized_segments'))
(a.output / 'control-contract.txt').write_text(
    'Original vendor producer specialized on compile-time segment kind. '
    'Runtime dispatch per segment in one kernel; shared scheduler, FD scratch, '
    'event initialization, drain and merge unchanged. No shape/capture fallback. '
    + ('Both producers outlined.' if False else
       'Partial producer behind noinline template wrapper; full inline.' if True else 'Both inline.') + '\n')

h=a.output/'vendor/flash_attention_regular.h'
s=h.read_text()
needle='uint64_t gmOffsetOFD = gmOffsetLseFD * 128;'
assert s.count(needle)==1
h.write_text(s.replace(needle,'uint64_t gmOffsetOFD = gmOffsetLseFD * embedV;'))
k=a.output/'kernel.cpp';s=k.read_text()
assert s.count('quota_specialized_partial_outlined')==2
k.write_text(s.replace('quota_specialized_partial_outlined','betterscale_context_parallel'))

# Host lengths are scheduling envelopes. Scale BOTH endpoints by the current
# device tile count, so adjacent pieces still meet exactly when MTP rejects.
s=h.read_text()
old='''                    uint32_t lo = BIdx == startBIdx ? startS2Idx : 0;
                    uint32_t hi = BIdx == endBIdx ? endS2Idx : blocks;
                    bool split = hi - lo < blocks;
                    runMainLoop(coreIdx, BIdx, startN1Idx, 0, split, lo, hi,
                        gmOffsetLseFD, gmOffsetOFD, globalTensors, pseQ, pseKv);'''
new='''                    uint32_t lo = BIdx == startBIdx ?
                        static_cast<uint64_t>(startS2Idx) * blocks / desc[6] : 0;
                    uint32_t hi = BIdx == endBIdx ?
                        static_cast<uint64_t>(endS2Idx) * blocks / desc[7] : blocks;
                    // Keep the static partial/reducer ownership even when one
                    // scaled piece covers everything or another becomes empty.
                    bool split = (BIdx == startBIdx && startS2Idx != 0) ||
                        (BIdx == endBIdx && endS2Idx != desc[7]);
                    if (split && lo == hi) {
#ifdef __DAV_C220_VEC__
                        uint32_t qrows = static_cast<uint32_t>(gActualQseqlen.GetValue(BIdx));
                        if (BIdx != 0) { qrows -= static_cast<uint32_t>(gActualQseqlen.GetValue(BIdx - 1)); }
                        writeEmptyPartial(globalTensors, gmOffsetLseFD, gmOffsetOFD, qrows * groupSize);
#endif
                    } else {
                        runMainLoop(coreIdx, BIdx, startN1Idx, 0, split, lo, hi,
                            gmOffsetLseFD, gmOffsetOFD, globalTensors, pseQ, pseKv);
                    }'''
assert s.count(old)==1
s=s.replace(old,new)
start=s.index('__gm__ uint32_t* desc =')
pos=outer_else(s,start)
s=s[:pos]+'''                // Only zero-KV graph padding follows the live request prefix.
                // Reuse the vendor's empty-context output initialization.
                if (coreIdx == 0) {
                    uint32_t live = *reinterpret_cast<__gm__ uint32_t*>(params.tiling + 3328);
                    for (uint32_t b = live; b < batch; ++b) {
                        runMainLoop(coreIdx, b, 0, 0, false, 0, 0, 0, 0,
                            globalTensors, pseQ, pseKv);
                    }
                }
'''+s[pos:]
needle='        template<bool SPLIT_SEGMENT>'
helper='''        // Empty partial is the reduction identity: O=0 and logsumexp=-inf.
        // DMA writes avoid scalar D-cache writeback on adjacent partial rows.
        // These are the same reserved UB regions/events as vendor InitOut.
        __aicore__ inline void writeEmptyPartial(GlobalTensorBundle& tensors,
            uint64_t lseOffset, uint64_t oOffset, uint32_t rows) {
#ifdef __DAV_C220_VEC__
            if (AscendC::GetSubBlockIdx() != 0) { return; }
            auto zeros = resource.ubBuf.template GetBufferByByte<float>(0);
            // Q16 x8 heads x256 FP32 zero output occupies128KiB.
            // Keep the LSE fill disjoint, within AtlasA2's192KiB UB.
            auto negInf = resource.ubBuf.template GetBufferByByte<float>(8 * 16384);
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID6);
            AscendC::Duplicate(zeros, 0.0f, rows * embedV);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID6);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID6);
            AscendC::DataCopy(tensors.gOFD[oOffset], zeros, rows * embedV);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID6);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID7);
            AscendC::Duplicate(negInf, -__builtin_inff(), rows);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID7);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID7);
            AscendC::DataCopy(tensors.gLseFD[lseOffset], negInf, rows);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID7);
            AscendC::PipeBarrier<PIPE_ALL>();
#endif
        }

'''
assert s.count(needle)==1
h.write_text(s.replace(needle,helper+needle))
