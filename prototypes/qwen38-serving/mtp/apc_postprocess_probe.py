"""Native V1 caller -> Ascend ABI bridge -> exact device boundary-copy oracle."""
import json
import os
from pathlib import Path
from types import SimpleNamespace as NS
import torch
import torch_npu
from vllm_ascend.utils import enable_custom_op
from betterscale.patches.qwen_gdn.mamba_abi import install


def main():
    assert enable_custom_op()
    torch.npu.set_device(0)
    install()
    from vllm.v1.worker.mamba_utils import MambaSpecDecodeGPUContext
    from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
    assert not is_conv_state_dim_first()
    torch.manual_seed(101)
    width = int(os.environ.get("MTP_TOKENS","2"))+1
    assert 2 <= width <= 5
    columns = width+2
    table_cpu = torch.arange(4*columns,dtype=torch.int32).reshape(4,columns)
    table = table_cpu.npu()
    seeds = [torch.randn(4*columns,width+2,5120).bfloat16(),torch.randn(4*columns,24,128,128)]
    pools = [x.npu() for x in seeds]
    def tensor(values,dtype=torch.int64):
        return torch.tensor(values,dtype=dtype,device='npu')
    ctx = NS(is_initialized=True,num_layers=1,num_state_types=2,block_size=16,
        block_table_ptrs=tensor([table.data_ptr()]),block_table_stride_req=columns,
        state_base_addrs=tensor([x.data_ptr() for x in pools]),
        state_block_strides=tensor([x.stride(0)*x.element_size() for x in pools]),
        state_elem_sizes=tensor([2,4],torch.int32),
        state_inner_sizes=tensor([5120,24*128*128]),
        state_conv_widths=tensor([width+2,0],torch.int32),
        state_group_indices=tensor([0,0],torch.int32),
        state_dim_row_count=tensor([0,0],torch.int32),
        state_dim_row_stride=tensor([0,0]),
        num_accepted_tokens_out=tensor([1]*4,torch.int32))
    accepted,source,computed = [tensor([1]*4,torch.int32) for _ in range(3)]
    scheduled,drafts = tensor([width]*4,torch.int32),tensor([width-1]*4,torch.int32)
    def call():
        MambaSpecDecodeGPUContext.run_fused_postprocess(ctx,4,accepted,source,scheduled,computed,drafts)
    cases = [([width,width,1,1],[1,0,0,0],[15-bias,15-bias,5,7])
             for bias in range(width)]
    with torch.inference_mode():
        # Warmup with no boundary copy, then exercise live decisions in replay.
        computed.fill_(1); call(); torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            call()
        rows=[]
        for counts,cols,positions in cases:
            accepted.copy_(torch.tensor(counts,dtype=torch.int32))
            source.copy_(torch.tensor(cols,dtype=torch.int32))
            computed.copy_(torch.tensor(positions,dtype=torch.int32))
            for pool,seed in zip(pools,seeds):pool.copy_(seed)
            expected = [x.clone() for x in seeds]
            expected_count = counts.copy()
            for i,(count,col,position) in enumerate(zip(counts,cols,positions)):
                running = position+1
                aligned = (running+count-1)//16*16
                if aligned<running:continue
                bias,dest = aligned-running,aligned//16-1
                if col==dest:expected_count[i]=1
                if col==dest and bias==0:continue
                src_id,dst_id = int(table_cpu[i,col]),int(table_cpu[i,dest])
                expected[0][dst_id,:width+2-bias] = seeds[0][src_id,bias:]
                expected[1][dst_id] = seeds[1][int(table_cpu[i,col+bias])]
            graph.replay();torch.npu.synchronize()
            for actual,reference in zip(pools,expected):
                torch.testing.assert_close(actual.cpu(),reference,rtol=0,atol=0)
            torch.testing.assert_close(ctx.num_accepted_tokens_out.cpu(),torch.tensor(expected_count,dtype=torch.int32),rtol=0,atol=0)
            rows.append(dict(accepted=counts,source=cols,computed=positions,exact=True))
        Path(os.environ['CAPSULE'],'receipt.json').write_text(json.dumps(dict(passed=True,mtp_tokens=width-1,cases=rows),indent=2))
        print('native V1 / Ascend SD postprocess bridge PASS',flush=True)


if __name__ == '__main__':main()
