"""Qualify device-only APC slot selection/precopy on Ascend SD pools.

Synthetic device acceptance; exact pool oracle. Not model serving or EOS/churn.
"""
from collections import deque
import json
import os
from pathlib import Path
from types import SimpleNamespace as NS

import torch
import torch_npu
from vllm_ascend.utils import enable_custom_op


def main():
    assert enable_custom_op()
    torch.npu.set_device(0)
    from betterscale.patches.qwen_gdn.mamba_abi import install
    install()
    from vllm.v1.worker.mamba_utils import MambaSpecDecodeGPUContext
    from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
    init_device_properties_triton()
    torch.manual_seed(109)
    width, rows, columns = 3, 4, 8
    table_cpu = torch.randperm(rows*columns,dtype=torch.int32).reshape(rows,columns)
    table = table_cpu.npu()
    seeds = [torch.randn(rows*columns,width+2,5120).bfloat16(),
             torch.randn(rows*columns,24,128,128)]
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
        state_dim_row_count=tensor([0,0],torch.int32),state_dim_row_stride=tensor([0,0]),
        num_accepted_tokens_out=tensor([1]*rows,torch.int32))
    initial_positions=[10,13,26,29]
    initial_columns=[(n+2)//16 for n in initial_positions]
    computed=tensor(initial_positions,torch.int32)
    state_idx=tensor(initial_columns,torch.int32)
    accepted=tensor([1]*rows,torch.int32)
    src=tensor([0]*rows,torch.int32);bias=tensor([0]*rows,torch.int32)
    mapping=tensor(list(range(rows)),torch.int32)
    cu=tensor([0,3,6,9,12],torch.int32)
    axis=tensor(list(range(rows)),torch.int32)
    counter=tensor([0],torch.int32)
    scheduled=tensor([width]*rows,torch.int32)
    drafts=tensor([width-1]*rows,torch.int32)
    synthetic_computed=tensor([0]*rows,torch.int32)
    def call():
        accepted.copy_((counter+axis)%3+1)
        computed.add_(accepted)
        # V2's generic Triton copy did not compile on this Ascend donor. Reuse
        # the qualified SD copy primitive with an exact synthetic boundary:
        # running=(dst+1)*B-bias, accepted=bias+1 => aligned=(dst+1)*B.
        # A no-copy lane uses running=1, whose next alignment is not reached.
        src.copy_(state_idx)
        bias.copy_(accepted-1)
        destination=(computed+3+15)//16-1
        migrate=(src>=0)&(src!=destination)
        synthetic_computed.copy_(torch.where(migrate,(destination+1)*16-bias-1,0))
        MambaSpecDecodeGPUContext.run_fused_postprocess(ctx,rows,accepted,src,
            scheduled,synthetic_computed,drafts)
        state_idx.copy_(destination)
        accepted.copy_(torch.where(migrate,1,accepted))
        counter.add_(1)
    # Compile outside capture; then reset all device truth.
    call();torch.npu.synchronize()
    graphs=[]
    observations=[]
    for _ in range(2):
        graph=torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            call()
            # Debug-only snapshots: canonical state is NOT double-banked.
            # D2H must not race the next graph mutating the live state pool.
            snapshot=[x.clone() for x in pools]
            receipt=torch.stack((computed,state_idx,accepted,src,bias,counter.expand(rows)))
        graphs.append(graph)
        observations.append(dict(snapshot=snapshot,receipt=receipt,
            host=[torch.empty_like(x,device='cpu',pin_memory=True) for x in snapshot],
            host_receipt=torch.empty_like(receipt,device='cpu',pin_memory=True),
            consumed=torch.npu.Event(),copied=torch.npu.Event(),used=False))
    computed.copy_(torch.tensor(initial_positions,dtype=torch.int32))
    state_idx.copy_(torch.tensor(initial_columns,dtype=torch.int32));counter.zero_()
    for pool,seed in zip(pools,seeds):pool.copy_(seed)
    expected=[x.clone() for x in seeds]
    positions=initial_positions.copy();cols=initial_columns.copy();checks=[]
    torch.npu.synchronize()
    compute=torch.npu.current_stream()
    egress=torch.npu.Stream()
    pending=deque()
    def collect():
        step,bank=pending.popleft()
        bank['copied'].synchronize()
        counts=[(step+i)%3+1 for i in range(rows)]
        positions[:]=[n+c for n,c in zip(positions,counts)]
        previous=cols.copy();cols[:]=[(n+3+15)//16-1 for n in positions]
        post_counts=counts.copy()
        for i,(old,new,count) in enumerate(zip(previous,cols,counts)):
            if old==new:continue
            post_counts[i]=1
            offset=count-1
            source,destination=int(table_cpu[i,old]),int(table_cpu[i,new])
            expected[0][destination,:width+2-offset] = expected[0][source,offset:].clone()
            expected[1][destination] = expected[1][int(table_cpu[i,old+offset])].clone()
        for observed,reference in zip(bank['host'],expected):
            torch.testing.assert_close(observed,reference,rtol=0,atol=0)
        assert bank['host_receipt'].tolist()==[
            positions,cols,post_counts,previous,[c-1 for c in counts],[step+1]*rows]
        checks.append(dict(step=step,computed=positions.copy(),columns=cols.copy(),accepted=post_counts))
    for step in range(24):
        if len(pending)==2:
            collect()
        bank=observations[step%2]
        if bank['used']:
            compute.wait_event(bank['copied'])
        graphs[step%2].replay()
        bank['consumed'].record(compute)
        with torch.npu.stream(egress):
            egress.wait_event(bank['consumed'])
            for host,snapshot in zip(bank['host'],bank['snapshot']):
                host.copy_(snapshot,non_blocking=True)
            bank['host_receipt'].copy_(bank['receipt'],non_blocking=True)
            bank['copied'].record(egress)
        bank['used']=True
        pending.append((step,bank))
    while pending:
        collect()
    Path(os.environ['CAPSULE'],'complete.json').write_text(json.dumps(dict(status='PASS',
        scope='Donor device slot selection and direct-pool precopy, synthetic acceptance, two outstanding waves; not service qualification',
        checks=checks),indent=2))
    print('PASS device APC preprocess/precopy',len(checks),'waves',flush=True)


if __name__=='__main__':main()
