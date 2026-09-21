"""Alternating captured metadata banks, changed device inputs, exact CPU oracle."""
import json
import os
from pathlib import Path
from types import SimpleNamespace as NS
import torch
import torch_npu
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
from device_metadata import publish_slots_reference
from device_slots import publish_slots
from layout_probe import elapsed


def metadata(decode, device):
    n=8 if decode else 4
    pre,ver=([],list(range(n))) if decode else ([1,3],[2,0])
    table=torch.arange(n*16,dtype=torch.int32,device=device).reshape(n,16)
    return NS(live=n,width=3,decode=decode,
        device_slot_source=(table,torch.full((n,),31,dtype=torch.int32,device=device),16,pre,ver),
        cu=torch.arange(10,dtype=torch.int32,device=device)*3,
        initial=torch.zeros(9,dtype=torch.bool,device=device),
        prefill_ids=torch.tensor(pre+[0]*(9-len(pre)),device=device),
        verify_ids=torch.tensor(ver+[0]*(9-len(ver)),device=device),
        prefill_conv=torch.full((9,1),-1,dtype=torch.int32,device=device),
        verify_conv=torch.full((9,1),-1,dtype=torch.int32,device=device),
        prefill=NS(state=torch.zeros((9,2),dtype=torch.int64,device=device)),
        verify=NS(slots=torch.full((9,3),-1,dtype=torch.int64,device=device)))


def fields(m):
    return [m.initial,m.prefill_conv,m.verify_conv,m.prefill.state,m.verify.slots]


def main():
    torch.npu.set_device(0);init_device_properties_triton();torch.set_num_threads(4)
    rows=[]
    for decode in (True,False):
        cpu=metadata(decode,'cpu');banks=[metadata(decode,'npu') for _ in range(2)]
        graphs=[]
        for m in banks:
            publish_slots(m);torch.npu.synchronize()
            g=torch.npu.NPUGraph()
            with torch.npu.graph(g):publish_slots(m)
            graphs.append(g)
        for wave in range(24):
            # GPU sequence length/table, not CPU acceptance feedback, author output.
            seq=torch.arange(cpu.live,dtype=torch.int32)*3+wave+1
            table=torch.arange(cpu.live*16,dtype=torch.int32).reshape(cpu.live,16).roll(wave,0)
            table[0,1]=0
            cpu.device_slot_source[0].copy_(table);cpu.device_slot_source[1].copy_(seq)
            m=banks[wave%2]
            m.device_slot_source[0].copy_(table);m.device_slot_source[1].copy_(seq)
            graphs[wave%2].replay();publish_slots_reference(cpu)
            for actual,expected in zip(fields(m),fields(cpu)):
                torch.testing.assert_close(actual.cpu(),expected,rtol=0,atol=0)
        controls=[]
        for fn in (publish_slots_reference,publish_slots):
            fn(banks[0]);torch.npu.synchronize()
            g=torch.npu.NPUGraph()
            with torch.npu.graph(g):
                for _ in range(8):fn(banks[0])
            controls.append(g)
        timing=[[],[]]
        for arm in (0,1,1,0):
            for _ in range(3):controls[arm].replay()
            timing[arm].append(elapsed(controls[arm])/8)
        row=dict(decode=decode,waves=24,exact=True,microseconds=timing)
        rows.append(row);print(json.dumps(row),flush=True)
    Path(os.environ['CAPSULE'],'receipt.json').write_text(json.dumps(dict(passed=True,rows=rows),indent=2))


if __name__=='__main__':
    with torch.inference_mode():main()
