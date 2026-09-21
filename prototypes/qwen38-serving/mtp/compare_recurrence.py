"""Same-host FULL replay: direct K-V MTP2 recurrence vs pinned native V-K."""
import json
import os
from pathlib import Path
import torch
import torch_npu
from vllm_ascend.utils import enable_custom_op
from betterscale.patches.qwen_gdn.decode_kv import fused_recurrent_gated_delta_rule_fwd as owned


def elapsed(graph):
    start, end = (torch.npu.Event(enable_timing=True) for _ in range(2))
    start.record()
    for _ in range(30):
        graph.replay()
    end.record(); end.synchronize()
    return start.elapsed_time(end) / 30


def main():
    assert enable_custom_op()
    torch.npu.set_device(0)
    torch.manual_seed(97)
    rows = []
    with torch.inference_mode():
        for requests in (1, 4, 8):
            for width in (1, 3):
                tokens = requests * width
                seed = torch.randn(requests*3, 24, 128, 128, device='npu') * .01
                state = seed.clone()
                native_state = seed.transpose(-1,-2).contiguous()
                q = torch.nn.functional.normalize(torch.randn(1,tokens,8,128,device='npu'),dim=-1).bfloat16()
                k = torch.nn.functional.normalize(torch.randn_like(q.float()),dim=-1).bfloat16()
                v = (torch.randn(1,tokens,24,128,device='npu')*.1).bfloat16()
                g = -torch.rand(1,tokens,24,device='npu')*.1
                beta = torch.rand_like(g).bfloat16()
                cu = torch.arange(requests+1,dtype=torch.int32,device='npu')*width
                actual = torch.tensor([0]+[width]*requests,dtype=torch.int32,device='npu')
                slots = torch.arange(requests*3,dtype=torch.int32,device='npu').reshape(requests,3)
                accepted = torch.full((requests,),3,dtype=torch.int32,device='npu') if width == 3 else None
                if width == 1:
                    slots = slots[:, 0].contiguous()
                # Native ABI requires acceptance <= current length. Width1 here
                # is the non-speculative regression/control, not the width3->1
                # continuation already checked against the CPU oracle.
                def candidate():
                    return owned(q,k,v,g,beta,128**-.5,state,cu_seqlens=cu,ssm_state_indices=slots,num_accepted_tokens=accepted)[0]
                def native():
                    return torch.ops._C_ascend.npu_recurrent_gated_delta_rule(
                        query=q[0],key=k[0],value=v[0],g=g[0],beta=beta[0],state=native_state,
                        scale=128**-.5,actual_seq_lengths=actual,
                        ssm_state_indices=slots.flatten(),num_accepted_tokens=accepted).unsqueeze(0)
                graphs, outputs = {}, {}
                for name, fn in [('owned',candidate),('native',native)]:
                    fn(); torch.npu.synchronize()
                    graph = torch.npu.NPUGraph()
                    with torch.npu.graph(graph):
                        outputs[name] = fn()
                    graphs[name] = graph
                state.copy_(seed); native_state.copy_(seed.transpose(-1,-2))
                for graph in graphs.values():
                    graph.replay()
                torch.npu.synchronize()
                # Cross-layout comparison is outside capture/timing only.
                torch.testing.assert_close(outputs['owned'],outputs['native'],rtol=.03,atol=1e-3)
                torch.testing.assert_close(state,native_state.transpose(-1,-2),rtol=.03,atol=1e-3)
                times = {name:[] for name in graphs}
                for name in ('native','owned','owned','native'):
                    times[name].append(elapsed(graphs[name]))
                row = dict(requests=requests,query_width=width,milliseconds=times)
                rows.append(row); print(json.dumps(row),flush=True)
        Path(os.environ['CAPSULE'],'receipt.json').write_text(json.dumps(dict(passed=True,cases=rows),indent=2))


if __name__ == '__main__':
    main()
