"""Check mixed chunk + speculative direct-pool composition in two FULL graphs."""
import json
import os
from pathlib import Path

import torch
import torch_npu
from vllm_ascend.utils import enable_custom_op
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

from mixed_core import MixedCore
from count_policy import WIDTH


def main():
    assert enable_custom_op()
    torch.npu.set_device(0)
    init_device_properties_triton()
    torch.set_num_threads(4)
    torch.manual_seed(83)
    pure = os.environ.get('PURE_VERIFY') == '1'
    capacity = 24 if pure else 64
    slots = torch.randperm(4*WIDTH).reshape(4,WIDTH).tolist()
    seed = torch.randn(4*WIDTH+1, 24, 128, 128) * .01
    conv_seed = (torch.randn(4*WIDTH+1, WIDTH+2, 5120) * .1).bfloat16()
    weight_cpu = (torch.randn(4, 5120) * .2).bfloat16()
    log_cpu = torch.randn(24) * .1
    bias_cpu = torch.randn(24) * .1
    state, conv = seed.npu(), conv_seed.npu()
    weight, log, bias = weight_cpu.npu(), log_cpu.npu(), bias_cpu.npu()
    x = torch.zeros(capacity, 5120, dtype=torch.bfloat16, device='npu')
    a = torch.zeros(capacity, 24, dtype=torch.bfloat16, device='npu')
    b = torch.zeros_like(a)
    if os.environ.get('USE_PUBLICATION') == '1':
        from types import SimpleNamespace as NS
        import numpy as np
        from service_metadata import MTPFrame

        class PublishedCore:
            def __init__(self):
                self.frame = MTPFrame(capacity, {0: None}, 'npu', torch.npu.Stream())
            def prepare(self, lengths, roles, table, accepted, initial):
                self.frame.acquire()
                seq = torch.tensor([n + (128 if warm else 0) for n,warm in zip(lengths,initial)])
                m = NS(seq_lens_cpu=seq)
                builder = NS(vllm_config=NS(cache_config=NS(mamba_cache_mode='none')))
                self.frame.fill_mtp(0,m,lengths,np.array(table),builder,
                    torch.tensor(accepted,dtype=torch.int32,device='npu'),
                    torch.tensor([0 if role else -1 for role in roles],dtype=torch.int32))
                self.frame.publish()
            def __call__(self,*args):
                return self.frame.metas[0](*args)
        cores = [PublishedCore() for _ in range(2)]
    else:
        cores = [MixedCore(capacity) for _ in range(2)]
    cases = [
        ([17, 3, 2, 1], [False, True, True, True], [1, 3, 1, 2]),
        ([3, 19, 1, 2], [True, False, True, True], [2, 1, 3, 1]),
        ([1, 1, 31, 3], [True, False, False, True], [3, 1, 1, 2]),
        ([2, 5, 3, 17], [True, False, True, False], [1, 1, 3, 1]),
        ([3, 1, 2, 57], [True, True, True, False], [3, 2, 1, 1]),
        ([61, 1, 1, 1], [False, True, True, True], [1, 3, 2, 1]),
    ]
    if pure:
        cases = [([3,3,3,3],[True]*4,[1,2,3,1]),
                 ([1,2,3,1],[True]*4,[3,1,2,3]),
                 ([2,1,1,3],[True]*4,[2,3,1,2]),
                 ([1,1,1,1],[True]*4,[3,3,3,3])]
    expanded = []
    for lengths, roles, counts in cases:
        lengths = [((WIDTH if n==3 else min(n,WIDTH)) if role else n) for n,role in zip(lengths,roles)]
        if sum(lengths)>capacity:
            i = max((i for i,role in enumerate(roles) if not role),key=lambda i:lengths[i])
            lengths[i] -= sum(lengths)-capacity
        counts = [WIDTH if n==3 else min(n,WIDTH) for n in counts]
        expanded.append((lengths,roles,counts))
    cases = expanded
    graphs, results = [], []
    with torch.inference_mode():
        for core in cores:
            core.prepare(*cases[0][:2], slots, cases[0][2], [True]*4)
            core(x, a, b, weight, log, bias, conv, state)
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                result = core(x, a, b, weight, log, bias, conv, state)
            graphs.append(graph); results.append(result)
        rows = []
        for wave, (lengths, speculative, accepted) in enumerate(cases):
            initial = [spec or (wave % 2 == 0) for spec in speculative]
            cores[wave % 2].prepare(lengths, speculative, slots, accepted, initial)
            cpu_x = (torch.randn(capacity, 5120) * .1).bfloat16()
            cpu_a = (torch.randn(capacity, 24) * .1).bfloat16()
            cpu_b = (torch.randn(capacity, 24) * .1).bfloat16()
            x.copy_(cpu_x); a.copy_(cpu_a); b.copy_(cpu_b)
            state.copy_(seed); conv.copy_(conv_seed)
            graphs[wave % 2].replay()
            if hasattr(cores[wave % 2], 'frame'):
                cores[wave % 2].frame.release()
            torch.npu.synchronize()
            actual = results[wave % 2].cpu()[0]
            expected_state, expected_conv = seed.clone(), conv_seed.clone()
            expected_output = []
            start = 0
            for i, (length, spec, count, warm) in enumerate(zip(lengths, speculative, accepted, initial)):
                pool_slots = slots[i]
                offset = count - 1 if spec else 0
                history = (conv_seed[pool_slots[0], offset:offset+3].clone() if warm else torch.zeros(3, 5120).bfloat16())
                old_history = history.clone()
                h = seed[pool_slots[offset]].clone() if warm else torch.zeros(24, 128, 128)
                for t in range(length):
                    index = start + t
                    window = torch.cat([history, cpu_x[index:index+1]])
                    y = torch.nn.functional.silu((window.float()*weight_cpu.float()).sum(0)).bfloat16()
                    history = window[1:]
                    q, k, v = [part.reshape(heads, 128).float() for part, heads in zip(y.split([1024,1024,3072]), [8,8,24])]
                    q = (q/(q.square().sum(-1,keepdim=True)+1e-6).sqrt()).bfloat16().float().repeat_interleave(3,0) * 128**-.5
                    k = (k/(k.square().sum(-1,keepdim=True)+1e-6).sqrt()).bfloat16().float().repeat_interleave(3,0)
                    g = -log_cpu.exp()*torch.nn.functional.softplus(cpu_a[index].float()+bias_cpu)
                    beta = cpu_b[index].float().sigmoid().bfloat16().float()
                    h *= g.exp()[:,None,None]
                    h += k[:,:,None]*((v-(h*k[:,:,None]).sum(1))*beta[:,None])[:,None,:]
                    expected_output.append((h*q[:,:,None]).sum(1).bfloat16())
                    if spec:
                        expected_state[pool_slots[t]] = h
                if spec:
                    expected_conv[pool_slots[0], :2] = old_history[1:]
                    expected_conv[pool_slots[0], 2:2+length] = cpu_x[start:start+length]
                else:
                    expected_state[pool_slots[0]] = h
                    expected_conv[pool_slots[0], :3] = history
                start += length
            expected_output = torch.stack(expected_output)
            actual_state = state.cpu()
            torch.testing.assert_close(actual[:start], expected_output, rtol=.03, atol=2e-4)
            torch.testing.assert_close(actual_state, expected_state, rtol=.03, atol=2e-4)
            torch.testing.assert_close(conv.cpu(), expected_conv, rtol=0, atol=0)
            row = dict(wave=wave, lengths=lengths, speculative=speculative,
                       output_max=float((actual[:start].float()-expected_output.float()).abs().max()),
                       state_max=float((actual_state-expected_state).abs().max()))
            rows.append(row); print(json.dumps(row), flush=True)
        receipt = dict(passed=True, graphs=2, cases=rows, scope='mixed operator composition only; no service qualification')
        Path(os.environ['CAPSULE'], 'receipt.json').write_text(json.dumps(receipt, indent=2))
        print('mixed state composition PASS', flush=True)


if __name__ == '__main__':
    main()
