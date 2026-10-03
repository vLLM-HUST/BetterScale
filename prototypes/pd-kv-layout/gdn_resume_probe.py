"""Owned TP2 GDN recurrence and quiescent host-checkpoint resume on one NPU.

Independent FP32 CPU recurrence; no conv, Store, MTP or whole-model claim.
Requires explicit hardware authority and an idle visible device.
"""
import argparse
import importlib.util
import json
from pathlib import Path

import torch
import torch_npu

def check(output_dir, checkpoint_transport=None):
    output_dir.mkdir(exist_ok=False, parents=True)
    torch.set_num_threads(4)
    torch.manual_seed(173)
    torch.npu.set_device(0)
    assert torch.npu.device_count() == 1
    source = Path(__file__).resolve().parents[2] / 'src/betterscale/models/qwen35/decode_kv.py'
    spec = importlib.util.spec_from_file_location('owned_decode', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    run = module.fused_recurrent_gated_delta_rule_fwd
    # Three active rows plus retired/padding row; three candidates each, one guard.
    state_cpu = torch.randn(13, 16, 128, 128) * .01
    state = state_cpu.npu()
    slots_cpu = torch.tensor([[0,1,2], [3,4,5], [6,7,8], [-1,-1,-1]])
    slots = slots_cpu.npu()
    accepted = torch.ones(4, dtype=torch.int32).npu()
    cu = torch.tensor([0,3,6,9,12], dtype=torch.int32).npu()
    q = torch.zeros(1,12,8,128, dtype=torch.bfloat16).npu()
    k = torch.zeros_like(q)
    v = torch.zeros(1,12,16,128, dtype=torch.bfloat16).npu()
    g = torch.zeros(1,12,16).npu()
    beta = torch.zeros_like(g)
    def call():
        return run(q, k, v, g, beta, 128**-.5, state,
                   cu_seqlens=cu, ssm_state_indices=slots,
                   num_accepted_tokens=accepted)
    call()  # Compile before graph capture; restore the initial bank afterwards.
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output, final = call()
    state.copy_(state_cpu)
    assert final.data_ptr() == state.data_ptr()
    receipts = []
    for wave in range(24):
        counts = torch.tensor([(wave+i)%3+1 for i in range(4)], dtype=torch.int32)
        if wave == 12:
            # Writer retired by the preceding synchronize. Export only accepted
            # target State, not rejected speculative candidates; normalize to slot0.
            selected = slots_cpu[0, counts[0]-1].item()
            checkpoint = state[selected].cpu().clone()
            torch.testing.assert_close(checkpoint, state_cpu[selected], rtol=.002, atol=1e-4)
            if checkpoint_transport is not None:
                transferred = checkpoint_transport(checkpoint)
                torch.testing.assert_close(transferred, checkpoint, rtol=0, atol=0)
                checkpoint = transferred
            restored = torch.zeros(3,16,128,128)
            restored[0].copy_(checkpoint)
            state[9:12].copy_(restored)
            torch.npu.synchronize()
            torch.testing.assert_close(state[9].cpu(), checkpoint, rtol=0, atol=0)
            # Reference retains its independently calculated State, not candidate
            # values, so subsequent comparisons cannot hide pre-checkpoint error.
            state_cpu[9:12].zero_()
            state_cpu[9].copy_(state_cpu[selected])
            slots_cpu[0] = torch.tensor([9,10,11])
            slots.copy_(slots_cpu)
            counts[0] = 1
        qc = (torch.randn(q.shape)*.2).bfloat16()
        kc = torch.nn.functional.normalize(torch.randn(k.shape), dim=-1).bfloat16()
        vc = (torch.randn(v.shape)*.2).bfloat16()
        gc = -torch.rand(g.shape)*.2
        bc = torch.rand(beta.shape)
        for device, host in ((q,qc),(k,kc),(v,vc),(g,gc),(beta,bc),(accepted,counts)):
            device.copy_(host)
        truth = torch.zeros(v.shape)
        for row in range(3):
            s = state_cpu[slots_cpu[row,counts[row]-1]].clone()
            for token in range(3):
                pos = row*3+token
                kk = kc[0,pos].float().repeat_interleave(2,dim=0)
                qq = qc[0,pos].float().repeat_interleave(2,dim=0)*128**-.5
                s *= gc[0,pos].exp()[:,None,None]
                delta = (vc[0,pos].float()-(s*kk[:,:,None]).sum(1))*bc[0,pos,:,None]
                s += kk[:,:,None]*delta[:,None,:]
                truth[0,pos] = (s*qq[:,:,None]).sum(1)
                state_cpu[slots_cpu[row,token]].copy_(s)
        graph.replay()
        torch.npu.synchronize()
        actual = output.cpu().float()
        actual_state = state.cpu()
        torch.testing.assert_close(actual, truth, rtol=.02, atol=.003)
        torch.testing.assert_close(actual_state, state_cpu, rtol=.002, atol=1e-4)
        torch.testing.assert_close(actual[:,9:], torch.zeros_like(actual[:,9:]), rtol=0, atol=0)
        receipts.append(dict(wave=wave, output_max=(actual-truth).abs().max().item(),
                             state_max=(actual_state-state_cpu).abs().max().item()))
        (output_dir/'progress.json').write_text(json.dumps(receipts,indent=2)+'\n')
    (output_dir/'complete.json').write_text(json.dumps(dict(status='passed',
        torch=torch.__version__, torch_npu=torch_npu.__version__, waves=receipts,
        checkpoint_wave=12, checkpoint_bytes=16*128*128*4,
        external_transport=checkpoint_transport is not None,
        scope='One-layer TP2 recurrence, selected target host checkpoint, new slot resume; no conv/model; transport described by caller'),indent=2)+'\n')
    print('PASS: 24 captured waves, accepted candidates 1..3, host checkpoint/new-slot resume, retired row and untouched slots')
    return json.loads((output_dir/'complete.json').read_text())


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    check(a.output)
