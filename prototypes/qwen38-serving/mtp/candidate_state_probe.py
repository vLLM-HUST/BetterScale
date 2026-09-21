"""Bounded NPU proof: accepted-prefix conv/GDN continuation in two FULL graphs.

No model weights or serving hooks. Independent CPU recurrence checks every live
output and candidate state, not merely graph/eager agreement. Metadata changes
between replays; state stays in its physical K-V pool throughout execution.
"""
import json
import os
from pathlib import Path

import torch
import torch_npu
from vllm_ascend.utils import enable_custom_op

from betterscale.patches.qwen_gdn.decode_kv import fused_recurrent_gated_delta_rule_fwd


def main():
    assert enable_custom_op()
    torch.npu.set_device(0)
    torch.manual_seed(71)
    torch.set_num_threads(4)
    device = 'npu'
    width = int(os.environ.get('MTP_TOKENS','2')) + 1
    assert 2 <= width <= 5
    requests, capacity, channels = 5, 4*width, 5120
    # Includes slot zero, disjoint permuted rows, and a negative padding row.
    physical = torch.cat((torch.randperm(4*width).reshape(4,width),
                          torch.full((1,width),-1)),0)
    pool_seed = torch.randn(4*width+1, 24, 128, 128) * .01
    conv_seed = (torch.randn(4*width+1, width+2, channels) * .1).bfloat16()
    weight_cpu = (torch.randn(4, channels) * .2).bfloat16()
    pool = pool_seed.to(device)
    conv = conv_seed.to(device)
    weight = weight_cpu.to(device)
    x = torch.empty(capacity, channels, dtype=torch.bfloat16, device=device)
    g = torch.empty(1, capacity, 24, device=device)
    beta = torch.empty_like(g)
    feedback = os.environ.get('DEVICE_FEEDBACK') == '1'
    resident = torch.tensor([width, 1, 2, width, 1], dtype=torch.int32, device=device)
    draft = torch.zeros(5, width-1, dtype=torch.int64, device=device)
    target = torch.zeros(5, width, dtype=torch.int64, device=device)
    active = torch.tensor([True]*4+[False], device=device)
    cursor = torch.zeros(5, dtype=torch.int64, device=device)
    anchor = torch.zeros(5, dtype=torch.int64, device=device)
    banks = []
    for _ in range(2):
        # Non-unit column stride guards the candidate address calculation.
        slots_storage = torch.full((requests, 2*width), -1, dtype=torch.int64, device=device)
        banks.append(dict(ids=torch.arange(5, dtype=torch.int64, device=device), cu=torch.tensor([0, width, 2*width, 3*width, 4*width, 4*width], dtype=torch.int32, device=device),
                          slots=slots_storage[:, ::2],
                          conv_slots=physical[:, :1].int().contiguous().to(device),
                          accepted=torch.ones(requests, dtype=torch.int32, device=device)))
        banks[-1]['slots'].copy_(physical)

    def forward(bank):
        if feedback:
            bank['accepted'].copy_(resident.index_select(0, bank['ids']))
        y = torch.empty_like(x)
        torch.ops._C_ascend.npu_causal_conv1d_custom(
            y, x, weight, conv_state=conv, bias_opt=None,
            query_start_loc_opt=bank['cu'], cache_indices_opt=bank['conv_slots'],
            initial_state_mode_opt=None, num_accepted_tokens_opt=bank['accepted'],
            activation_mode=1, pad_slot_id=-1, run_mode=1)
        q, k, v = (part.reshape(1, capacity, heads, 128).contiguous()
                   for part, heads in zip(y.split([1024, 1024, 3072], -1), [8, 8, 24]))
        out, _ = fused_recurrent_gated_delta_rule_fwd(
            q, k, v, g, beta, 128 ** -.5, pool,
            cu_seqlens=bank['cu'], ssm_state_indices=bank['slots'],
            num_accepted_tokens=bank['accepted'], use_qk_l2norm_in_kernel=True)
        if feedback:
            # Stand-in for the native device verifier; the service must reuse
            # its valid-count tensor rather than install a second sampler.
            committed = (target[:, :width-1] == draft).int().cumprod(1).sum(1).int() + 1
            resident.copy_(torch.where(active, committed, resident))
            cursor.add_(torch.where(active, committed, 0))
            chosen_anchor = target.gather(1, (committed.long()-1)[:, None]).squeeze(1)
            anchor.copy_(torch.where(active, chosen_anchor, anchor))
        return y, out

    x.zero_(); g.fill_(-.1); beta.fill_(.5)
    graphs, outputs = [], []
    with torch.inference_mode():
        for bank in banks:
            forward(bank)
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                result = forward(bank)
            graphs.append(graph); outputs.append(result)
        pool.copy_(pool_seed); conv.copy_(conv_seed)
        resident.copy_(torch.tensor([width, 1, 2, width, 1], dtype=torch.int32))
        cursor.zero_()
        reference_cursor = torch.zeros(5, dtype=torch.int64)
        reference_anchor = torch.zeros(5, dtype=torch.int64)
        carried_accept = [width, 1, 2, width]
        expected_pool, expected_conv = pool_seed.clone(), conv_seed.clone()
        previous_lengths = [width] * 4
        rows = []
        for wave in range(24):
            lengths = ([width]*4, [1, 2, width, 0], [width, 1, 2, 1], [2, width, 1, width])[wave % 4]
            accepted = [1 + (wave + i + 2) % previous_lengths[i] for i in range(4)]
            # Explicitly exercise current T=1 selecting old candidate column 2.
            if wave % 4 == 1:
                accepted[0] = width
            if feedback:
                accepted = carried_accept
            order = list(range(4)) if wave % 2 == 0 else [2, 0, 3, 1]
            ordered_lengths = [lengths[i] for i in order] + [0]
            ordered_accepted = [accepted[i] for i in order] + [1]
            mapping = physical[order + [4]]
            ends = torch.tensor([0] + ordered_lengths).cumsum(0).int()
            cpu_x = (torch.randn(capacity, channels) * .1).bfloat16()
            cpu_g = -torch.rand(1, capacity, 24) * .2
            cpu_beta = torch.rand(1, capacity, 24)
            bank = banks[wave % 2]
            bank['cu'].copy_(ends); bank['slots'].copy_(mapping)
            bank['conv_slots'].copy_(mapping[:, :1].int())
            if not feedback:
                bank['accepted'].copy_(torch.tensor(ordered_accepted, dtype=torch.int32))
            else:
                bank['ids'].copy_(torch.tensor(order + [4]))
                # Prepare token evidence, not a host accepted-count publication.
                next_lengths = [lengths[i] or previous_lengths[i] for i in range(4)]
                next_counts = [1 + (wave + 1 + i + 2) % next_lengths[i] for i in range(4)]
                if (wave + 1) % 4 == 1:
                    next_counts[0] = width
                next_counts = [next_counts[i] if lengths[i] else accepted[i] for i in range(4)]
                carried_accept = next_counts
                next_counts = next_counts + [1]
                active_cpu = torch.tensor([n > 0 for n in lengths] + [False])
                active.copy_(active_cpu)
                draft_cpu = torch.arange(5*(width-1)).reshape(5, width-1) + 100
                target_cpu = torch.full((5, width), -1, dtype=torch.int64)
                for i, count in enumerate(next_counts):
                    target_cpu[i, :count-1] = draft_cpu[i, :count-1]
                draft.copy_(draft_cpu); target.copy_(target_cpu)
            x.copy_(cpu_x); g.copy_(cpu_g); beta.copy_(cpu_beta)
            graphs[wave % 2].replay()
            torch.npu.synchronize()
            actual_y, actual_o = [t.cpu() for t in outputs[wave % 2]]
            if feedback:
                reference_cursor += torch.where(active_cpu, torch.tensor(next_counts), 0)
                torch.testing.assert_close(cursor.cpu(), reference_cursor, rtol=0, atol=0)
                expected_anchor = target_cpu[torch.arange(5), torch.tensor(next_counts)-1]
                reference_anchor = torch.where(active_cpu, expected_anchor, reference_anchor)
                torch.testing.assert_close(anchor.cpu(), reference_anchor, rtol=0, atol=0)
            reference_y, reference_o = [], []
            for row, request in enumerate(order):
                length = lengths[request]
                if length == 0:
                    continue
                start = int(ends[row]); count = accepted[request]
                slots = mapping[row]
                conv_slot = int(slots[0])
                history = expected_conv[conv_slot, count - 1:count + 2].clone()
                initial = history.clone()
                h = expected_pool[int(slots[count - 1])].clone()
                for t in range(length):
                    index = start + t
                    window = torch.cat([history, cpu_x[index:index + 1]])
                    y = torch.nn.functional.silu((window.float() * weight_cpu.float()).sum(0)).bfloat16()
                    reference_y.append(y)
                    history = window[1:]
                    # Check convolution separately below. Feed its observed BF16
                    # rounding into the independent CPU recurrence so a rare
                    # one-ULP SiLU rounding difference cannot masquerade as a
                    # persistent candidate-state protocol error.
                    q, k, v = [p.reshape(heads, 128).float() for p, heads in zip(actual_y[index].split([1024, 1024, 3072]), [8, 8, 24])]
                    q = q / (q.square().sum(-1, keepdim=True) + 1e-6).sqrt()
                    k = k / (k.square().sum(-1, keepdim=True) + 1e-6).sqrt()
                    q = q.repeat_interleave(3, 0) * 128 ** -.5
                    k = k.repeat_interleave(3, 0)
                    h *= cpu_g[0, index].exp()[:, None, None]
                    delta = (v - (h * k[:, :, None]).sum(1)) * cpu_beta[0, index, :, None]
                    h += k[:, :, None] * delta[:, None, :]
                    reference_o.append((h * q[:, :, None]).sum(1).bfloat16())
                    expected_pool[int(slots[t])] = h
                expected_conv[conv_slot, :2] = initial[1:]
                expected_conv[conv_slot, 2:2 + length] = cpu_x[start:start + length]
                previous_lengths[request] = length
            used = sum(lengths)
            expected_y, expected_o = torch.stack(reference_y), torch.stack(reference_o)
            # BF16 convolution permits one rounding unit; propagation is checked
            # against a separate CPU history across all 24 waves, never reset.
            torch.testing.assert_close(actual_y[:used], expected_y, rtol=.02, atol=2e-4)
            torch.testing.assert_close(actual_o[0, :used], expected_o, rtol=.03, atol=3e-4)
            actual_pool = pool.cpu()
            state_errors = (actual_pool - expected_pool).abs().flatten(1).amax(1).tolist()
            print(json.dumps(dict(wave=wave, lengths=lengths, accepted=accepted,
                                  state_errors=state_errors)), flush=True)
            if max(state_errors) > 5e-4:
                torch.save(dict(actual=actual_pool, expected=expected_pool,
                                wave=wave, lengths=lengths, accepted=accepted),
                           Path(os.environ['CAPSULE'], 'state-diagnostic.pt'))
            torch.testing.assert_close(actual_pool, expected_pool, rtol=1e-4, atol=1e-7)
            torch.testing.assert_close(conv.cpu(), expected_conv, rtol=0, atol=0)
            rows.append(dict(wave=wave, lengths=lengths, accepted=accepted,
                             conv_max=float((actual_y[:used].float()-expected_y.float()).abs().max()),
                             output_max=float((actual_o[0, :used].float()-expected_o.float()).abs().max()),
                             state_max=float((actual_pool-expected_pool).abs().max())))
        receipt = dict(passed=True, mtp_tokens=width-1, graphs=2, waves=rows, state_layout='K-V', device_feedback=feedback,
                       scope='operator candidate continuation; not serving/MTP integration')
        Path(os.environ['CAPSULE'], 'receipt.json').write_text(json.dumps(receipt, indent=2))
        print(json.dumps(dict(passed=True, waves=len(rows), device_feedback=feedback,
                              state_max=max(r['state_max'] for r in rows))), flush=True)


if __name__ == '__main__':
    main()
