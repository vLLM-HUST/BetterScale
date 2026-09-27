"""Bounded native grouped-MLP / State-sized DMA overlap experiment, not serving.

Run only under selected-device admission. Buffers, streams, events and graphs
are warmed before timing. Independent directions never alias host/device data.
"""
import json
import math
import os
from pathlib import Path
import random
import time

import torch
import torch_npu


def main():
    run = Path(os.environ['CAPSULE'])
    torch.npu.set_device(0)
    torch.set_num_threads(4)
    packets = json.loads((run / 'packets.json').read_text())
    assert len(packets) == 90 and sum(p['bytes'] for p in packets) == 118673460
    compute, down, up = (torch.npu.Stream() for _ in range(3))
    # Byte transport retains exact packet lengths/fragmentation, not model dtype math.
    host_src, host_dst, dev_src, dev_dst = [], [], [], []
    for i, p in enumerate(packets):
        h = torch.empty(p['bytes'], dtype=torch.uint8, pin_memory=True)
        h.copy_((torch.arange(p['bytes'], dtype=torch.int64) % 251 + i).to(torch.uint8))
        host_src.append(h)
        host_dst.append(torch.empty_like(h, pin_memory=True).zero_())
        dev_src.append(h.to('npu'))
        dev_dst.append(torch.empty_like(dev_src[-1]).zero_())
    generator = torch.Generator().manual_seed(173)
    x_cpu = torch.randn(192, 2048, generator=generator).bfloat16()
    w1_cpu = (torch.randn(256, 2048, 512, generator=generator) / math.sqrt(2048)).bfloat16()
    w2_cpu = (torch.randn(256, 256, 2048, generator=generator) / 16).bfloat16()
    x, w1, w2 = (t.to('npu') for t in (x_cpu, w1_cpu, w2_cpu))
    # Synthetic routing: one row each for experts 0..191, zero for the remaining64.
    groups = torch.tensor([min(i+1, 192) for i in range(256)], dtype=torch.int64, device='npu')

    def block():
        y = torch_npu.npu_grouped_matmul([x], [w1], split_item=2,
                group_type=0, group_list=groups, group_list_type=0)[0]
        y = torch_npu.npu_swiglu(y)
        return torch_npu.npu_grouped_matmul([y], [w2], split_item=2,
                group_type=0, group_list=groups, group_list_type=0)[0]

    # Independent FP32 CPU oracle with BF16 boundaries matching native operators.
    y = torch.bmm(x_cpu.float().unsqueeze(1), w1_cpu[:192].float()).squeeze(1).bfloat16()
    a, b = y.float().chunk(2, dim=-1)
    mid = (torch.nn.functional.silu(a) * b).bfloat16()
    ref = torch.bmm(mid.float().unsqueeze(1), w2_cpu[:192].float()).squeeze(1).bfloat16()
    with torch.npu.stream(compute):
        for _ in range(3):
            eager = block()
    torch.npu.synchronize()
    torch.testing.assert_close(eager.cpu(), ref, rtol=.04, atol=.04)
    with torch.npu.stream(compute):
        prefix, body = torch.npu.NPUGraph(), torch.npu.NPUGraph()
        with torch.npu.graph(prefix):
            prefix_out = block()
        with torch.npu.graph(body):
            for _ in range(64):
                output = block()
        prefix.replay()
        body.replay()
    torch.npu.synchronize()
    torch.testing.assert_close(output.cpu(), eager.cpu(), rtol=0, atol=0)
    events = {key: torch.npu.Event(enable_timing=True) for key in
              ('start', 'gate', 'end', 'd0', 'd1', 'h0', 'h1')}
    for event in events.values():
        event.record(compute)
    torch.npu.synchronize()
    replays = 1

    def trial(mode, label):
        c = mode.startswith('compute')
        d, h = 'd2h' in mode or 'both' in mode, 'h2d' in mode or 'both' in mode
        torch_npu.npu.mstx.mark(label + ':begin')
        wall = time.perf_counter()
        if c:
            with torch.npu.stream(compute):
                events['start'].record()
                prefix.replay()
                events['gate'].record()
                for _ in range(replays):
                    body.replay()
                events['end'].record()
            # Body is ALREADY queued before this host event wait. No global sync.
            events['gate'].synchronize()
        for enabled, stream, first, last, dest, src in (
            (d, down, 'd0', 'd1', host_dst, dev_src),
            (h, up, 'h0', 'h1', dev_dst, host_src),
        ):
            if enabled:
                torch_npu.npu.mstx.mark(label + ':' + first + ':submit')
                with torch.npu.stream(stream):
                    events[first].record()
                    for dst, source in zip(dest, src, strict=True):
                        dst.copy_(source, non_blocking=True)
                    events[last].record()
        # Completion waits only after both directions have been submitted.
        for enabled, key in ((c, 'end'), (d, 'd1'), (h, 'h1')):
            if enabled:
                events[key].synchronize()
        result = dict(mode=mode, label=label, wall_ms=(time.perf_counter()-wall)*1000)
        for enabled, key, first, last in ((c, 'compute_ms', 'gate', 'end'),
                (d, 'd2h_ms', 'd0', 'd1'), (h, 'h2d_ms', 'h0', 'h1')):
            if enabled:
                result[key] = events[first].elapsed_time(events[last])
        torch_npu.npu.mstx.mark(label + ':end')
        return result

    modes = ['compute', 'd2h', 'h2d', 'both', 'compute_d2h', 'compute_h2d', 'compute_both']
    calibration = trial('compute', 'calibration')
    # Predetermined compute-only calibration target: >=100ms nominal queued work.
    replays = max(1, min(32, math.ceil(100 / calibration['compute_ms'])))
    for mode in modes:
        trial(mode, 'warm:' + mode)
    timings = []
    rng = random.Random(173)
    for round_id in range(7):
        order = modes.copy()
        rng.shuffle(order)
        for mode in order:
            result = trial(mode, f'timing:{round_id}:{mode}')
            result['round'] = round_id
            timings.append(result)
    # Validate every transferred byte and the graph result, outside timing.
    for actual, expected in zip(host_dst, host_src, strict=True):
        assert torch.equal(actual, expected)
    for actual, expected in zip(dev_dst, host_src, strict=True):
        assert torch.equal(actual.cpu(), expected)
    torch.testing.assert_close(output.cpu(), eager.cpu(), rtol=0, atol=0)
    summary = dict(pid=os.getpid(), visible=os.environ.get('ASCEND_RT_VISIBLE_DEVICES'),
                   packets=packets, calibration=calibration, replays=replays,
                   blocks_per_replay=64, correctness=True, timings=timings,
                   torch=torch.__version__, torch_npu=torch_npu.__version__)
    (run / 'timings.json').write_text(json.dumps(summary, indent=2)+'\n')
    print('TIMINGS', json.dumps({k: summary[k] for k in ('pid','visible','replays','correctness')}), flush=True)
    config = torch_npu.profiler._ExperimentalConfig(
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        export_type=torch_npu.profiler.ExportType.Db,
        msprof_tx=True, data_simplification=False)
    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
        record_shapes=False, profile_memory=False, with_stack=False,
        experimental_config=config,
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
            str(run / 'raw'), worker_name='continuous-dma', analyse_flag=False)):
        # One predetermined profile sample per condition, separate from timing.
        for mode in modes:
            trial(mode, 'profile:' + mode)
    print('PROFILE_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
