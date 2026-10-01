"""Bounded single-device pinned-host staging; no model/Store/overlap claims."""
import argparse
import json
import statistics
import time
import torch
import torch_npu


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    p.add_argument('--device', type=int, default=0)
    args = p.parse_args()
    torch.npu.set_device(args.device)
    stream = torch.npu.Stream(device=args.device)
    receipt = {'scope': 'single NPU pinned-host byte copies; no Store or model',
               'torch': torch.__version__, 'torch_npu': torch_npu.__version__,
               'device': args.device, 'results': []}
    for mib in (20, 80, 320):
        size = mib * 1024**2
        source = torch.empty(size, dtype=torch.uint8, pin_memory=True)
        source.numpy()[:] = 173
        source.numpy()[::4096] = 29
        target = torch.empty_like(source, pin_memory=True)
        device = torch.empty(size, dtype=torch.uint8, device=f'npu:{args.device}')
        times = {'h2d_ms': [], 'd2h_ms': []}
        for repeat in range(4):
            for name, dst, src in [('h2d_ms', device, source), ('d2h_ms', target, device)]:
                start = time.perf_counter()
                with torch.npu.stream(stream):
                    dst.copy_(src, non_blocking=True)
                    done = torch.npu.Event()
                    done.record(stream)
                done.synchronize()  # Do not consume/reuse host bytes before completion.
                if repeat:
                    times[name].append((time.perf_counter()-start)*1000)
            assert torch.equal(source, target), 'byte mismatch'
        receipt['results'].append({'mib': mib, 'exact': True,
                                  **{k: statistics.median(v) for k, v in times.items()}})
        del source, target, device
    torch.npu.synchronize()
    torch.npu.empty_cache()
    with open(args.output, 'w') as f:
        json.dump(receipt, f, indent=2)
        f.write('\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
