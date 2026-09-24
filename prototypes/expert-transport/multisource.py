"""A2E1 real-weight leaf: isolate each client, then exercise both resident slots.

One 96-row decode-shaped client and one 4096-row prefill-shaped client.
No real attention/scheduler, so these are FFN interference bounds, not serving.
CPU barriers delimit whole graph replays, never individual expert requests.
"""
import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import time


def worker(role, device, barrier, a):
    with (a.output / f'{role}.log').open('w', buffering=1) as log:
        os.dup2(log.fileno(), 1); os.dup2(log.fileno(), 2)
        os.environ.update(ASCEND_RT_VISIBLE_DEVICES=str(device),
            BETTERSCALE_EXPERT_DRAFT_LAYERS='1', BETTERSCALE_EXPERT_MODEL=a.model,
            BETTERSCALE_EXPERT_PERSISTENT_BUILD=a.build,
            BETTERSCALE_EXPERT_FINE_PACK='0', BETTERSCALE_EXPERT_RETURN_MODE=a.return_mode,
            BETTERSCALE_EXPERT_GRAPH_BATCH='4')
        import torch
        import torch_npu
        torch.npu.set_device(0)
        from betterscale.patches.expert_service.placement import Placement
        placement = Placement('layer', 1)
        if role == 'server':
            from betterscale.patches.expert_service.persistent_server import serve
            result = serve(a.output/'control', placement, 0, 2, layers=(0,))
            (a.output/'server.json').write_text(json.dumps(result, indent=2))
            return
        source = int(role)
        from betterscale.patches.expert_service.persistent_remote import PersistentRemote
        from betterscale.patches.expert_service.checkpoint import Checkpoint, plain_experts
        remote = PersistentRemote(a.output/'control', placement, source)
        rows, repeats = (96, 64) if source == 0 else (4096, 16)
        torch.manual_seed(20260924+source)
        weights = Checkpoint().experts(0)
        x = (torch.randn(rows, 2048, device='npu')*.1).bfloat16()
        ids = (torch.arange(rows*8, device='npu').reshape(rows,8)%256).long()
        probs = torch.full((rows,8), .125, device='npu')
        expected = plain_experts(weights, x, ids, probs)
        remote.prepare_graph_rows(rows)
        remote(0,x,ids,probs); torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            for _ in range(repeats): actual = remote(0,x,ids,probs)
        graph.replay(); torch.npu.synchronize()
        def check():
            error = float(torch.linalg.vector_norm(actual.float()-expected.float()) /
                          torch.linalg.vector_norm(expected.float()).clamp_min(1e-12))
            assert torch.isfinite(actual).all() and error < .02, error
            return error
        initial_error = check()
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        results = []
        for phase in ('source0', 'source1', 'both'):
            active = phase == 'both' or phase == f'source{source}'
            for sample in range(6):
                barrier.wait(timeout=180)
                if active:
                    start.record(); graph.replay(); end.record(); end.synchronize()
                    elapsed = start.elapsed_time(end)*1000/repeats
                barrier.wait(timeout=180)
                if active and sample:
                    results.append(dict(phase=phase, sample=sample, us_per_call=elapsed))
        # Changed contents after all timings, without stale graph constants.
        x.mul_(.5); ids.add_(17).remainder_(256)
        probs.copy_(torch.softmax(torch.randn(rows,8,device='npu'),dim=-1))
        expected = plain_experts(weights,x,ids,probs)
        barrier.wait(timeout=180)
        graph.replay(); torch.npu.synchronize()
        changed_error = check()
        barrier.wait(timeout=180)
        receipt = remote.receipt()
        remote.close()
        (a.output/f'client{source}.json').write_text(json.dumps(dict(status='PASS',
            rows=rows, repeats=repeats, initial_l2=initial_error,
            changed_l2=changed_error, results=results, receipt=receipt),indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--devices',required=True)
    p.add_argument('--build',required=True)
    p.add_argument('--return-mode', choices=('pull','push'), default='pull')
    p.add_argument('--model',default='/workspace/models/Qwen3.5-35B-A3B')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    devices=[int(d) for d in a.devices.split(',')]
    assert len(devices)==len(set(devices))==3
    a.output.mkdir(exist_ok=False); (a.output/'control').mkdir(mode=0o700)
    ctx=mp.get_context('spawn'); barrier=ctx.Barrier(2); jobs=[]
    def launch(role, device):
        child=ctx.Process(target=worker,args=(role,device,barrier,a))
        child.start(); jobs.append(child); return child
    try:
        server=launch('server',devices[2]); deadline=time.monotonic()+300
        while not (a.output/'control/e0.sock').exists():
            assert server.is_alive(), 'server startup failed'
            assert time.monotonic()<deadline, 'server startup timeout'
            time.sleep(.5)
        for source in range(2): launch(str(source),devices[source])
        deadline=time.monotonic()+900
        while any(job.is_alive() for job in jobs):
            assert all(job.exitcode in (None,0) for job in jobs), 'worker failed'
            assert time.monotonic()<deadline, 'probe timeout'
            time.sleep(1)
        assert all(job.exitcode == 0 for job in jobs), 'nonzero worker exit'
        server=json.loads((a.output/'server.json').read_text())
        for source in range(2):
            client=json.loads((a.output/f'client{source}.json').read_text())
            assert client['receipt']['peer_generations']['0']==server['completed'][str(source)]
        (a.output/'result.json').write_text(json.dumps(dict(status='PASS',
            devices=devices, return_mode=a.return_mode, scope='real-weight A2E1 leaf, not attention or serving'),indent=2))
    finally:
        for job in jobs:
            if job.is_alive(): job.terminate()
        for job in jobs:
            job.join(15)
            if job.is_alive(): job.kill(); job.join()


if __name__=='__main__': main()
