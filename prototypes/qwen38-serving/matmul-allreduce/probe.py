"""Real TP2 row-parallel BF16 GEMM: PG split vs native MatmulAllReduce.

No model load or runtime modifications. Independent graph banks, changed inputs,
FP32 reference and matched graph replay timing. Kernel timing is not service gain.
"""
import datetime
import json
import os
from pathlib import Path
import statistics
import time

import torch
import torch_npu
import torch.distributed as dist

rank = int(os.environ['LOCAL_RANK'])
torch.npu.set_device(rank)
dist.init_process_group('gloo', timeout=datetime.timedelta(seconds=120))
pg = dist.new_group(backend='hccl', timeout=datetime.timedelta(seconds=120))
# Establish the communicator with a real collective before MC2 resource allocation.
bootstrap = torch.ones(1, device='npu')
dist.all_reduce(bootstrap, group=pg)
torch.npu.synchronize()
dist.barrier()
hcom = pg._get_backend(torch.device('npu')).get_hccl_comm_name(rank)
mc2_pg = None
if os.environ.get('PROBE_SEPARATE_MC2') == '1':
    mc2_pg = dist.new_group(backend='hccl', timeout=datetime.timedelta(seconds=120))
    bootstrap_mc2 = torch.ones(1, device='npu')
    dist.all_reduce(bootstrap_mc2, group=mc2_pg)
    torch.npu.synchronize()
    dist.barrier()
    hcom = mc2_pg._get_backend(torch.device('npu')).get_hccl_comm_name(rank)
root = Path(os.environ['CAPSULE'])
arm_name = os.environ['PROBE_ARM']
assert arm_name in ('split', 'fused', 'both')
arms = ('split', 'fused') if arm_name == 'both' else (arm_name,)
copies = int(os.environ.get('PROBE_COPIES', '1'))
assert copies in (1, 2, 16)
raw = None
if os.environ.get('PROBE_RAW_SPLIT') == '1':
    from vllm_ascend.distributed.device_communicators.pyhccl import PyHcclCommunicator
    from vllm_ascend.distributed.device_communicators.pyhccl_wrapper import buffer_type, aclrtStream_t, hcclDataTypeEnum, hcclRedOpTypeEnum
    raw = PyHcclCommunicator(dist.group.WORLD, torch.device(f'npu:{rank}'))
    assert raw.available and not raw.disabled
receipt = dict(separate_mc2=mc2_pg is not None, copies_per_graph=copies, timing_unit="us per projection including communication and consumer", outplace=os.environ.get('PROBE_OUTPLACE') == '1', generations=int(os.environ.get('PROBE_GENERATIONS', '4')), raw_same_stream=raw is not None, rank=rank, status='RUNNING', cases=[], scope='TP2 BF16 ND, transposed weight view; CPU/Gloo FP32 oracle and correctness phase fences; two independent graph banks. Arm/group/copy/generation settings are explicit receipt fields. Timings normalized per projection; not end-to-end speedup.')

def save():
    (root / f'receipt-rank{rank}.json').write_text(json.dumps(receipt, indent=2))

def sync():
    torch.npu.synchronize()
    dist.barrier()

try:
    with torch.inference_mode():
        for k in (3072, 8704):
            torch.manual_seed(419 + rank + k)
            weight = (torch.randn((5120,k), device='npu', dtype=torch.float32) / k**.5).to(torch.bfloat16)
            for n in (1, 1024, 1536):
                case = dict(tokens=n, k=k, checks=[], timings=[])
                print('BEGIN', rank, n, k, arm_name, flush=True)
                receipt['cases'].append(case); save()
                banks = {}
                for arm in arms:
                    banks[arm] = []
                    for bank in (0, 1):
                        x = torch.randn((n,k), device='npu', dtype=torch.bfloat16)
                        def body():
                            if arm == 'split':
                                y = torch.nn.functional.linear(x, weight)
                                pre = y.clone() if os.environ.get('PROBE_TAP') == '1' else None
                                if raw is None:
                                    dist.all_reduce(y, group=pg)
                                elif os.environ.get('PROBE_OUTPLACE') == '1':
                                    y = raw.all_reduce(y, stream=torch.npu.current_stream())
                                else:
                                    raw.hccl.hcclAllReduce(buffer_type(y.data_ptr()), buffer_type(y.data_ptr()), y.numel(), hcclDataTypeEnum.from_torch(y.dtype), hcclRedOpTypeEnum.from_torch(dist.ReduceOp.SUM), raw.comm, aclrtStream_t(torch.npu.current_stream().npu_stream))
                            else:
                                y = torch_npu.npu_mm_all_reduce_base(x, weight.T, hcom, reduce_op='sum')
                                pre = None
                            return pre, y, y + .125
                        for _ in range(3):
                            for _ in range(copies): body()
                        print('CAPTURE', rank, n, k, arm, bank, flush=True)
                        sync()
                        graph = torch.npu.NPUGraph()
                        with torch.npu.graph(graph):
                            outputs = [body() for _ in range(copies)]
                        sync()
                        banks[arm].append(dict(graph=graph, x=x, outputs=outputs))
                for generation in range(int(os.environ.get('PROBE_GENERATIONS', '4'))):
                    b = generation % 2
                    # Distinct rank data and new generations; no repeated constant oracle.
                    torch.manual_seed(1909 + 13*generation + rank)
                    value = torch.randn((n,k), device='npu', dtype=torch.float32).to(torch.bfloat16)
                    local_reference = value.float() @ weight.float().T
                    reference = local_reference.cpu()
                    dist.all_reduce(reference)
                    reference = (reference + .125).to('npu')
                    sync()
                    for arm in arms:
                        sync()
                        item = banks[arm][b]; item['x'].copy_(value)
                        if generation % 2 == rank: time.sleep(.003)
                        item['graph'].replay(); torch.npu.synchronize()
                        for copy, (pre, intermediate, output) in enumerate(item["outputs"]):
                            actual = output.float()
                            error = (actual-reference).abs()
                            valid = bool(torch.isfinite(actual).all()) and bool(torch.allclose(actual, reference, atol=.03, rtol=.03))
                            case['checks'].append(dict(arm=arm, bank=b, generation=generation, copy=copy, passed=valid,
                                                       max_abs=float(error.max()), rmse=float((error**2).mean().sqrt())))
                            if pre is not None:
                                pre_error = pre.float() - local_reference
                                case['checks'][-1].update(pre_rmse=float(pre_error.square().mean().sqrt()), pre_max_abs=float(pre_error.abs().max()), actual_local_rmse=float((actual-.125-local_reference).square().mean().sqrt()))
                            assert valid, (rank,n,k,case['checks'][-1])
                for arm in (*arms, *reversed(arms)):
                    sync()
                    for i in range(4): banks[arm][i%2]['graph'].replay()
                    sync(); samples=[]
                    for i in range(20):
                        start,end=torch.npu.Event(enable_timing=True),torch.npu.Event(enable_timing=True)
                        start.record(); banks[arm][i%2]['graph'].replay(); end.record(); end.synchronize()
                        samples.append(start.elapsed_time(end)*1000/copies)
                    case['timings'].append(dict(arm=arm, median_us=statistics.median(samples),samples_us=samples))
                print(json.dumps(dict(rank=rank,n=n,k=k,timings=[(v['arm'],v['median_us']) for v in case['timings']])),flush=True)
                save(); del banks; sync()
        receipt['status']='PASS'
except BaseException as exc:
    receipt.update(status='FAIL',error=f'{type(exc).__name__}: {exc}')
    raise
finally:
    save()
if raw is not None:
    raw.hccl.hcclCommDestroy(raw.comm)
if mc2_pg is not None:
    dist.destroy_process_group(mc2_pg)
dist.destroy_process_group(pg)
dist.destroy_process_group()
