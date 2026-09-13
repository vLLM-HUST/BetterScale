"""Read-only device audit of native input preparation's host projection.

Uses LiveInfer runtime/shadow.py's TorchDispatch/FakeTensor separation, not
its numerical shadow oracle. No production scheduling or ingress is replaced.
Fake fallback is disabled, but direct Triton launch bypasses Torch dispatch.
The reviewed slot-mapping device action is therefore explicitly excluded from
the host projection; unknown direct launchers must not be treated as safe.
"""
import json
import os
from pathlib import Path
import torch
from torch._subclasses.fake_tensor import FakeTensorMode, is_fake
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten, tree_map
from cross_step import stable_verification


class HostProjection(TorchDispatchMode):
    def __init__(self, device_type='npu'):
        super().__init__()
        self.device_type=device_type
        self.fake=FakeTensorMode(allow_fallback_kernels=False)
        self.operations={}
        self.ingress=[]
        self.readbacks=[]

    def convert(self, x):
        return self.fake.from_tensor(x) if isinstance(x,torch.Tensor) and not is_fake(x) else x

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs={} if kwargs is None else kwargs
        tensors=[x for x in tree_flatten((args,kwargs))[0] if isinstance(x,torch.Tensor)]
        destination=kwargs.get('device')
        target=None if destination is None else torch.device(destination).type
        touches=target==self.device_type or any(x.device.type==self.device_type or is_fake(x) for x in tensors)
        if not touches:return func(*args,**kwargs)
        name=str(func)
        self.operations[name]=self.operations.get(name,0)+1
        source=None
        if (func._schema.name=='aten::_to_copy' or (func._schema.name=='aten::to' and func._schema.overload_name=='dtype_layout')) and args and args[0].device.type=='cpu' and target==self.device_type:
            source=args[0]
        elif func._schema.name=='aten::copy_' and len(args)>1 and args[0].device.type==self.device_type and args[1].device.type=='cpu':
            source=args[1]
        if source is not None:
            self.ingress.append(dict(operation=name,shape=list(source.shape),dtype=str(source.dtype),bytes=source.numel()*source.element_size()))
        if target=='cpu' and any(x.device.type==self.device_type for x in tensors):
            self.readbacks.append(name)
        with self.fake:
            return func(*tree_map(self.convert,args),**tree_map(self.convert,kwargs))


def install(worker):
    r=worker.model_runner
    assert r.use_compress and r.use_async_spec_decode and not r.use_dcp
    original=r._prepare_inputs
    done=False
    def audited(schedule,counts):
        nonlocal done
        result=original(schedule,counts)
        if done or not stable_verification(r,schedule,counts,r.max_num_reqs):return result
        done=True
        torch.npu.synchronize()
        # Projection is not allowed to publish FakeTensor attributes into the
        # running donor. Existing device tensors themselves are never mutated.
        runner_fields=dict(r.__dict__);batch_fields=dict(r.input_batch.__dict__)
        device_state=[r.num_computed_tokens,r.input_ids.gpu,r.positions,r.seq_lens]
        device_state += [t.slot_mapping.gpu for t in r.input_batch.block_table.block_tables]
        before=[x.clone() for x in device_state]
        projection=HostProjection()
        receipt=dict(status='RUNNING',scope='native _prepare_inputs host projection with explicit slot-mapping device boundary',production_path_changed=False)
        tables=r.input_batch.block_table
        mapping=tables.compute_slot_mapping
        device_actions=[]
        def device_mapping(num_reqs,query_start_loc,positions):
            # This writes only the persistent device slot map. No host result
            # exists. Do not launch Triton with fake pointers in a host shadow.
            device_actions.append(dict(action='compute_slot_mapping',requests=num_reqs,tokens=positions.shape[0]))
        tables.compute_slot_mapping=device_mapping
        try:
            with projection:original(schedule,counts)
            receipt['status']='PASS'
        except Exception as error:
            receipt.update(status='BOUNDARY',error=f'{type(error).__name__}: {error}')
        finally:
            tables.compute_slot_mapping=mapping
            r.__dict__.clear();r.__dict__.update(runner_fields)
            r.input_batch.__dict__.clear();r.input_batch.__dict__.update(batch_fields)
            root=Path(os.environ['DONOR_DP_OUTPUT'])
            receipt.update(operations=projection.operations,ingress=projection.ingress,readbacks=projection.readbacks,device_actions=device_actions)
            (root/f'producer-shadow-rank{worker.rank}.json').write_text(json.dumps(receipt,indent=2))
            torch.npu.synchronize()
            unchanged=all(torch.equal(x,y) for x,y in zip(device_state,before))
            receipt.update(device_preparation_state_unchanged=unchanged,operations=projection.operations,ingress=projection.ingress,readbacks=projection.readbacks)
            root=Path(os.environ['DONOR_DP_OUTPUT'])
            (root/f'producer-shadow-rank{worker.rank}.json').write_text(json.dumps(receipt,indent=2))
            assert unchanged, 'Host projection mutated device preparation State'
        return result
    r._prepare_inputs=audited
    return {'scope':'one stable preparation host-projection audit'}
