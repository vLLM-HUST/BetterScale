"""CPU contracts for configurable execution rows and resident State capacity."""
import ast
import os
from pathlib import Path
from types import SimpleNamespace as S
from unittest.mock import patch

import numpy as np
import pytest
import torch

from betterscale.models.qwen35 import capture_sizes, validate, STATE_SCHEDULER
from betterscale.models.qwen35.capacity import seat_counts
from betterscale.models.qwen35.count_policy import spec_capacities, capture_lengths
from betterscale.models.qwen35.host_metadata import HostMetadata
from betterscale.models.qwen35.draft_fia import compact_padding
from betterscale.models.qwen35.state_backend import fixed_state_bytes
from betterscale.patches.qwen_gdn.metadata import Metadata
from test_qwen35_baseline import config


@pytest.mark.parametrize('execution,resident', [(16,20), (32,36), (36,36)])
def test_capacity_admission(execution, resident):
    c = config()
    c.scheduler_config.max_num_seqs = execution
    c.scheduler_config.scheduler_cls = STATE_SCHEDULER
    c.additional_config = dict(using_live_runtime=True, state_resident_seats=resident)
    c.compilation_config.cudagraph_capture_sizes = capture_sizes(execution)
    assert seat_counts(c) == (execution,resident)
    assert validate(c) == 'qwen35-live-state'
    assert spec_capacities(execution)[-1] >= execution * 3
    assert not set(spec_capacities(execution)) & {16,32,64,128,256,512,1024,1536,2048,4096}
    assert capture_lengths([4]*execution, requests=execution) == (3,)*execution
    c.additional_config['state_resident_seats'] = execution - 1
    with pytest.raises(ValueError): seat_counts(c)


def test_resident_budget_not_execution_times_context():
    from betterscale.live.llm.qwen35.state import Geometry
    geometry = Geometry(('linear_attention','full_attention'), 1,4,1,1,2,2,4,4)
    c = S(scheduler_config=S(max_num_seqs=16),additional_config={})
    with patch('betterscale.models.qwen35.state_backend.geometry', return_value=geometry):
        small = fixed_state_bytes(c)
        c.scheduler_config.max_num_seqs=32
        large = fixed_state_bytes(c)
        assert large * 20 == small * 36
        c.scheduler_config.max_num_seqs=36
        c.additional_config['state_resident_seats']=36
        assert fixed_state_bytes(c) == large


@pytest.mark.parametrize('requests', [16,32,36])
def test_metadata_rows_and_empty_sentinel(requests):
    # Execute the real constructor without importing NPU arithmetic leaves.
    from betterscale.models import qwen35
    source = Path(qwen35.__file__).with_name('mixed_core.py')
    node = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.ClassDef))
    init = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name=='__init__')
    cls = ast.ClassDef(name='MixedCore',bases=[],keywords=[],body=[init],decorator_list=[])
    namespace=dict(torch=torch,os=os,WIDTH=3,Metadata=Metadata,SimpleNamespace=S)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls],type_ignores=[])),str(source),'exec'),namespace)
    with patch.dict('sys.modules', {'betterscale.patches.qwen_gdn.runtime': S(Kernels=lambda *a,**kw:S())}), patch.dict(os.environ, {'BETTERSCALE_GDN_LIBRARY':'cpu-fixture'}):
        core = namespace['MixedCore'](4096,'cpu',requests=requests)
    core.decode=False
    core.verify_ids=torch.zeros(requests+1,dtype=torch.int64)
    host=HostMetadata(core)
    lengths=[4]*requests
    roles=[i%2 == 1 for i in range(requests)]
    lengths=[3 if role else n for role,n in zip(roles,lengths)]
    slots=np.arange(requests*3).reshape(requests,3)
    host.prepare(lengths,roles,slots,np.ones(requests,dtype=bool))
    assert core.cu.shape == (requests+2,)
    assert core.verify_map.numel()==requests*3
    assert core.verify_conv[requests-1,0] == slots[-1,0]
    assert core.verify_conv[requests,0] == -1
    for indices in core.prefill.indices.values():
        assert indices[-1,0] == requests
    assert int(core.cu[-1]) == sum(lengths)


def test_draft_compaction_preserves_rows_above_sixteen():
    for requests in (16,32,36):
        m=S(actual_seq_lengths_q=list(range(1,129)),seq_lens_list=[7]*requests+[0]*(128-requests),
            seq_lens=torch.arange(128),_mtp_device_seq_lens=torch.arange(128))
        compact=compact_padding(m,requests)
        assert compact.actual_seq_lengths_q==list(range(1,requests+1))+[128]
        assert compact.seq_lens_list==[7]*requests+[0]
        assert len(compact._mtp_device_seq_lens)==requests
        m.seq_lens_list[requests]=1
        with pytest.raises(ValueError):compact_padding(m,requests)
