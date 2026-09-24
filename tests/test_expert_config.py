"""CPU contracts for topology, physical draft catalog and pre-load lifecycle."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace as S
import unittest
from unittest.mock import patch
from betterscale.patches.expert_service.config import ServiceConfig
from betterscale.patches.expert_service.build import emit


class ExpertConfig(unittest.TestCase):
    def test_explicit_opt_in_and_mtp(self):
        values = dict(experimental=True, control='/task/control', build='/task/build', owners=2, sources=6, source=5)
        config = S(additional_config={'betterscale_experts': values}, speculative_config=None)
        self.assertEqual(ServiceConfig.from_vllm(config).draft_layers, 0)
        config.speculative_config = S(method='mtp', num_speculative_tokens=2)
        service = ServiceConfig.from_vllm(config)
        self.assertEqual(service.draft_layers, 1)
        for key, bad in [('source', 6), ('owners', 3), ('sources', 7), ('experimental', False), ('source', True)]:
            with self.subTest(key=key, bad=bad):
                with patch.dict(values, {key:bad}), self.assertRaises(ValueError):
                    ServiceConfig.from_vllm(config)
        config.speculative_config.num_speculative_tokens=3
        with self.assertRaises(ValueError): ServiceConfig.from_vllm(config)

    def test_model_selection_is_explicit_and_keeps_native_parallelism(self):
        from test_worker_composition import BOOT
        code=BOOT+"""
hf.model_type='qwen3_5_moe_text'; hf.num_hidden_layers=40; hf.hidden_size=2048
hf.moe_intermediate_size=512; hf.num_experts=256; hf.num_experts_per_tok=8
c.model_config.enforce_eager=True
c.parallel_config.tensor_parallel_size=1
c.scheduler_config.max_num_batched_tokens=4096
try: select(c)
except ValueError as e: assert 'experimental' in str(e)
else: raise AssertionError('implicit expert activation')
c.additional_config={'betterscale_experts':dict(experimental=True,
 control='/task/control',build='/task/build',owners=2,sources=6,source=0)}
implementation,route=select(c)
assert implementation.__name__=='betterscale.models.qwen_experts'
assert route[0]=='qwen35-experts'
c.parallel_config.enable_expert_parallel=True
try: select(c)
except ValueError as e: assert 'TP1' in str(e)
else: raise AssertionError('native EP and separated service mixed')
"""
        subprocess.run([sys.executable,'-c',code],check=True)

    def test_build_must_match_physical_draft_catalog(self):
        with tempfile.TemporaryDirectory() as root:
            output=emit(Path(root)/'build', draft_layers=1)
            service=ServiceConfig('/task/control', str(output), 2, 6, 0, 1)
            with self.assertRaisesRegex(ValueError, 'Missing expert binary'): service.check_build()
            for name in ('launch.so','persistent_vector.o','persistent_cube.o','queue_service.o'):
                (output/name).write_bytes(b'fixture')
            service.check_build()
            with self.assertRaisesRegex(ValueError, 'ABI'):
                ServiceConfig('/task/control', str(output), 2, 6, 0, 0).check_build()

    def test_bind_rejects_stale_geometry(self):
        code='''
from betterscale.patches.expert_service.config import ServiceConfig
from betterscale.patches.expert_service.model_geometry import GEOMETRY
assert GEOMETRY.total_layers==40
try: ServiceConfig('/task/control','/task/build',2,6,0,1).bind('/model',96)
except RuntimeError as e: assert 'fresh process' in str(e)
else: raise AssertionError('stale catalog accepted')
'''
        env={k:v for k,v in os.environ.items() if not k.startswith('BETTERSCALE_EXPERT_')}
        subprocess.run([sys.executable,'-c',code],env=env,check=True)

    def test_all_41_layers_have_one_owner(self):
        code='''
from betterscale.patches.expert_service.config import ServiceConfig
ServiceConfig('/task/control','/task/build',2,6,0,1).bind('/model',96)
from betterscale.patches.expert_service.placement import Placement
for count in (1,2,4):
 p=Placement('layer',count)
 layers=[layer for owner in range(count) for layer in p.layers(owner)]
 assert len(layers)==41 and set(layers)==set(range(41))
 assert p.targets(40)==(40%count,)
'''
        env={k:v for k,v in os.environ.items() if not k.startswith('BETTERSCALE_EXPERT_')}
        subprocess.run([sys.executable,'-c',code],env=env,check=True)

    def test_worker_wraps_native_load_and_drains_before_shutdown(self):
        from test_worker_composition import BOOT
        code=BOOT+'''
Native.shutdown=lambda self: calls.append('native-shutdown')
implementation=S(check=lambda c: None, before_init=lambda w,c: None,
 load_model=lambda w,n: (calls.append('scope-enter'), n(), calls.append('scope-exit'))[1],
 model_loaded=lambda w: calls.append('loaded'),
 shutdown=lambda w: calls.append('drain'))
with patch.object(entry,'select',return_value=(implementation,'experts')):
 w=entry.Worker(c)
 assert w.load_model()=='loaded'
 w.shutdown()
assert calls==['native-init','scope-enter','native-load','scope-exit','loaded','drain','native-shutdown'],calls
'''
        subprocess.run([sys.executable,'-c',code],check=True)

class CheckpointLayout(unittest.TestCase):
    def test_target_fused_draft_unfused_names(self):
        # CPU-only module loading: no torch-npu or native worker import.
        code='''
import os
os.environ['BETTERSCALE_EXPERT_DRAFT_LAYERS']='1'
from betterscale.patches.expert_service.checkpoint import expert_names
names=expert_names(0)
assert names==['model.language_model.layers.0.mlp.experts.gate_up_proj',
 'model.language_model.layers.0.mlp.experts.down_proj']
names=expert_names(40)
assert len(names)==768 and len(set(names))==768
assert names[0]=='mtp.layers.0.mlp.experts.0.gate_proj.weight'
assert names[-1]=='mtp.layers.0.mlp.experts.255.down_proj.weight'
try: expert_names(41)
except ValueError: pass
else: raise AssertionError('extra physical draft layer accepted')
'''
        subprocess.run([sys.executable,'-c',code],check=True)

class DonorSeams(unittest.TestCase):
    def test_shared_expert_hook_uses_declared_pinned_runner_members(self):
        import ast
        root=Path(__file__).resolve().parents[1]
        source=ast.parse((root/'upstream/vllm-ascend/vllm_ascend/ops/fused_moe/fused_moe.py').read_text())
        runner=next(node for node in source.body if isinstance(node,ast.ClassDef) and node.name=='AscendMoERunner')
        declared={n.name for n in runner.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
        declared.update(n.attr for n in ast.walk(runner) if isinstance(n,ast.Attribute)
                        and isinstance(n.value,ast.Name) and n.value.id=='self' and isinstance(n.ctx,ast.Store))
        client=ast.parse((root/'src/betterscale/patches/expert_service/client.py').read_text())
        consumed={n.attr for n in ast.walk(client) if isinstance(n,ast.Attribute)
                  and isinstance(n.value,ast.Name) and n.value.id=='module'}
        self.assertTrue(consumed)
        self.assertEqual(consumed-declared,set())
