"""Public launch, compatibility and worker-envelope contracts; CPU only."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS
import tempfile
import unittest
from unittest.mock import patch

from strengthen_dsv4.cli import command
from strengthen_dsv4.compat import check_runtime
from strengthen_dsv4.config import engine_options, validate_worker_config


def config():
    return NS(parallel_config=NS(tensor_parallel_size=8,data_parallel_size=1,pipeline_parallel_size=1,
        enable_expert_parallel=True,decode_context_parallel_size=1,prefill_context_parallel_size=1),
        scheduler_config=NS(max_num_seqs=4,max_num_batched_tokens=4128,scheduler_cls=None),
        model_config=NS(hf_config=NS(model_type='deepseek_v4',num_hidden_layers=43,hidden_size=4096,
        n_routed_experts=256),max_model_len=15104,quantization='ascend'),
        speculative_config=NS(method='dspark',num_speculative_tokens=5,enforce_eager=True,rejection_sample_method='standard'),
        additional_config={'enable_dsa_cp':True,'strengthen_dsv4':{'profile':'optimized','artifacts':'/tmp/x'}},
        cache_config=NS(enable_prefix_caching=False),load_config=NS(load_format='auto'),
        compilation_config=NS(cudagraph_mode='FULL'))


class LaunchContract(unittest.TestCase):
    def test_native_launch_and_explicit_baseline(self):
        candidate=engine_options('/models/dsv4','/tmp/candidate')
        baseline=engine_options('/models/dsv4','/tmp/baseline','baseline')
        self.assertEqual(candidate['worker_cls'],'strengthen_dsv4.worker.Worker')
        self.assertEqual(candidate['compilation_config']['cudagraph_mode'],'FULL')
        self.assertEqual(baseline['compilation_config']['cudagraph_mode'],'FULL_DECODE_ONLY')
        argv=command(candidate,'127.0.0.1',8000,'dsv4')
        self.assertIn('--no-enable-prefix-caching',argv)
        self.assertIn('--enable-expert-parallel',argv)
        self.assertEqual(json.loads(argv[argv.index('--speculative-config')+1])['num_speculative_tokens'],5)
        self.assertNotIn('prototype',' '.join(argv))
        self.assertNotIn('scheduler_cls',candidate)

    def test_qualified_config(self):
        self.assertEqual(validate_worker_config(config())['profile'],'optimized')

    def test_reject_unqualified_parallelism_and_scheduler(self):
        for field,value in [('data_parallel_size',2),('tensor_parallel_size',4),('decode_context_parallel_size',2)]:
            c=config();setattr(c.parallel_config,field,value)
            with self.assertRaisesRegex(ValueError,'qualified'):validate_worker_config(c)
        c=config();c.scheduler_config.scheduler_cls='experimental.N2'
        with self.assertRaisesRegex(ValueError,'native scheduler'):validate_worker_config(c)

    def test_reject_unqualified_request_or_draft_capacity(self):
        c=config();c.scheduler_config.max_num_seqs=8
        with self.assertRaises(ValueError):validate_worker_config(c)
        c=config();c.speculative_config.num_speculative_tokens=3
        with self.assertRaises(ValueError):validate_worker_config(c)
        with self.assertRaises(ValueError):engine_options('m','/tmp/x',kv_gib=-1)

    def test_fail_closed_on_changed_private_api(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'worker.py';source.write_bytes(b'qualified')
            manifest={'versions':{'donor':'1.0'},'commits':{'donor':'pin'},'source_files':[
                {'distribution':'donor','path':'worker.py','sha256':hashlib.sha256(b'qualified').hexdigest()}]}
            with patch('strengthen_dsv4.compat.pins',return_value=manifest), \
                 patch('strengthen_dsv4.compat.metadata.version',return_value='1.0+build'), \
                 patch('strengthen_dsv4.compat.metadata.distribution',return_value=NS(locate_file=lambda p:root/p)):
                check_runtime.cache_clear();self.assertEqual(check_runtime()['checked_source_files'],1)
                source.write_bytes(b'changed')
                check_runtime.cache_clear()
                with self.assertRaisesRegex(RuntimeError,'differs'):check_runtime()
        check_runtime.cache_clear()

class WorkerLifecycle(unittest.TestCase):
    def test_patches_install_after_native_capture_without_private_rpc(self):
        import ast, os, sys
        from types import ModuleType
        source=Path(__file__).resolve().parents[1]/'src/strengthen_dsv4/worker.py'
        node=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.ClassDef))
        for profile in ('baseline','optimized'):
            calls=[]
            class Native:
                def __init__(self,*args,**kwargs):
                    calls.append('native-init');self.rank=0
                def compile_or_warm_up_model(self):calls.append('native-capture');return 'native-times'
                def shutdown(self):calls.append('native-shutdown')
            modules={}
            for name,function in [('target','install'),('split_draft','install'),('cross_step','install'),
                                  ('ordered_replay','configure'),('qli_cpu','configure')]:
                module=ModuleType('strengthen_dsv4.patches.'+name)
                setattr(module,function,lambda *a,_name=name,**kw:calls.append(_name))
                modules[module.__name__]=module
            with tempfile.TemporaryDirectory() as folder, patch.dict(sys.modules,modules),patch.dict(os.environ):
                ns=dict(__name__='strengthen_dsv4.worker_test',__package__='strengthen_dsv4',
                    NPUWorker=Native,check_runtime=lambda:{},
                    validate_worker_config=lambda c:{'profile':profile,'artifacts':folder},
                    Path=Path,os=os,log=NS(info=lambda *a,**kw:None))
                exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),ns)
                worker=ns['Worker'](None)
                worker.strengthen_status=lambda:{'max_length_concurrency':4.7,'patches':[]}
                worker._write_strengthen_status=lambda *args:calls.append('ready')
                self.assertEqual(worker.compile_or_warm_up_model(),'native-times')
                expected=['target','native-init','native-capture']
                if profile=='optimized':expected+=['split_draft','cross_step','ordered_replay','qli_cpu']
                self.assertEqual(calls,expected+['ready'])
