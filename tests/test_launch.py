"""Public launch, compatibility and worker-envelope contracts; CPU only."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS
import tempfile
import unittest
from unittest.mock import patch

from strengthen_dsv4.compat import check_runtime
from strengthen_dsv4.config import PATCH_IDS, DP_PATCH_IDS, validate_worker_config


def config():
    return NS(parallel_config=NS(tensor_parallel_size=8,data_parallel_size=1,pipeline_parallel_size=1,
        enable_expert_parallel=True,decode_context_parallel_size=1,prefill_context_parallel_size=1),
        scheduler_config=NS(max_num_seqs=4,max_num_batched_tokens=4128,scheduler_cls=None),
        model_config=NS(hf_config=NS(model_type='deepseek_v4',num_hidden_layers=43,hidden_size=4096,
        n_routed_experts=256),max_model_len=15104,quantization='ascend'),
        speculative_config=NS(method='dspark',num_speculative_tokens=5,enforce_eager=True,rejection_sample_method='standard'),
        additional_config={'enable_dsa_cp':True},
        cache_config=NS(enable_prefix_caching=False),load_config=NS(load_format='auto'),
        compilation_config=NS(cudagraph_mode='FULL'))


class LaunchContract(unittest.TestCase):
    def test_native_config_needs_no_private_policy_and_is_not_mutated(self):
        import copy
        c=config();before=copy.deepcopy(c)
        self.assertIsNone(validate_worker_config(c))
        self.assertEqual(c,before)

    def test_capacity_is_not_set_by_the_patch(self):
        # User-selected KV budgets are unrelated to the graph admission contract.
        c=config();c.cache_config.kv_cache_memory_bytes=1024
        validate_worker_config(c)
        self.assertEqual(c.cache_config.kv_cache_memory_bytes,1024)

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
    def worker_class(self,calls):
        import ast
        source=Path(__file__).resolve().parents[1]/'src/strengthen_dsv4/worker.py'
        node=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.ClassDef))
        class Native:
            def __init__(self,config,*args,**kwargs):
                calls.append('native-init');self.rank=0;self.config=config
            def compile_or_warm_up_model(self):calls.append('native-capture');return 'native-times'
        ns=dict(__name__='strengthen_dsv4.worker_test',__package__='strengthen_dsv4',
            NPUWorker=Native,check_runtime=lambda:calls.append('compat'),
            validate_worker_config=validate_worker_config,PATCH_IDS=PATCH_IDS,DP_PATCH_IDS=DP_PATCH_IDS,
            log=NS(info=lambda *a,**kw:calls.append('ready')))
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),ns)
        return ns['Worker']

    def test_only_worker_cls_is_needed_and_native_configuration_is_preserved(self):
        import copy,os,sys
        from types import ModuleType
        calls=[];modules={}
        package=ModuleType('strengthen_dsv4.patches')
        modules[package.__name__]=package
        for name in ('compat_lcm','target_full','split_draft','cross_step','ordered_replay','qli_cpu'):
            function='install'
            module=ModuleType('strengthen_dsv4.patches.'+name)
            setattr(module,function,lambda *a,_name=name,**kw:calls.append(_name))
            modules[module.__name__]=module
            setattr(package,name,module)
        c=config();before=copy.deepcopy(c);environment=dict(os.environ)
        with patch.dict(sys.modules,modules), patch('pathlib.Path.mkdir',side_effect=AssertionError('No artifact directory')), \
             patch('pathlib.Path.write_text',side_effect=AssertionError('No receipt files')):
            worker=self.worker_class(calls)(c)
            self.assertIs(worker.config,c)
            self.assertEqual(worker.compile_or_warm_up_model(),'native-times')
        self.assertEqual(c,before)
        self.assertEqual(dict(os.environ),environment)
        self.assertEqual(calls,['compat','compat_lcm','target_full','native-init','native-capture',
            'split_draft','cross_step','ordered_replay','qli_cpu','ready'])

    def test_bad_configuration_fails_before_native_initialization(self):
        calls=[];c=config();c.speculative_config.num_speculative_tokens=3
        with self.assertRaises(ValueError):self.worker_class(calls)(c)
        self.assertEqual(calls,['compat'])


class NativeDPAdmission(unittest.TestCase):
    def test_bounded_dp_config_and_backend_are_validated_without_mutation(self):
        import copy
        c = config()
        c.parallel_config.tensor_parallel_size = 1
        c.parallel_config.data_parallel_size = 8
        c.scheduler_config.max_num_seqs = 2
        c.scheduler_config.max_num_batched_tokens = 1026
        c.model_config.max_model_len = 16384
        c.additional_config["enable_dsa_cp"] = False
        before = copy.deepcopy(c)
        validate_worker_config(c)
        self.assertEqual(c, before)
        c.additional_config["enable_dsa_cp"] = True
        with self.assertRaisesRegex(ValueError, "backend"):
            validate_worker_config(c)
        c.additional_config["enable_dsa_cp"] = False
        c.scheduler_config.max_num_seqs = 4
        with self.assertRaisesRegex(ValueError, "seats"):
            validate_worker_config(c)


class DPLifecycle(WorkerLifecycle):
    def test_native_dp_composition_excludes_tp_draft_and_experiment_hooks(self):
        import copy, sys
        from types import ModuleType
        calls = []
        package = ModuleType("strengthen_dsv4.patches")
        modules = {package.__name__: package}
        for name in ("compat_lcm", "target_full", "cross_step", "async_decode"):
            module = ModuleType("strengthen_dsv4.patches." + name)
            module.install = lambda *a, _name=name, **kw: calls.append((_name, kw))
            if name == "async_decode":
                module.install_capture = lambda: calls.append("capture-hook")
            modules[module.__name__] = module
            setattr(package, name, module)
        c = config()
        c.parallel_config.tensor_parallel_size = 1
        c.parallel_config.data_parallel_size = 8
        c.scheduler_config.max_num_seqs = 2
        c.scheduler_config.max_num_batched_tokens = 1026
        c.additional_config["enable_dsa_cp"] = False
        before = copy.deepcopy(c)
        with patch.dict(sys.modules, modules), patch("pathlib.Path.write_text", side_effect=AssertionError("No receipts")):
            worker = self.worker_class(calls)(c)
            self.assertEqual(worker.compile_or_warm_up_model(), "native-times")
        self.assertEqual(c, before)
        self.assertEqual(calls, ["compat", ("compat_lcm", {}),
            ("target_full", {"native_dsa": True}), "capture-hook", "native-init", "native-capture",
            ("cross_step", {"native_dsa": True, "max_requests": 2}),
            ("async_decode", {}), "ready"])
