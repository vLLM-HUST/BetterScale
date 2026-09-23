"""CPU-only diagnostic observer contract: preserve native keyword dispatch."""
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch


class ProfileObserver(unittest.TestCase):
    def test_native_keyword_call(self):
        calls=[]
        def dispatch(num_tokens,num_reqs,num_scheduled_tokens_np,**kwargs):
            calls.append((num_tokens,num_reqs,num_scheduled_tokens_np,kwargs))
            return ('FULL','descriptor')
        runner=SimpleNamespace(_determine_batch_execution_and_padding=dispatch)
        class Native:
            def load_model(self):return 'loaded'
        native=ModuleType('native_worker');native.Worker=Native
        profiling=ModuleType('profiling');profiling.ProfileWindow=object
        with patch.dict(sys.modules,{'native_worker':native,'profiling':profiling}):
            path=Path(__file__).resolve().parents[1]/'prototypes/qwen35-moe-serving/dummy_worker.py'
            spec=importlib.util.spec_from_file_location('dummy_observer_test',path)
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            worker=module.Worker();worker.model_runner=runner
            self.assertEqual(worker.load_model(),'loaded')
            result=runner._determine_batch_execution_and_padding(
                num_tokens=12,num_reqs=4,num_scheduled_tokens_np=[3]*4,force=True)
            self.assertEqual(result,('FULL','descriptor'))
            self.assertEqual(calls,[(12,4,[3]*4,{'force':True})])
            class Values(list):
                def __getitem__(self,key):
                    value=super().__getitem__(key)
                    return Values(value) if isinstance(key,slice) else value
                def tolist(self):return list(self)
            runner.input_batch=SimpleNamespace(
                num_computed_tokens_cpu=Values([241664]),
                num_prompt_tokens_cpu_tensor=Values([262016]))
            worker.min_context=240000
            worker.window=SimpleNamespace(started=False,closed=False,count=59,
                prof=SimpleNamespace(start=Mock()),mark=Mock(),schedule=[])
            torch=ModuleType('torch');torch.npu=SimpleNamespace(synchronize=Mock())
            with patch.dict(sys.modules,{'torch':torch}):
                runner._determine_batch_execution_and_padding(
                    num_tokens=4096,num_reqs=1,num_scheduled_tokens_np=Values([4096]))
            self.assertTrue(worker.window.started)
            self.assertEqual(worker.window.warmup_steps,59)
            worker.window.prof.start.assert_called_once()
            self.assertEqual(worker.window.schedule[0]['computed'],[241664])
