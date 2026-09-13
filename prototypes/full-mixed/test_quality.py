"""Quality harness admission and complete case/result mapping, without NPUs."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import quality


class Engine:
    def __init__(self, capacity=4): self.capacity=capacity; self.generated=[]
    def collective_rpc(self,name,args=()):
        if name=='quality_capacity':return [dict(rank=i,max_length_concurrency=self.capacity) for i in range(8)]
        if name=='graph_receipt':return [dict(wrappers=[dict(entries=[dict(captured=True,replays=1,tokens=4128)])]) for _ in range(8)]
        return []
    def generate(self,prompts,params):
        self.generated.extend(prompts)
        assert len(prompts)==len(params)==4
        assert all(p.temperature==0 and not p.ignore_eos and p.stop_token_ids==[1] for p in params)
        return [NS(prompt_token_ids=p['prompt_token_ids'],finished=True,
                   outputs=[NS(token_ids=[17],finish_reason='stop',stop_reason=1)]) for p in prompts]


class QualityTests(unittest.TestCase):
    def exercise(self,capacity):
        with TemporaryDirectory() as tmp, patch.dict('sys.modules',vllm=NS(SamplingParams=NS)):
            root=Path(tmp); source=root/'inputs.json'
            source.write_text(json.dumps([dict(request_id=str(i),prompt_token_ids=[i+2]*7,max_new_tokens=32,eos_token_id=1) for i in range(32)]))
            engine=Engine(capacity)
            if capacity<4:
                with self.assertRaisesRegex(AssertionError,'KV capacity'):
                    quality.run(engine,root,source)
                self.assertEqual(engine.generated,[])
            else:
                quality.run(engine,root,source)
                result=json.loads((root/'quality-result.json').read_text())
                self.assertEqual(len(engine.generated),32)
                self.assertEqual([r['request_id'] for r in result['requests']],[str(i) for i in range(32)])
                self.assertEqual(result['status'],'COMPLETED_UNSCORED')
    def test_complete_case_mapping(self):self.exercise(4)
    def test_capacity_rejects_before_generation(self):self.exercise(1.19)
