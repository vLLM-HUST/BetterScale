"""CPU guards for draft-only policy and explicit donor source staging."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

ROOT = Path(__file__).resolve().parents[1]/'prototypes/qwen-mtp-small-fish'
def load(name):
    spec = importlib.util.spec_from_file_location('small_fish_'+name, ROOT/(name+'.py'))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m
runtime, stage = load('runtime'), load('stage')

class SmallFish(unittest.TestCase):
    def fixture(self):
        p = NS(vllm_config=NS(parallel_config=NS(tensor_parallel_size=2,
            data_parallel_size=1,pipeline_parallel_size=1,enable_expert_parallel=False),
            model_config=NS(hf_text_config=NS(model_type='qwen3_5_moe_text',vocab_size=248320),
                            dtype='torch.bfloat16',quantization=None)),
            method='mtp',num_speculative_tokens=2,parallel_drafting=False,extra_slots_per_request=1,
            model=NS(logits_processor=NS(),lm_head=NS(num_org_embeddings_per_partition=124160,
                                                    num_embeddings_per_partition=124160)))
        return p, NS(logits_processor=NS()), NS(enable_reduce_sample=False)

    def test_target_stays_independent(self):
        p,t,c = self.fixture()
        result = runtime.admit_draft(p,t,c,False)
        self.assertIs(result,p.model.logits_processor)
        result._betterscale_draft_greedy=True
        self.assertFalse(hasattr(t.logits_processor,'_betterscale_draft_greedy'))
        self.assertFalse(c.enable_reduce_sample)
        self.assertIs(runtime.admit_draft(p,NS(get_language_model=lambda:t),c,False),result)

    def test_unqualified_routes_rejected(self):
        for change in (
            lambda p,t,c:setattr(c,'enable_reduce_sample',True),
            lambda p,t,c:setattr(p,'parallel_drafting',True),
            lambda p,t,c:setattr(p,'num_speculative_tokens',3),
            lambda p,t,c:setattr(p.model.lm_head,'num_embeddings_per_partition',124288),
            lambda p,t,c:setattr(t,'logits_processor',p.model.logits_processor),
            lambda p,t,c:setattr(t.logits_processor,'_betterscale_draft_greedy',True),
            lambda p,t,c:setattr(p.vllm_config.model_config,'quantization','int8'),
        ):
            p,t,c=self.fixture();change(p,t,c)
            with self.assertRaises(ValueError): runtime.admit_draft(p,t,c,False)
        p,t,c=self.fixture()
        with self.assertRaises(ValueError): runtime.admit_draft(p,t,c,True)

    def test_gate_patches_do_not_cascade(self):
        text='def f(self):\n    if True:\n        if True:\n            b = b.contiguous()\n            a = a.contiguous()\n        else:\n            if True:\n                b = b.contiguous()\n                a = a.contiguous()\n'
        changed=stage.gdn_source(text)
        compile(changed,'fixture','exec')
        self.assertEqual(changed.count('if not getattr(self, "_betterscale_strided_gates", False):'),2)
        class Buffer:
            def __init__(self): self.copies=0
            def contiguous(self): self.copies+=1;return self
        # An already-staged source must fail closed rather than stack patches.
        with self.assertRaises(ValueError): stage.gdn_source(changed)

    def test_only_selected_method_is_patched(self):
        code='class AscendSpecDecodeBaseProposer:\n    def unrelated(self):\n        if get_ascend_config().enable_reduce_sample: pass\n    def _run_merged_draft(self):\n        if get_ascend_config().enable_reduce_sample: pass\n        if get_ascend_config().enable_reduce_sample: pass\n'
        changed=stage.proposer_source(code)
        self.assertEqual(changed.count('_betterscale_draft_greedy'),2)
        self.assertIn('def unrelated(self):\n        if get_ascend_config().enable_reduce_sample: pass',changed)
        compile(changed,'fixture','exec')
        with self.assertRaises(ValueError): stage.proposer_source(changed)

if __name__=='__main__': unittest.main()
