"""CPU checks for token/request separation before merged draft graph capture."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import sys
import tempfile

PATH = Path(__file__).resolve().parents[1]/'prototypes/qwen35-moe-serving/draft_sampling.py'
spec = importlib.util.spec_from_file_location('draft_sampling', PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Buffer:
    def __init__(self, rows):
        self.rows = rows
        self.shape = (len(rows),)

    def __getitem__(self, key):
        return Buffer(self.rows[key])


class DraftSampling(unittest.TestCase):
    def proposer(self, live=0):
        return SimpleNamespace(runner=SimpleNamespace(max_num_reqs=16,
            input_batch=SimpleNamespace(num_reqs=live)),
            token_indices_to_sample=Buffer([0]*4096))

    def test_capture_and_profile_share_request_envelope(self):
        p = self.proposer()
        for tokens in (3,6,12,16,24,32,40,48,64,128,256,512,1024,1536,2048,4096):
            for profile in (False, True):
                donor_rows = max(tokens//3, 1)
                if profile:
                    donor_rows = min(donor_rows,16)
                count, indices = module.sampling_inputs(p,tokens,donor_rows,
                    p.token_indices_to_sample[:donor_rows])
                self.assertEqual(count,min(tokens,16))
                self.assertEqual(indices.shape,(count,))
                self.assertEqual(indices.rows,[0]*count)

    def test_live_indices_and_shrink_reuse_are_not_changed(self):
        for live in (16,1,8,2,16):
            p = self.proposer(live)
            indices = Buffer([4095-i*3 for i in range(live)])
            count, result = module.sampling_inputs(p,4096,live,indices)
            self.assertEqual(count,live)
            self.assertIs(result,indices)
        # Nonuniform three-token wave can contain three distinct requests.
        p = self.proposer(3)
        self.assertEqual(module.sampling_inputs(p,3,3,Buffer([0,1,2]))[0],3)

    def test_staging_preserves_seed_and_installs_before_model_load(self):
        path = PATH.with_name('stage_draft_sampling.py')
        spec = importlib.util.spec_from_file_location('stage_sampling_test',path)
        stage = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(stage)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            bank = output/'draft_banks.py'
            bank.write_text('def install():\n    install_live_draft_rows()\n')
            stage.install_in_capsule(output)
            self.assertEqual((output/'draft_sampling.py').read_text(),PATH.read_text())
            self.assertIn('    install_request_sampling()\n',bank.read_text())
            before = bank.read_text()
            with self.assertRaises(AssertionError):
                stage.install_in_capsule(output)
            self.assertEqual(bank.read_text(),before)

    def test_installed_wrapper_bounds_before_original_computation(self):
        calls = []
        class Proposer:
            def _run_merged_draft(self, *args):
                calls.append(args)
                return args[1], args[2]
        fake = SimpleNamespace(AscendSpecDecodeBaseProposer=Proposer)
        with patch.dict(sys.modules, {'vllm_ascend.spec_decode.llm_base_proposer': fake}):
            module.install()
            wrapped = Proposer._run_merged_draft
            module.install()
            self.assertIs(Proposer._run_merged_draft, wrapped)
        p = Proposer()
        p.__dict__.update(self.proposer().__dict__)
        p.method, p.num_speculative_tokens = 'mtp', 2
        p.extra_slots_per_request, p.parallel_drafting = 1, False
        position, embeds, metadata = object(), object(), object()
        count, indices = p._run_merged_draft(4096,1365,Buffer([0]*1365),
            position,embeds,metadata,4096)
        self.assertEqual((count,indices.shape),(16,(16,)))
        self.assertEqual(calls[0][0],4096)  # Never shrink model/attention input.
        self.assertIs(calls[0][3],position)
        self.assertIs(calls[0][5],metadata)
        p.parallel_drafting = True
        with self.assertRaises(ValueError):
            p._run_merged_draft(4096,1365,Buffer([0]*1365),position,embeds,metadata,4096)

    def test_overflow_is_rejected_not_silently_sliced(self):
        for live,tokens,batch,rows in ((17,4096,17,17),(16,12,16,16),(8,24,9,9),(8,24,8,7)):
            with self.assertRaises(ValueError):
                module.sampling_inputs(self.proposer(live),tokens,batch,Buffer([0]*rows))


if __name__ == '__main__':
    unittest.main()
