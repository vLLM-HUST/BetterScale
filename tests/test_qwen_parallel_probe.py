"""The two-chip matrix must express independent attention/expert axes."""
import importlib.util
import json
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest


class ParallelProbe(unittest.TestCase):
    def test_partition_gate_rejects_replication_and_overlapping_experts(self):
        path = Path(__file__).resolve().parents[1] / 'prototypes/qwen35-moe-serving/check_parallel.py'
        spec = importlib.util.spec_from_file_location('matrix_check', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for ep in (False, True):
            receipts = []
            for rank in (0, 1):
                expert = {'name': 'expert', 'intermediate_size': 512,
                          'weights': {'w13_weight': [128, 2048, 1024] if ep else [256, 2048, 512],
                                      'w2_weight': [128, 512, 2048] if ep else [256, 256, 2048]},
                          'expert_parallel': {'tp_size': 1 if ep else 2,
                                              'ep_size': 2 if ep else 1, 'dp_size': 2,
                                              'use_ep': ep, 'ep_rank': rank if ep else 0,
                                              'tp_rank': 0 if ep else rank},
                          'expert_map': [i - rank*128 if rank*128 <= i < (rank+1)*128 else -1
                                         for i in range(256)] if ep else None}
                receipt = {'attention_tp': 1, 'attention_dp': 2,
                           'enable_expert_parallel': ep, 'data_parallel_rank': rank, 'rank': 0}
                for scope, count, attention in [('target', 40, 10), ('draft', 1, 1)]:
                    receipt[scope] = [dict(copy.deepcopy(expert), name=f'layer{i}.expert')
                                      for i in range(count)]
                    receipt[scope] += [{'name': f'layer{i}.qkv_proj',
                                        'weights': {'weight': [9216, 2048]}}
                                       for i in range(attention)]
                receipts.append(receipt)
            self.assertEqual(module.check(receipts)['status'], 'PASS')
            if ep:
                receipts[1]['target'][0]['expert_map'] = receipts[0]['target'][0]['expert_map']
            else:
                receipts[1]['target'][0]['weights']['w2_weight'] = [256, 512, 2048]
            with self.assertRaises(AssertionError):
                module.check(receipts)

    def test_geometry_replacements_are_simultaneous_and_fail_closed(self):
        path = Path(__file__).resolve().parents[1] / 'prototypes/qwen35-moe-serving/stage_dp_attention.py'
        spec = importlib.util.spec_from_file_location('matrix_stage', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        text = 'q=(1, t, 8, 128); v=(1, t, 16, 128); requests=16; align=8'
        self.assertEqual(module.replace(text, {
            '(1, t, 8, 128)': '(1, t, 16, 128)',
            '(1, t, 16, 128)': '(1, t, 32, 128)',
        }), 'q=(1, t, 16, 128); v=(1, t, 32, 128); requests=16; align=8')
        with self.assertRaises(ValueError):
            module.replace(text, {'not-present': 'unsafe'})

    def test_native_four_topologies_keep_real_mtp_and_native_context(self):
        path = Path(__file__).resolve().parents[1] / 'prototypes/qwen35-moe-serving/probe.py'
        spec = importlib.util.spec_from_file_location('matrix_probe', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for tp, dp in [(2, 1), (1, 2)]:
            for ep in [False, True]:
                with self.subTest(tp=tp, dp=dp, ep=ep):
                    args = SimpleNamespace(model='/model', port=33981,
                        worker='parallel_worker.Worker', tensor_parallel_size=tp,
                        data_parallel_size=dp, enable_expert_parallel=ep)
                    cmd = module.server_command(args)
                    value = lambda key: cmd[cmd.index(key) + 1]
                    self.assertEqual(value('--tensor-parallel-size'), str(tp))
                    self.assertEqual(value('--data-parallel-size'), str(dp))
                    self.assertEqual('--enable-expert-parallel' in cmd, ep)
                    self.assertEqual(value('--max-model-len'), '262144')
                    self.assertEqual(value('--dtype'), 'bfloat16')
                    self.assertEqual(json.loads(value('--speculative-config')),
                                     {'method': 'mtp', 'num_speculative_tokens': 2})
