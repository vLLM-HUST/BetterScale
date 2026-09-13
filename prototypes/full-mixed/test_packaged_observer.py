"""CPU contract between the packaged observer and the existing quality harness."""
import sys
import unittest
from types import ModuleType, SimpleNamespace as NS
from unittest.mock import patch
import torch
from packaged_observer import PackagedObserver


class QualityReceipt(unittest.TestCase):
    def test_native_capacity_survives_observer_extraction(self):
        module = ModuleType('vllm.v1.core.kv_cache_utils')
        config, cache = object(), object()
        def capacity(c, k):
            self.assertIs(c, config)
            self.assertIs(k, cache)
            return 32768, 2.0
        module.get_kv_cache_capacity = capacity
        runner = NS(vllm_config=config, kv_cache_config=cache,
                    _async_decode=(NS(sequence=9, slots={0: None}), NS(entries={}, replays=8)),
                    model=NS(_decode_pair=NS(replays=[5, 4])))
        config = runner.vllm_config = NS(parallel_config=NS(data_parallel_rank=3))
        observer = PackagedObserver()
        observer.model_runner = runner
        npu = NS(memory_allocated=lambda: 1, memory_reserved=lambda: 2, max_memory_allocated=lambda: 3)
        with patch.dict(sys.modules, {module.__name__: module}), patch.object(torch, 'npu', npu, create=True):
            receipt = observer.donor_receipt()
        self.assertEqual(receipt['kv_capacity_tokens'], 32768)
        self.assertGreaterEqual(receipt['max_length_concurrency'], 2)
        self.assertEqual(receipt['rank'], 3)


class FailureVisibility(unittest.TestCase):
    def test_parent_barrier_breaks_before_native_teardown(self):
        import donor_dp
        events = []
        barrier = NS(abort=lambda: events.append('aborted'))
        with patch.object(donor_dp, 'rank_main', side_effect=KeyError('max_length_concurrency')), \
             patch('traceback.print_exc', side_effect=lambda: events.append('traceback')):
            with self.assertRaises(KeyError):
                donor_dp.rank_entry(NS(), 0, barrier)
        self.assertEqual(events, ['traceback', 'aborted'])
