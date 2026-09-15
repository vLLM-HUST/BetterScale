import sys
import types
import unittest
from pathlib import Path
from capture_memory_trace import CaptureMemoryTrace


class TraceTests(unittest.TestCase):
    def test_counters_and_trace_restoration_without_device_operations(self):
        counters = dict(allocated=0, peak=0, reserved=0, peak_reserved=0)
        npu = types.SimpleNamespace(memory_allocated=lambda: counters['allocated'],
            max_memory_allocated=lambda: counters['peak'],
            memory_reserved=lambda: counters['reserved'],
            max_memory_reserved=lambda: counters['peak_reserved'])
        old = sys.modules.get('torch')
        sys.modules['torch'] = types.SimpleNamespace(npu=npu)
        previous = sys.gettrace()
        try:
            def exercise():
                counters.update(allocated=64*2**20, peak=64*2**20, reserved=80*2**20, peak_reserved=80*2**20)
                counters.update(allocated=0)
                counters.update(peak=0, peak_reserved=0)
            with CaptureMemoryTrace([Path(__file__).parent]) as trace:
                exercise()
            self.assertIs(sys.gettrace(), previous)
            self.assertTrue(any(r['new_peak'] for r in trace.records))
            self.assertTrue(any(r['peak_reset'] for r in trace.records))
            self.assertTrue(all('reserved' in r for r in trace.records))
        finally:
            if old is None: sys.modules.pop('torch')
            else: sys.modules['torch'] = old


if __name__ == '__main__': unittest.main()
