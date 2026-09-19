import contextlib
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from transport import TwoBankExecutor


class Stream:
    def __init__(self):
        self.waits = []

    def wait_event(self, event):
        self.waits.append(event)


class Event:
    def record(self, stream=None):
        self.stream = stream

    def synchronize(self):
        pass


class TransportContracts(unittest.TestCase):
    def setUp(self):
        self.egress = torch.zeros((2, 14), dtype=torch.int64)
        self.sequence = 0
        def replay():
            self.egress[self.sequence % 2].fill_(self.sequence)
            self.sequence += 1
        self.graph = SimpleNamespace(replay=replay)
        backend = SimpleNamespace(Stream=Stream, Event=Event,
                                  stream=lambda _: contextlib.nullcontext())
        self.executor = TwoBankExecutor(self.graph, self.egress, backend=backend)
        native_empty = torch.empty_like
        self.allocator = patch('transport.torch.empty_like',
                               side_effect=lambda t, **kw: native_empty(t))
        self.allocator.start()
        self.addCleanup(self.allocator.stop)

    def test_two_unread_banks_reject_third_launch(self):
        self.executor.submit()
        self.executor.submit()
        with self.assertRaisesRegex(RuntimeError, 'window is full'):
            self.executor.submit()
        self.assertEqual(self.sequence, 2)

    def test_reuse_waits_for_old_copy_and_preserves_receipts(self):
        first, second = self.executor.submit(), self.executor.submit()
        self.executor.copy_out(first)
        self.executor.copy_out(second)
        self.executor.receive_oldest()
        third = self.executor.submit()
        self.assertIs(self.executor.compute.waits[-1], first.copied)
        self.executor.copy_out(third)
        self.assertEqual(self.executor.finish(), [[n] * 14 for n in range(3)])

    def test_no_duplicate_copy_or_finish_before_copy(self):
        first = self.executor.submit()
        with self.assertRaises(RuntimeError):
            self.executor.finish()
        self.executor.copy_out(first)
        with self.assertRaises(RuntimeError):
            self.executor.copy_out(first)

    def test_sequence_mismatch_poisoned(self):
        first = self.executor.submit()
        self.executor.copy_out(first)
        first.host[0] = 99
        with self.assertRaisesRegex(RuntimeError, 'stale or torn'):
            self.executor.finish()
        with self.assertRaises(RuntimeError):
            self.executor.submit()

    def test_failed_submission_cannot_retry_partially_executed_wave(self):
        def fail():
            raise RuntimeError('native failure')
        self.graph.replay = fail
        with self.assertRaisesRegex(RuntimeError, 'native failure'):
            self.executor.submit()
        with self.assertRaisesRegex(RuntimeError, 'failed'):
            self.executor.submit()


if __name__ == '__main__':
    unittest.main()
