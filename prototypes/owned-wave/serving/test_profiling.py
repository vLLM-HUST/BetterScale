"""Short profile windows start after drained warmup and stop once."""

import unittest
from unittest.mock import Mock, patch
from profiling import ProfileWindow


class ProfileTests(unittest.TestCase):
    def test_warmup_then_four_steps(self):
        window = object.__new__(ProfileWindow)
        window.steps = 4
        window.warmup_steps = 8
        window.count = 0
        window.started = window.closed = False
        window.prof = Mock()
        with patch("torch.npu.synchronize") as sync:
            for _ in range(7):
                window.step()
            window.prof.start.assert_not_called()
            window.step()
            window.prof.start.assert_called_once()
            for _ in range(3):
                window.step()
            window.prof.stop.assert_not_called()
            window.step()
            window.close()
            window.step()
            window.prof.stop.assert_called_once()
            self.assertEqual(sync.call_count, 2)

    def test_unprofiled_has_no_backend_effects(self):
        window = ProfileWindow("unused", 0, 0, warmup_steps=8)
        window.step()
        window.close()
        self.assertIsNone(window.prof)
