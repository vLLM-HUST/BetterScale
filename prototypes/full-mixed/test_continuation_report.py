import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from continuation_report import analyze


class CommonWindowTests(unittest.TestCase):
    def fixture(self, root):
        window = root / 'decode-test'
        window.mkdir()
        for rank in range(2):
            events, modes = [], []
            for i in range(6):
                events.extend([
                    dict(label='strengthen::target_forward', device_start_ms=i * 60,
                         device_elapsed_ms=40, host_elapsed_ms=1),
                    dict(label='strengthen::draft_forward', device_start_ms=i * 60 + 42,
                         device_elapsed_ms=5, host_elapsed_ms=20)])
                modes.append(dict(mode='FULL', dummy=rank == 1 and i == 1,
                                  actual_requests=2, actual_tokens=12))
            (window / f'decode-test-timing-rank{rank}.json').write_text(json.dumps(events))
            (window / f'decode-test-modes-rank{rank}.json').write_text(json.dumps(modes))
        return window

    def test_same_ordinals_on_both_ranks_and_additive_per_wave_spans(self):
        with TemporaryDirectory() as temp:
            result = analyze(self.fixture(Path(temp)), 2, 2, count=2)
        self.assertEqual(result['common_waves'], [2, 3])
        self.assertEqual(result['median_ms']['draft_to_target'], 13)
        for rank in result['ranks']:
            for row in rank['waves']:
                m = row['ms']
                self.assertEqual(m['cycle'], sum(m[k] for k in (
                    'target', 'draft', 'target_to_draft', 'draft_to_target')))

    def test_refuse_missing_common_window_instead_of_using_rank_local_selection(self):
        with TemporaryDirectory() as temp:
            window = self.fixture(Path(temp))
            with self.assertRaisesRegex(AssertionError, 'common occupied window'):
                analyze(window, 2, 2, count=4)


if __name__ == '__main__':
    unittest.main()
