"""Step plots must select observed work, not requested HTTP concurrency."""

import importlib.util
from pathlib import Path
import unittest


path = Path(__file__).parents[1] / "prototypes/qwen38-serving/summarize_steps.py"
spec = importlib.util.spec_from_file_location("summarize_steps", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def row(scheduled, computed, prompts):
    return dict(scheduled=scheduled, computed=computed, prompt_tokens=prompts,
                period_ms=12, forward_ms=10, prepare_ms=1, after_forward_ms=1)


class StepSelectionTest(unittest.TestCase):
    def test_decode_requires_occupied_common_context_window(self):
        case = dict(kind="decode", batch=2, context=1024)
        value = row([1, 1], [1032, 1033], [1024, 1024])
        self.assertTrue(module.selected(value, value, case))
        for other in (row([1], [1033], [1024]),
                      row([1, 1], [1024, 1025], [1024, 1024]),
                      row([1, 1], [1033, 1034], [4096, 4096])):
            self.assertFalse(module.selected(other, value, case))
        self.assertFalse(module.selected(value, row([1], [1034], [1024]), case))
        self.assertFalse(module.selected(value, None, case))

    def test_prefill_does_not_count_cached_or_chunked_work_as_full_prompt(self):
        case = dict(kind="prefill", batch=1, context=2048)
        following = row([1], [2048], [2048])
        self.assertTrue(module.selected(row([2048], [0], [2048]), following, case))
        self.assertFalse(module.selected(row([512], [1536], [2048]), following, case))

    def test_mixed_requires_actual_decode_and_cold_joining_prefill(self):
        case = dict(kind="mixed", batch=1, context=1024, joining=512)
        following = row([1, 1], [1029, 512], [1024, 512])
        self.assertTrue(module.selected(row([512, 1], [0, 1028], [512, 1024]), following, case))
        self.assertFalse(module.selected(row([512], [0], [512]), following, case))
        self.assertFalse(module.selected(row([512, 1], [512, 1028], [1024, 1024]), following, case))

    def test_timing_and_tp_alignment_are_checked(self):
        value = row([128], [0], [128])
        end = row([1], [128], [128])
        cohort = dict(case=dict(kind="prefill", batch=1, context=128),
                      measurement=dict(results=[dict(rank=r, steps=[dict(value), dict(end)]) for r in (0, 1)]))
        self.assertEqual(len(module.cohort_rows(cohort)[0]["rows"]), 1)
        cohort["measurement"]["results"][1]["steps"][0]["period_ms"] = 20
        with self.assertRaises(AssertionError):
            module.cohort_rows(cohort)


if __name__ == "__main__":
    unittest.main()
