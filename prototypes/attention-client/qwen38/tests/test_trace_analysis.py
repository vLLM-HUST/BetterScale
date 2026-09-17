import sys
from pathlib import Path
import json
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze_traces import summarize


class AnalysisTests(unittest.TestCase):
    def test_tp1_counts_every_attention_source_once(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            for i in range(2):
                row = dict(
                    status="PASS",
                    tp_size=1,
                    prefix_first_page_exact=True,
                    tp_output_exact=True,
                    mtp_tokens=1,
                    seconds=2 + i,
                    sessions=[
                        dict(
                            trace_id=str(i),
                            completed_turns=1,
                            output_tokens=7,
                            prefill_tokens=3,
                            prefix_reused_tokens=0,
                            final_encoded_context=9,
                            events=[dict(ttft_seconds=0.5)],
                        )
                    ],
                    output_intervals_seconds=[0.1, 0.2],
                    memory=dict(peak_allocated=1024, reserved=2048),
                )
                (directory / f"attention{i}.json").write_text(json.dumps(row))
            result = summarize(directory, 2, tp_size=1)
            self.assertEqual(result["committed_tokens"], 14)
            self.assertEqual(result["max_source_seconds"], 3)
            self.assertEqual(len(result["sessions"]), 2)
            with self.assertRaises(ValueError):
                summarize(directory, 1, tp_size=2)


if __name__ == "__main__":
    unittest.main()
