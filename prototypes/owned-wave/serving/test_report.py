"""Reports must honor configured TP width and retain peer disagreement gates."""

import json
from pathlib import Path
import tempfile
import unittest
from report import compare
from compare_candidate import compare_candidate


class ReportTests(unittest.TestCase):
    def fixture(self, root, tp):
        engine = root / "engine"
        engine.mkdir()
        (engine / "config.json").write_text(
            json.dumps(dict(model="fixture", tensor_parallel_size=tp))
        )
        record = dict(
            session=0,
            turn=0,
            prompt_tokens=4,
            output_tokens=2,
            token_ids=[1, 2],
            ttft_s=0.1,
            tpot_s=0.1,
            latency_s=0.2,
        )
        row = dict(
            repeat=0,
            elapsed_s=0.2,
            output_tokens=2,
            hit_tokens=0,
            records=[record],
            waves=1,
        )
        (engine / "native.json").write_text(
            json.dumps(dict(rounds=[row], profiled=False))
        )
        for rank in range(tp):
            (engine / f"candidate-rank{rank}.json").write_text(
                json.dumps(dict(status="PASS", rounds=[row], runner_calls=[]))
            )
        return engine

    def test_single_rank(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root, 1)
            self.assertEqual(compare(root)["token_comparison"], "PASS")

    def test_peer_disagreement(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            engine = self.fixture(root, 2)
            path = engine / "candidate-rank1.json"
            row = json.loads(path.read_text())
            row["rounds"][0]["records"][0]["token_ids"] = [1, 3]
            path.write_text(json.dumps(row))
            with self.assertRaisesRegex(AssertionError, "TP disagreement"):
                compare(root)


class ReusedBaselineTests(unittest.TestCase):
    fixture = ReportTests.fixture

    def test_reuse_and_reject_configuration_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate, baseline = root / "candidate", root / "baseline"
            trace = dict(
                sessions=[dict(calls=[dict(prompt_ids=[1] * 4, output_tokens=2)])]
            )
            for capsule in (candidate, baseline):
                capsule.mkdir()
                self.fixture(capsule, 1)
                (capsule / "trace.json").write_text(json.dumps(trace))
                (capsule / "engine/complete.json").write_text(
                    json.dumps(dict(status="PASS", profiled=False))
                )
                (capsule / "run").mkdir()
                (capsule / "run/exit.txt").write_text("0\n")
            result = compare_candidate(candidate, [baseline])
            self.assertEqual(result["rounds"][0]["reused_native_samples"], 1)
            path = baseline / "engine/config.json"
            path.write_text(json.dumps(dict(model="different", tensor_parallel_size=1)))
            with self.assertRaisesRegex(ValueError, "configuration/trace differs"):
                compare_candidate(candidate, [baseline])
