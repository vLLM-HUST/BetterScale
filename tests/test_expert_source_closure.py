"""Source closure checks do not import Torch or touch NPUs."""
import json
from pathlib import Path
import tempfile
import unittest
from betterscale.patches.expert_service.build import emit


class SourceClosure(unittest.TestCase):
    def test_source_and_abi_are_packaged_without_prototype_dependencies(self):
        with tempfile.TemporaryDirectory() as d:
            out = emit(Path(d) / 'plain')
            abi = json.loads((out / 'abi.json').read_text())
            self.assertEqual(abi['layer_count'], 40)
            self.assertEqual(abi['rows'], 4096)
            self.assertEqual(abi['persistent_launch_timeout_us'], 0)
            self.assertTrue(abi['external_watchdog'])
            for p in (out / 'source').iterdir():
                self.assertNotIn('/workspace/', p.read_text())
                self.assertNotIn('/root/', p.read_text())
            self.assertTrue((out / 'source/client_kernel.cpp').is_file())
            with self.assertRaises(FileExistsError):
                emit(out)

    def test_mtp_adds_one_physical_layer_not_one_per_speculative_step(self):
        with tempfile.TemporaryDirectory() as d:
            plain = emit(Path(d) / 'plain')
            mtp = emit(Path(d) / 'mtp', draft_layers=1)
            self.assertEqual(json.loads((mtp / 'abi.json').read_text())['layer_count'], 41)
            for p in (plain / 'source').iterdir():
                other = mtp / 'source' / p.name
                if p.name == 'persistent_protocol.hpp':
                    self.assertEqual(p.read_text().replace('LAYERS = 40', 'LAYERS = 41'), other.read_text())
                else:
                    self.assertEqual(p.read_bytes(), other.read_bytes())
            with self.assertRaises(ValueError):
                emit(Path(d) / 'bad', draft_layers=2)


if __name__ == '__main__':
    unittest.main()
