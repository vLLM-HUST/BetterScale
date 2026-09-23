"""The MoE head port must preserve the independent Q/K1024-token stride."""
import importlib.util
from pathlib import Path
import unittest


class GeometryPort(unittest.TestCase):
    def test_head_literal_does_not_change_qk_offset(self):
        path=Path(__file__).resolve().parents[1]/'prototypes/qwen35-moe-serving/stage_mixed.py'
        spec=importlib.util.spec_from_file_location('stage_mixed_test',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        source='kx = X + off + 1024; dest = token * 1024; g = token * 24; shape=(1,t,24,128)'
        self.assertEqual(module.replace_exact(source,'24','16'),
            'kx = X + off + 1024; dest = token * 1024; g = token * 16; shape=(1,t,16,128)')
        with self.assertRaises(ValueError):
            module.replace_exact('QK = 1024', '24', '16')
