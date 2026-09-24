"""Changing admitted requests must also capture the MTP verification envelope."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import unittest


class ServingCapacity(unittest.TestCase):
    def test_commands_preserve_default_and_expand_verification(self):
        path = Path(__file__).resolve().parents[1]/'prototypes/qwen35-moe-serving/probe.py'
        spec = importlib.util.spec_from_file_location('moe_probe_capacity', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for full in (False, True):
            for requests in (8, 16, 32):
                args = SimpleNamespace(model='/model', port=1234, worker='worker.Worker',
                    candidate_full=full, max_num_seqs=requests, gpu_memory_utilization=.95,
                    max_num_batched_tokens=4096, kv_cache_memory_bytes=21743271936)
                command = module.server_command(args)
                value = lambda flag: command[command.index(flag)+1]
                self.assertEqual(value('--max-model-len'), '262144')
                self.assertEqual(value('--max-num-seqs'), str(requests))
                self.assertEqual(value('--max-num-batched-tokens'), '4096')
                self.assertEqual(value('--gpu-memory-utilization'), '0.95')
                self.assertEqual(value('--kv-cache-memory-bytes'), '21743271936')
                graphs = json.loads(value('--compilation-config'))
                self.assertIn(requests*3, graphs['cudagraph_capture_sizes'])
                self.assertEqual(graphs['max_cudagraph_capture_size'], max(graphs['cudagraph_capture_sizes']))
                self.assertEqual(graphs['cudagraph_mode'], 'FULL' if full else 'FULL_AND_PIECEWISE')
