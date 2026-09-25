"""The owned runtime must work with the external package explicitly forbidden."""
import os
from pathlib import Path
import subprocess
import sys


def test_qwen_state_without_external_livemodule():
    repo = Path(__file__).resolve().parents[1]
    # Can also exercise an extracted/installed wheel instead of the checkout.
    source = os.environ.get('BETTERSCALE_TEST_PACKAGE_ROOT', str(repo / 'src'))
    program = r'''
import importlib.abc
import pkgutil
import sys
import unittest

class RejectExternal(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if (fullname == 'livemodule' or fullname.startswith('livemodule.')
                or fullname in ('state', 'gdn_graph', 'gdn_candidates')):
            raise AssertionError('external dependency attempted: ' + fullname)

sys.meta_path.insert(0, RejectExternal())
sys.path.insert(0, sys.argv[1])
import betterscale.live as live
from pathlib import Path
assert Path(live.__file__).is_relative_to(Path(sys.argv[1]).resolve())
for module in pkgutil.walk_packages(live.__path__, live.__name__ + '.'):
    # This explicit numerical leaf needs the pinned donor Triton environment.
    # Root construction and every common runtime module stay donor-independent.
    if module.name != 'betterscale.live.llm.qwen35.gdn_candidates':
        importlib.import_module(module.name)
suite = unittest.defaultTestLoader.discover(sys.argv[2], pattern='test_state.py')
result = unittest.TextTestRunner(verbosity=1).run(suite)
assert result.testsRun == 14
assert not any(n == 'livemodule' or n.startswith('livemodule.') for n in sys.modules)
assert not any(n in sys.modules for n in ('vllm', 'torch_npu'))
sys.exit(not result.wasSuccessful())
'''
    result = subprocess.run(
        [sys.executable, '-I', '-c', program, source,
         str(repo / 'prototypes/qwen35-state-lanes')],
        cwd='/tmp', env={**os.environ, 'TORCH_DEVICE_BACKEND_AUTOLOAD': '0'},
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
