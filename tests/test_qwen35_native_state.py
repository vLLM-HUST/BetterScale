import os
from pathlib import Path
import subprocess
import sys


def test_native_page_lifetime_hooks():
    path = Path(__file__).with_name("native_qwen35_state_fixture.py")
    subprocess.run(
        [sys.executable, str(path)],
        check=True,
        env=dict(os.environ, TORCH_DEVICE_BACKEND_AUTOLOAD="0", VLLM_PLUGINS=""),
    )
