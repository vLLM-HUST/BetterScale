"""CPU-only build of the scoped, pinned native FIA host adapter."""

import argparse
import os
from pathlib import Path
import subprocess

p = argparse.ArgumentParser()
p.add_argument("output", type=Path)
a = p.parse_args()
root = Path(os.environ["ASCEND_HOME_PATH"])
a.output.parent.mkdir(parents=True, exist_ok=True)
subprocess.run(
    [
        "c++",
        "-shared",
        "-fPIC",
        "-O2",
        "-std=c++17",
        f"-I{root / 'include'}",
        str(Path(__file__).with_name("host_metadata.cpp")),
        f"-L{root / 'lib64'}",
        "-lopapi",
        "-ldl",
        "-o",
        str(a.output.resolve()),
    ],
    check=True,
    timeout=120,
)
print(a.output.resolve())
