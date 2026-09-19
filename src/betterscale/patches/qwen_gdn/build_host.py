"""Build only the framework host adapter against the unchanged pinned runtime."""

import argparse
from pathlib import Path
import subprocess
import sysconfig
import torch
import torch_npu

p = argparse.ArgumentParser()
p.add_argument("source", type=Path)
p.add_argument("output", type=Path)
a = p.parse_args()
a.output.parent.mkdir(parents=True, exist_ok=True)
t = Path(torch.__file__).parent
n = Path(torch_npu.__file__).parent
command = [
    "c++",
    "-std=c++17",
    "-shared",
    "-fPIC",
    "-O2",
    f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}",
    f"-I{t / 'include'}",
    f"-I{t / 'include/torch/csrc/api/include'}",
    f"-I{n / 'include'}",
    f"-I{sysconfig.get_path('include')}",
    str(a.source.resolve()),
    f"-L{t / 'lib'}",
    f"-L{n / 'lib'}",
    "-ltorch",
    "-ltorch_cpu",
    "-lc10",
    "-ltorch_npu",
    "-ldl",
    "-Wl,-rpath,$ORIGIN/../../../torch/lib",
    "-Wl,-rpath,$ORIGIN/../../../torch_npu/lib",
    "-o",
    str(a.output.resolve()),
]
subprocess.run(command, check=True, timeout=240)
print(a.output.resolve())
