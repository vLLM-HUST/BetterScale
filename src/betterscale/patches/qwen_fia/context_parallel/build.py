"""CPU-only, task-owned CANN build; never modifies an installed runtime."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

p = argparse.ArgumentParser()
p.add_argument('output', type=Path)
p.add_argument('--cann', type=Path, default=Path(os.environ.get('ASCEND_HOME_PATH', '/usr/local/Ascend/cann-9.0.1')))
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
build = a.output.resolve()
source = Path(__file__).resolve().parent
subprocess.run([sys.executable, str(source / 'prepare.py'), '--output', str(build), '--cann', str(a.cann)], check=True)
k = a.cann / 'opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/fused_infer_attention_score'
compiler = a.cann / 'bin/bisheng'
commands = [
    [str(compiler), '-std=c++17', '-O3', '-c', '--asc-aicore-lang', str(build/'kernel.cpp'),
     '--npu-arch=dav-2201', '-fPIC', '-Wno-macro-redefined', '-Wno-ignored-attributes', '-Wno-unused-value',
     '--cce-auto-sync=off', '-mllvm', '-cce-vf-remove-membar=false',
     '-mllvm', '-cce-aicore-hoist-movemask=false', '-mllvm', '-cce-aicore-dcci-insert-for-scalar=false',
     '-I'+str(build), '-I'+str(k), '-I'+str(a.cann/'include'), '-o', str(build/'kernel.o')],
    [str(compiler), '-shared', str(build/'kernel.o'), '-L'+str(a.cann/'lib64'),
     '-lascendc_runtime', '-lruntime', '-lascendcl', '-Wl,-rpath,'+str(a.cann/'lib64'),
     '-o', str(build/'libbs_fia_cp.so')],
]
(build/'commands.json').write_text(json.dumps(commands, indent=2)+'\n')
for command in commands:
    subprocess.run(command, check=True, timeout=600)
artifact=build/'libbs_fia_cp.so'
(build/'build.json').write_text(json.dumps(dict(cann=str(a.cann.resolve()),
    artifact=artifact.name, sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
    status='compiled-only-not-device-qualified'), indent=2)+'\n')
print(artifact)
