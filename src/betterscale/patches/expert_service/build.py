"""Build the packaged persistent expert closure, without prototype codegen.

Compilation is explicit and CPU-only. It does not admit devices, initialize a
runtime, or launch an expert process. Deployment owns the external watchdog.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess


def emit(output: Path, *, draft_layers: int = 0):
    if type(draft_layers) is not int or draft_layers not in (0, 1):
        raise ValueError('Qwen35 supports exactly zero or one physical MTP layer')
    root = Path(__file__).parent
    abi = json.loads((root / 'abi-template.json').read_text())
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = output / 'source'
    shutil.copytree(root / 'native', source)
    if draft_layers:
        header = source / 'persistent_protocol.hpp'
        text = header.read_text()
        old = 'LOCAL_EXPERTS = 256, LAYERS = 40, GROUPS = 256'
        if text.count(old) != 1:
            raise RuntimeError('packaged layer geometry changed unexpectedly')
        header.write_text(text.replace(old, old.replace('LAYERS = 40', 'LAYERS = 41')))
        abi['layer_count'] = 41
    (output / 'abi.json').write_text(json.dumps(abi, indent=2) + '\n')
    return output


def build(output: Path, *, draft_layers=0, emit_only=False):
    output = emit(output, draft_layers=draft_layers)
    if not emit_only:
        root = Path(__file__).parent
        for unit, name, script in (
            ('persistent_vector.cpp', 'persistent_vector', 'build-vector.sh'),
            ('persistent_cube.cpp', 'persistent_cube', 'build-cube.sh'),
            ('client_kernel.cpp', 'queue_service', 'build-vector.sh'),
        ):
            env = dict(os.environ, OUTPUT_DIR=str(output),
                       SOURCE=str(output / 'source' / unit), OBJECT_NAME=name,
                       LAUNCH_SOURCE=str(output / 'source/launch.cpp'))
            with (output / f'{name}.build.log').open('w') as log:
                subprocess.run(['bash', str(root / script)], env=env,
                               stdout=log, stderr=subprocess.STDOUT, check=True)
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('output', type=Path)
    p.add_argument('--draft-layers', type=int, choices=(0, 1), default=0)
    p.add_argument('--emit-only', action='store_true')
    args = p.parse_args()
    print(build(args.output, draft_layers=args.draft_layers, emit_only=args.emit_only))


if __name__ == '__main__':
    main()
