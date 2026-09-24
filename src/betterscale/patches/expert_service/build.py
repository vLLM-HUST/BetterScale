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


def emit(output: Path, *, draft_layers: int = 0, placement='layer', owners=2, sources_per_wave=1, route_plan_min_rows=0):
    if type(draft_layers) is not int or draft_layers not in (0, 1):
        raise ValueError('Qwen35 supports exactly zero or one physical MTP layer')
    if placement not in ('layer','expert') or owners not in (2,4):
        raise ValueError('Build requires layer/expert placement and E2/E4')
    if type(sources_per_wave) is not int or not 1 <= sources_per_wave <= 7:
        raise ValueError('sources_per_wave must be an integer in [1, 7]')
    if type(route_plan_min_rows) is not int or not 0 <= route_plan_min_rows <= 4096:
        raise ValueError('route_plan_min_rows must be an integer in [0, 4096]')
    if route_plan_min_rows and (placement != 'layer' or sources_per_wave != 1):
        raise ValueError('Native route plans currently require layer placement and cap1')
    root = Path(__file__).parent
    abi = json.loads((root / 'abi-template.json').read_text())
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = output / 'source'
    shutil.copytree(root / 'native', source)
    if route_plan_min_rows:
        plan_header = source / 'route_plan.hpp'
        plan_header.write_text(plan_header.read_text().replace(
            'constexpr int MIN_ROWS = 0;', f'constexpr int MIN_ROWS = {route_plan_min_rows};'))
        abi.update(client_route_plan='native-v2-cap1', client_route_plan_min_rows=route_plan_min_rows)
        source_bytes = ((abi['payload_words']*4 + abi['rows']*2048*2 + (2<<20)-1)//(2<<20))*(2<<20)
        if (50176 + 4096*2048//2 + 4096*8 + 512)*4 > source_bytes:
            raise ValueError('Native route plan does not fit the exported source window')
    header = source / 'persistent_protocol.hpp'
    text = header.read_text()
    old = 'constexpr int SOURCES_PER_WAVE = 1;'
    if text.count(old) != 1:
        raise RuntimeError('packaged admission geometry changed unexpectedly')
    header.write_text(text.replace(old, f'constexpr int SOURCES_PER_WAVE = {sources_per_wave};'))
    abi['sources_per_wave'] = sources_per_wave
    if draft_layers:
        header = source / 'persistent_protocol.hpp'
        text = header.read_text()
        old = 'LOCAL_EXPERTS = 256, LAYERS = 40, GROUPS = 256'
        if text.count(old) != 1:
            raise RuntimeError('packaged layer geometry changed unexpectedly')
        header.write_text(text.replace(old, old.replace('LAYERS = 40', 'LAYERS = 41')))
        abi['layer_count'] = 41
    if placement == 'expert':
        local=256//owners
        header=source/'persistent_protocol.hpp'
        text=header.read_text().replace('LOCAL_EXPERTS = 256',f'LOCAL_EXPERTS = {local}').replace('GROUPS = 256',f'GROUPS = {local}')
        header.write_text(text)
        client=source/'client_kernel.cpp'
        client.write_text(client.read_text().replace('COLLECT_OWNERS=1,COLLECT_EXPERTS=256',f'COLLECT_OWNERS={owners},COLLECT_EXPERTS={local}'))
        abi.update(placement=placement,expert_owners=owners,local_experts=local,combined_return=False,pipelined_client_collect=True)
    # Both return modes join the complete DOWN/SEND chain in this closure.
    abi['early_return'] = False
    (output / 'abi.json').write_text(json.dumps(abi, indent=2) + '\n')
    return output


def build(output: Path, *, draft_layers=0, emit_only=False, placement='layer', owners=2, sources_per_wave=1, route_plan_min_rows=0):
    output = emit(output, draft_layers=draft_layers, placement=placement, owners=owners, sources_per_wave=sources_per_wave, route_plan_min_rows=route_plan_min_rows)
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
    p.add_argument('--placement', choices=('layer','expert'), default='layer')
    p.add_argument('--owners', type=int, choices=(2,4), default=2)
    p.add_argument('--sources-per-wave', type=int, choices=range(1, 8), default=1,
                   help='Opportunistic compatible-source cap; 1 preserves the qualified control')
    p.add_argument('--route-plan-min-rows', type=int, default=0,
                   help='Experimental native client routing threshold; 0 disables, layer/cap1 only')
    p.add_argument('--emit-only', action='store_true')
    args = p.parse_args()
    print(build(args.output, draft_layers=args.draft_layers, emit_only=args.emit_only, placement=args.placement, owners=args.owners, sources_per_wave=args.sources_per_wave, route_plan_min_rows=args.route_plan_min_rows))


if __name__ == '__main__':
    main()
