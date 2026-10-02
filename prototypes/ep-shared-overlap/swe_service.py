"""Task-local hw112 launch, retaining downloaded historical point arguments."""
import argparse
import json
import os
from pathlib import Path
import sys

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--arm', choices=('serial', 'early'), required=True)
p.add_argument('--layout', choices=('tp', 'ep'), required=True)
p.add_argument('--target-only', action='store_true')
p.add_argument('--capacity', type=int, choices=(16, 32), default=16)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
root = Path('/workspace/overlap-swe')
point = json.loads((root / 'reference-points.json').read_text())[str(a.capacity)]
argv = point['configuration']['parameters']['server_command']
argv = [sys.executable if i == 0 else {'$PYTHON': sys.executable,
    '$MODEL': '/models/Qwen3.5-35B-A3B', '$PORT': '27720',
    '$SERVED_MODEL': 'qwen35-overlap'}.get(v, v) for i, v in enumerate(argv)]
argv[argv.index('--worker-cls') + 1] = ('ep_service_entry.Worker'
    if a.layout == 'ep' else 'betterscale.qwen35_worker.Worker')
i = argv.index('--additional-config') + 1
extra = json.loads(argv[i])
extra.update(multistream_overlap_shared_expert=a.arm == 'early',
             betterscale_shared_expert_overlap=a.arm == 'early')
argv[i] = json.dumps(extra)
if a.layout == 'ep':
    argv.append('--enable-expert-parallel')
if a.target_only:
    assert a.layout == 'ep', 'Target-only qualification is an EP-specific experiment'
    os.environ['OVERLAP_TARGET_ONLY'] = '1'
    argv[argv.index('--scheduler-cls') + 1] = 'ep_service_entry.TargetOnlyScheduler'
argv += ['--profiler-config', json.dumps(dict(profiler='torch',
    torch_profiler_dir=str(a.output / 'profile'), torch_profiler_with_stack=False,
    ignore_frontend=True, delay_iterations=2, max_iterations=5))]
(a.output / 'launch.json').write_text(json.dumps(dict(arm=a.arm, layout=a.layout,
    reference_point=point['id'], argv=argv, pid=os.getpid(), target_only=a.target_only,
    baseline_source=point['configuration']['parameters']['mod_revision'],
    runtime_adaptation='CANN9.1/post4, rebuilt native host37; full donor pins retained'), indent=2))
print('LAUNCH', a.layout, a.arm, a.output, flush=True)
os.execv(sys.executable, argv)
