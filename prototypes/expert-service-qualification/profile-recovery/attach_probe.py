"""Model-free attach-contract discriminator in the intended serving runtime."""
import json, os, subprocess, sys, time
from pathlib import Path
import msprof_helpers as h

ROOT = Path(sys.argv[1])
ROOT.mkdir(exist_ok=False)
WORKER = '''
import json, os, sys, time
from pathlib import Path
import torch, torch_npu
torch.npu.set_device(0)
x=torch.ones((128,128),device='npu',dtype=torch.bfloat16)
y=x@x
torch.npu.synchronize()
p=Path(sys.argv[1]); (p/'ready').write_text(str(os.getpid()))
deadline=time.monotonic()+90
while not (p/'stop').exists() and time.monotonic()<deadline:
    y=x@x
    torch.npu.synchronize()
    time.sleep(.01)
assert bool((y==128).all().item())
'''
(ROOT/'worker.py').write_text(WORKER)
results=[]
for dynamic in (False,True):
    run=ROOT/('dynamic' if dynamic else 'no-dynamic'); run.mkdir()
    env=dict(os.environ,ASCEND_RT_VISIBLE_DEVICES='0')
    env.pop('PROFILING_MODE',None)
    if dynamic: env['PROFILING_MODE']='dynamic'
    children=[]
    row={'dynamic':dynamic,'status':'STARTED'}
    with (run/'worker.log').open('w') as log:
        worker=subprocess.Popen([sys.executable,str(ROOT/'worker.py'),str(run)],env=env,stdout=log,stderr=subprocess.STDOUT)
    try:
        h.wait_for_file(run/'ready',timeout=60,watched_pid=worker.pid)
        # Controller and worker share /proc; use the worker's local PID, not a guessed host PID.
        command=h.msprof_command(Path('/usr/local/Ascend/cann-9.0.1/bin/msprof'),target_pid=worker.pid,output=run/'raw')
        command.remove('--hccl=on')
        (run/'raw').mkdir()
        stdout=run/'stdout.log'; stderr=run/'stderr.log'
        with stdout.open('w') as out,stderr.open('w') as err:
            p=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=out,stderr=err)
        c=dict(rank=0,process=p,output=str(run/'raw'),stdout=str(stdout),stderr=str(stderr),command=command)
        children.append(c)
        row['command']=command
        if not dynamic:
            row['returncode']=p.wait(timeout=20)
            assert row['returncode']!=0 and 'no valid pid values' in stdout.read_text()+stderr.read_text()
            row['status']='EXPECTED_REJECTION'
        else:
            h._send_command(c,'start'); h._wait_for_ack(children,command='start',timeout=30)
            time.sleep(2)
            h._send_command(c,'stop'); h._wait_for_ack(children,command='stop',timeout=30)
            h._send_command(c,'quit'); p.stdin.close()
            row['returncode']=p.wait(timeout=30); assert row['returncode']==0
            row['status']='CAPTURED'
    finally:
        h._terminate(children)
        (run/'stop').touch()
        try: row['worker_exit']=worker.wait(timeout=15)
        except subprocess.TimeoutExpired:
            worker.terminate(); row['worker_exit']=worker.wait(timeout=15)
        results.append(row)
        (ROOT/'receipt.json').write_text(json.dumps(results,indent=2)+'\n')
    assert row['worker_exit']==0
assert [r['status'] for r in results]==['EXPECTED_REJECTION','CAPTURED']
print('PASS: absent dynamic rejects; early dynamic + local PID attaches',flush=True)
