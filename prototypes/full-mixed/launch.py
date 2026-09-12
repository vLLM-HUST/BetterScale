"""Lease/admit a bounded subset; monitor only selected cards, preserve others."""
import argparse,fcntl,json,os,re,shutil,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,'/workspace/my-ascend-workspace/runs/query-gang/20260911-lhtb-long-real-gang-v1/harness')
from probe_host_npus import parse_devices
from supervise import descendants,stop_group,group_members
p=argparse.ArgumentParser();p.add_argument('--devices',default='0');p.add_argument('--output',type=Path,required=True);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
devices={int(x) for x in a.devices.split(',')};assert devices and devices<=set(range(8))
a.output.mkdir(parents=True,exist_ok=False)
lock=open('/root/tp8.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
known_host_owners=set()
def inspect(owned=None):
 text=subprocess.check_output(['npu-smi','info'],text=True,timeout=20);readings=parse_devices(text)
 owners=set()
 for line in text.split('Process id',1)[1].splitlines():
  cells=[x.strip() for x in line.split('|')[1:-1]]
  if cells and re.fullmatch(r'\d+\s+\d+',cells[0]) and int(cells[0].split()[0]) in devices:
   assert len(cells)>=5 and cells[4].isdigit()
   host_pid,local_pid=int(cells[1]),int(cells[4])
   if owned is not None and local_pid in owned:known_host_owners.add(host_pid)
   if local_pid==0 and host_pid in known_host_owners:continue
   owners.add(local_pid or host_pid)
 return text,readings,owners
text,readings,owners=inspect();(a.output/'admission.txt').write_text(text)
assert not owners,'BUSY selected devices'
for d in devices:
 s=readings[d];assert s.hbm_used_mb<=4096 and s.aicore_percent==0 and re.search(rf'\|\s*{d}\s+910B2\s*\|\s*OK',text),f'BUSY/UNHEALTHY {d}'
command=a.command[1:] if a.command[:1]==['--'] else a.command
source=a.output/'source';source.mkdir()
for f in Path(__file__).parent.glob('*.py'):shutil.copyfile(f,source/f.name)
command=[str(source/'probe.py') if x.endswith('/full-mixed/probe.py') else x for x in command]
os.environ['PYTHONPATH']=str(source)+os.pathsep+os.environ.get('PYTHONPATH','')
log=open(a.output/'run.log','w');child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd=a.output)
try:
 deadline=time.monotonic()+900
 while child.poll() is None:
  if time.monotonic()>deadline:raise TimeoutError('bounded probe exceeded 900s')
  text,_,owners=inspect(descendants(child.pid)|group_members(child.pid));(a.output/'latest.txt').write_text(text);foreign=owners-(descendants(child.pid)|group_members(child.pid))
  if foreign:
   (a.output/'foreign.txt').write_text(text);raise RuntimeError(f'foreign owner on selected cards: {foreign}')
  time.sleep(3)
 rc=child.returncode
finally:
 stop_group(child);log.close();(a.output/'release.txt').write_text(subprocess.check_output(['npu-smi','info'],text=True,timeout=20))
(a.output/'exit.txt').write_text(str(rc)+'\n');print('probe exit',rc,flush=True);sys.exit(rc)
