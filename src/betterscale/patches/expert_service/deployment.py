"""Owned multi-role startup and concurrent IPC drain, without host lease policy.

The caller admits the selected devices and holds an external bounded watchdog.
Only loopback HTTP and task-private control sockets are exposed here.
"""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request
from .config import ServiceConfig


def http(port, path, body=None, timeout=10):
    request = urllib.request.Request(f'http://127.0.0.1:{port}{path}',
        data=None if body is None else json.dumps(body).encode(),
        headers={'Content-Type': 'application/json'})
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=timeout) as response:
        value = response.read().decode()
        return value if path == '/metrics' or not value else json.loads(value)


def capture_sizes(max_seqs, mtp_tokens):
    width = mtp_tokens + 1
    return sorted({width*n for n in (1,2,4,8,16,32) if n <= max_seqs} | {width*max_seqs})


class Deployment:
    """One owned foreground group; native Workers own all model execution."""
    def __init__(self, args):
        self.args = args
        self.output = Path(args.output).resolve()
        self.devices = args.devices.split(',')
        if (len(self.devices) != args.sources+args.owners or len(set(self.devices)) != len(self.devices)
                or any(not d.isdigit() or int(d) not in range(8) for d in self.devices)):
            raise ValueError('devices must name exactly one distinct physical NPU per role')
        if not 1 <= args.max_seqs <= 32 or args.mtp_tokens not in (0,2):
            raise ValueError('requires <=32 requests and MTP0 or MTP2')
        if not 1 <= args.port_base <= 65536-args.sources or not 0 < args.kv_gib <= 48:
            raise ValueError('invalid loopback ports or KV budget')
        if not 0 < args.lifetime <= 10800:
            raise ValueError('invalid bounded service lifetime')
        if os.environ.get('BETTERSCALE_EXPERT_EXTERNAL_WATCHDOG') != '1':
            raise ValueError('requires admitted devices and an actual bounded external watchdog')
        self.synthetic_length=getattr(args,'synthetic_acceptance_length',None)
        if self.synthetic_length is not None and (args.mtp_tokens!=2 or not 1 <= self.synthetic_length <= 3):
            raise ValueError('benchmark acceptance length requires MTP2 and AL in [1,3]')
        self.service = ServiceConfig(str(self.output/'control'), str(Path(args.build).resolve()),
            args.owners, args.sources, 0, int(bool(args.mtp_tokens)), args.qualification)
        self.service.validate(); self.service.check_build()
        self.ports = list(range(args.port_base,args.port_base+args.sources))
        self.children = []; self.ready = False; self.closed = False; self.stopping = False
        self.deadline = time.monotonic()+args.lifetime
        self.receipt = dict(status='STARTED', commands=[], model=str(args.model),
            sources=args.sources, owners=args.owners, mtp_tokens=args.mtp_tokens,
            sampling_policy='real' if self.synthetic_length is None else 'benchmark-only synthetic',
            synthetic_acceptance_length=self.synthetic_length,
            package_root=str(Path(__file__).parents[2]), devices=self.devices)

    def save(self):
        temporary=self.output/'deployment.json.tmp'
        temporary.write_text(json.dumps(self.receipt,indent=2)+'\n')
        temporary.replace(self.output/'deployment.json')

    def launch(self, role, command, device):
        self.receipt['commands'].append(dict(role=role,argv=command,device=device))
        with (self.output/f'{role}.log').open('w') as log:
            child = subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,
                env=dict(os.environ, ASCEND_RT_VISIBLE_DEVICES=device, VLLM_SERVER_DEV_MODE='1'))
        self.children.append(child)

    def check(self):
        if self.stopping:
            raise InterruptedError('expert deployment startup cancelled')
        if time.monotonic() >= self.deadline:
            raise TimeoutError('bounded expert deployment lifetime exceeded')
        for child in self.children:
            if child.poll() is not None:
                raise RuntimeError(f'expert deployment child exited: pid={child.pid}, rc={child.returncode}')

    def wait(self, predicate):
        while not predicate():
            self.check(); time.sleep(1)

    def healthy(self, port):
        try:
            http(port,'/health',timeout=2)
            return True
        except OSError:
            return False

    def rpc(self, port, method):
        return http(port,'/collective_rpc',dict(method=method,timeout=180),timeout=200)['results'][0]

    def start(self):
        a = self.args
        self.output.mkdir(parents=True,exist_ok=False)
        control = self.output/'control'; control.mkdir(mode=0o700)
        try:
            for owner in range(a.owners):
                self.launch(f'expert{owner}',[sys.executable,'-m','betterscale.patches.expert_service.server',
                    '--model',str(a.model),'--control',str(control),'--build',self.service.build,
                    '--owners',str(a.owners),'--sources',str(a.sources),'--owner',str(owner),
                    '--draft-layers',str(int(bool(a.mtp_tokens))),'--receipt',str(self.output/f'expert{owner}.json')],
                    self.devices[a.sources+owner])
            self.wait(lambda:all((control/f'e{i}.sock').exists() for i in range(a.owners)))
            sizes = capture_sizes(a.max_seqs,a.mtp_tokens)
            for source,port in enumerate(self.ports):
                expert = dict(experimental=True,control=str(control),build=self.service.build,
                              owners=a.owners,sources=a.sources,source=source)
                if a.qualification:expert['qualification']=a.qualification
                command=[sys.executable,'-m','vllm.entrypoints.cli.main','serve',str(a.model),
                    '--host','127.0.0.1','--port',str(port),'--served-model-name','qwen35',
                    '--worker-cls','betterscale.worker.Worker','--tensor-parallel-size','1',
                    '--dtype','bfloat16','--max-model-len','262144','--max-num-seqs',str(a.max_seqs),
                    '--max-num-batched-tokens','4096','--kv-cache-memory-bytes',str(int(a.kv_gib*1024**3)),
                    '--enable-prefix-caching','--mamba-cache-mode','align','--async-scheduling',
                    '--limit-mm-per-prompt','{"image":0,"video":0}',
                    '--additional-config',json.dumps(dict(enable_cpu_binding=False,betterscale_experts=expert)),
                    '--seed','17','--generation-config','vllm','--shutdown-timeout','60',
                    '--safetensors-load-strategy','lazy','--compilation-config',json.dumps(dict(mode=0,
                        cudagraph_mode='FULL_DECODE_ONLY',cudagraph_capture_sizes=sizes,max_cudagraph_capture_size=max(sizes)))]
                if self.synthetic_length is not None:
                    command+=['--override-generation-config','{"temperature":0}']
                if a.mtp_tokens:
                    spec=dict(method='mtp',num_speculative_tokens=a.mtp_tokens)
                    if self.synthetic_length is not None:
                        spec.update(rejection_sample_method='synthetic',synthetic_acceptance_length=self.synthetic_length)
                    command+=['--speculative-config',json.dumps(spec)]
                self.launch(f'attention{source}',command,self.devices[source])
            self.save(); self.wait(lambda:all(self.healthy(port) for port in self.ports)); self.ready=True
            self.receipt['startup']=[self.rpc(port,'expert_receipt') for port in self.ports]
            self.receipt['status']='READY'; self.save()
            (self.output/'ready').write_text(json.dumps(dict(ports=self.ports))+'\n')
            return self
        except BaseException as error:
            self.receipt['status']='FAIL'; self.receipt['error']=repr(error)
            self.close()
            raise

    def close(self):
        if self.closed:return
        self.closed=True
        graceful=False
        if self.ready:
            try:
                self.receipt['final']=[self.rpc(port,'expert_receipt') for port in self.ports]
                with ThreadPoolExecutor(max_workers=len(self.ports)) as pool:
                    self.receipt['drain']=list(pool.map(lambda port:self.rpc(port,'close_expert_service'),self.ports))
                graceful=True
            except Exception as error:
                self.receipt['status']='FAIL'; self.receipt['drain_error']=repr(error)
        # A successful EOF lets owners write receipts naturally. On partial
        # startup/failure, terminate every owned role under the external guard.
        for i,child in enumerate(self.children):
            if child.poll() is None and (not graceful or i>=self.args.owners):
                child.send_signal(signal.SIGINT)
        for child in self.children:
            try:child.wait(timeout=90)
            except subprocess.TimeoutExpired:
                child.terminate()
                try:child.wait(timeout=30)
                except subprocess.TimeoutExpired:child.kill();child.wait(timeout=10)
        codes=[child.returncode for child in self.children]; self.receipt['exit_codes']=codes
        if any(codes):self.receipt['status']='FAIL'
        if graceful and self.receipt['status']!='FAIL':
            try:
                for owner in range(self.args.owners):
                    server=json.loads((self.output/f'expert{owner}.json').read_text())
                    expected={str(source):r['peer_generations'][str(owner)] for source,r in enumerate(self.receipt['drain'])}
                    if server['status']!='PASS' or server['completed']!=expected:
                        raise RuntimeError(f'expert {owner} generation mismatch')
                    if server['server_launch_mode']!='direct' or server['persistent_kernel_launches']!=2 or server['persistent_graph_launches']!=0:
                        raise RuntimeError(f'expert {owner} did not use the direct resident pair')
                self.receipt['status']='PASS'
            except Exception as error:
                self.receipt['status']='FAIL';self.receipt['validation_error']=repr(error)
        self.save()

    def __enter__(self):return self.start()

    def __exit__(self, kind, error, traceback):
        if error is not None:self.receipt['status']='FAIL';self.receipt['error']=repr(error)
        self.close()
        if error is None and self.receipt['status']!='PASS':
            raise RuntimeError('expert deployment did not drain cleanly; see deployment.json')


def add_arguments(parser):
    parser.add_argument('model',type=Path)
    parser.add_argument('--output',type=Path,required=True,help='new task-private output/control directory')
    parser.add_argument('--build',required=True)
    parser.add_argument('--devices',required=True)
    parser.add_argument('--sources',type=int,required=True)
    parser.add_argument('--owners',type=int,required=True)
    parser.add_argument('--port-base',type=int,default=32510)
    parser.add_argument('--max-seqs',type=int,default=16)
    parser.add_argument('--kv-gib',type=float,default=32)
    parser.add_argument('--mtp-tokens',type=int,choices=(0,2),default=2)
    parser.add_argument('--qualification')
    parser.add_argument('--synthetic-acceptance-length',type=float,help='benchmark ONLY; never use generated output as quality evidence')
    parser.add_argument('--lifetime',type=int,default=10800)


def run(args):
    group=Deployment(args)
    def stop(*_):
        group.stopping=True
    previous={sig:signal.signal(sig,stop) for sig in (signal.SIGINT,signal.SIGTERM)}
    try:
        with group:
            while not group.stopping and not (group.output/'stop').exists():
                group.check();time.sleep(1)
    finally:
        for sig,handler in previous.items():signal.signal(sig,handler)
