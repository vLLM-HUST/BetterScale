"""Task-local CPU/TCP Mooncake master and two clients; explicit loopback ports."""
from contextlib import contextmanager
import os,signal,socket,subprocess,sys,time
from pathlib import Path
CPU_ENV=Path('/workspace/pd-kv-layout-results/store-cpu-venv')

@contextmanager
def dram_store(output,port=55381):
    original_int=signal.getsignal(signal.SIGINT)
    if not callable(original_int) and original_int not in (signal.SIG_DFL,signal.SIG_IGN):
        original_int=signal.default_int_handler # native C handler has no Python handle
    # This isolated directory contains only the CPU Mooncake wheel and pip.
    # Do not accidentally import the container's older NPU Store wheel.
    site=CPU_ENV/'lib/python3.12/site-packages';sys.path.insert(0,str(site))
    import mooncake.store
    from mooncake.store import MooncakeDistributedStore
    assert Path(mooncake.store.__file__).resolve().is_relative_to(site)
    reservations=[]
    try:
        for number in (port,port+1,port+2,port+3,port+4,*range(port+20,port+28)):
            s=socket.socket();reservations.append(s);s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);s.bind(('127.0.0.1',number))
    finally:
        for s in reservations:s.close()
    env=os.environ.copy();env.pop('LD_PRELOAD',None);env.pop('LD_LIBRARY_PATH',None)
    output=Path(output);output.mkdir(exist_ok=True,parents=True)
    clients=[]
    with (output/'master.log').open('w') as log:
        proc=subprocess.Popen([str(CPU_ENV/'bin/mooncake_master'),'--rpc_address=127.0.0.1',f'--rpc_port={port}',
            '--enable_http_metadata_server=true','--http_metadata_server_host=127.0.0.1',f'--http_metadata_server_port={port+1}',
            '--metrics_host=127.0.0.1',f'--metrics_port={port+2}','--enable_metric_reporting=false',
            '--default_kv_lease_ttl=2000','--enable_offload=false'],env=env,stdout=log,stderr=log)
        try:
            deadline=time.monotonic()+15
            while True:
                if proc.poll() is not None:raise RuntimeError('DRAM Store master exited')
                try:
                    with socket.create_connection(('127.0.0.1',port),timeout=.2):break
                except OSError:
                    if time.monotonic()>deadline:raise TimeoutError('DRAM Store startup')
                    time.sleep(.1)
            for i in range(2):
                client=MooncakeDistributedStore();clients.append(client)
                rc=client.setup(f'127.0.0.1:{port+3+i}',f'http://127.0.0.1:{port+1}/metadata',
                    1024*1024**2 if i==0 else 0,64*1024**2,'tcp','',f'127.0.0.1:{port}')
                if rc!=0:raise RuntimeError(f'DRAM client setup: {rc}')
            (output/'master.pid').write_text(str(proc.pid)+'\n')
            # Native libraries may install a process-exit SIGINT handler. Keep
            # Python's unwind path so interruption closes this owned master.
            signal.signal(signal.SIGINT,signal.default_int_handler)
            yield tuple(clients)
        finally:
            try:
                for client in reversed(clients):client.close()
            finally:
                proc.terminate()
                try:proc.wait(timeout=10)
                except subprocess.TimeoutExpired:proc.kill();proc.wait(timeout=5)
                signal.signal(signal.SIGINT,original_int)
