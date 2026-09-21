"""Bounded TP2 MTP/APC HTTP acceptance and synthetic warm-cohort timings."""
import concurrent.futures
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import urllib.request

from service_probe import request
from count_policy import MTP_TOKENS, WIDTH, SPEC_CAPACITIES


def main():
    root = Path(os.environ['CAPSULE'])
    raw = json.loads((root/'prompt.json').read_text())
    tokens = raw['prompt_token_ids'] if isinstance(raw,dict) else raw
    assert isinstance(tokens,list) and len(tokens)*10 >= 4096
    assert all(isinstance(t,int) and t >= 0 for t in tokens)
    port = 32492
    url = f'http://127.0.0.1:{port}'
    command = [sys.executable,'-m','vllm.entrypoints.cli.main','serve',os.environ['QWEN_MODEL_PATH'],
        '--host','127.0.0.1','--port',str(port),'--served-model-name','qwen27',
        '--tensor-parallel-size','2','--distributed-executor-backend','mp',
        '--worker-cls','betterscale.worker.Worker','--dtype','bfloat16',
        '--max-model-len','4096','--max-num-seqs','8','--max-num-batched-tokens','2048',
        '--kv-cache-memory-bytes',str(6*1024**3),'--seed','17',
        '--enable-prefix-caching','--mamba-cache-mode','align','--async-scheduling',
        '--shutdown-timeout','60','--limit-mm-per-prompt','{"image":0,"video":0}',
        '--additional-config','{"enable_cpu_binding":false}',

        '--compilation-config',json.dumps(dict(cudagraph_mode='FULL',cudagraph_capture_sizes=sorted(set((SPEC_CAPACITIES if MTP_TOKENS else (1,2,4,8))+(16,32,64,128,256,512,1024,1536,2048))),max_cudagraph_capture_size=2048))]
    native = os.environ.get('NATIVE_BASELINE') == '1'
    if native:
        command[-1] = json.dumps(dict(cudagraph_mode='FULL_AND_PIECEWISE',
            cudagraph_capture_sizes=[WIDTH*n for n in (1,2,4,8)],max_cudagraph_capture_size=8*WIDTH))
    if MTP_TOKENS:
        command += ['--speculative-config',json.dumps(dict(method='mtp',num_speculative_tokens=MTP_TOKENS))]
    boundary_apc = os.environ.get('MTP_APC_BOUNDARY') == '1'
    if boundary_apc:
        assert MTP_TOKENS == 2 and not native
        command += ['--scheduler-cls','apc_boundary.BoundaryScheduler']
    profiling = os.environ.get('MTP_PROFILE') == '1'
    if profiling:
        command += ['--profiler-config',json.dumps(dict(profiler='torch',torch_profiler_dir=str(root/'profiles'),ignore_frontend=True))]
    receipt = dict(mtp_tokens=MTP_TOKENS,status='STARTING',arm='native-abi-only' if native else 'owned-mtp-full',
                   command=command,rows=[],benchmarks=[])
    server_env = os.environ.copy()
    if native:
        server_env.pop('LD_PRELOAD',None)
    if boundary_apc:
        server_env['VLLM_SERVER_DEV_MODE'] = '1'  # localhost-only diagnostic API
    path = root/'receipt.json'
    path.write_text(json.dumps(receipt,indent=2))
    log = (root/'server.log').open('w')
    server = subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=server_env)
    cancelled = False
    def cancel(signum, frame):
        nonlocal cancelled
        cancelled = True
        raise KeyboardInterrupt(f'service probe cancelled by signal {signum}')
    # Admission owns this harness's group, while the server has a separate one.
    # Route cancellation through finally instead of orphaning the NPU workers.
    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, cancel)
    try:
        deadline = time.monotonic()+1200
        while True:
            if server.poll() is not None:
                raise RuntimeError(f'server exited {server.returncode}')
            try:
                with urllib.request.urlopen(url+'/health',timeout=2) as response:
                    if response.status == 200:
                        break
            except Exception:
                pass
            if time.monotonic()>deadline:
                raise TimeoutError('model startup deadline')
            time.sleep(2)
        def run(prompt,budget=24,callback=None):
            result = request(url,prompt,budget,on_first_content=callback)
            receipt['rows'].append(result)
            path.write_text(json.dumps(receipt,indent=2))
            return result
        tokens = tokens * 10
        # Eagle/MTP reserves one trailing1536-token block. Prime a checkpoint
        # beyond two blocks; lookup also drops one full block, so replay must
        # itself contain at least two full blocks (3073 includes an uncached tail).
        receipt['checkpoint_primer'] = run(tokens[:3073],1)
        run(tokens[:32])
        first = run(tokens[:3073])
        repeat = run(tokens[:3073])
        receipt['repeat_text_equal'] = first['text'] == repeat['text']
        # Start more requests only after an existing request is decoding.
        started = threading.Event()
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            older = pool.submit(run,tokens[:65],64,started.set)
            if not started.wait(120):
                raise TimeoutError('first decoded token deadline')
            others = [pool.submit(run,tokens[:n]) for n in (33,97,257)]
            older.result()
            for future in others:
                future.result()
        if boundary_apc:
            from apc_verification import verify
            receipt['boundary_checks'] = verify(url,tokens,run,root)
            path.write_text(json.dumps(receipt,indent=2))
        # Cold shape/JIT effects are excluded by one full warm cohort per C.
        for concurrency in (1,4,8):
            lengths = [3073 + 8*i for i in range(concurrency)]
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
                for repetition in range(4):
                    start = time.perf_counter()
                    futures = [pool.submit(request,url,tokens[:n],64) for n in lengths]
                    results = [future.result() for future in futures]
                    elapsed = time.perf_counter()-start
                    if repetition:
                        receipt['benchmarks'].append(dict(concurrency=concurrency,
                            repetition=repetition,elapsed_s=elapsed,
                            output_tokens=64*concurrency,output_tokens_per_second=64*concurrency/elapsed,
                            latency_s=[r['latency_s'] for r in results],ttft_s=[r['ttft_s'] for r in results]))
                        path.write_text(json.dumps(receipt,indent=2))
        with urllib.request.urlopen(url+'/metrics',timeout=5) as response:
            metrics = response.read().decode().splitlines()
        receipt['metrics'] = [line for line in metrics if not line.startswith('#') and
            any(key in line for key in ('prefix_cache_hits_total','prefix_cache_queries_total',
                                        'spec_decode_num_drafts_total','spec_decode_num_draft_tokens_total',
                                        'spec_decode_num_accepted_tokens'))]
        hits = [float(line.rsplit(' ',1)[1]) for line in receipt['metrics']
                if line.startswith('vllm:prefix_cache_hits_total')]
        if not hits or sum(hits)<=0:
            raise RuntimeError('APC enabled but no actual prefix hit; qualification incomplete')
        if profiling:
            def control(path):
                req = urllib.request.Request(url+path,data=b'',method='POST')
                with urllib.request.urlopen(req,timeout=120) as response:
                    assert response.status==200
            control('/start_profile')
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                futures = [pool.submit(request,url,tokens[:3073+8*i],128) for i in range(4)]
                for f in futures: f.result()
            control('/stop_profile')
            # Same joining workload used during smoke; collect only six steps.
            started = threading.Event()
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                lead = pool.submit(request,url,tokens[:65],128,on_first_content=started.set)
                if not started.wait(120): raise TimeoutError('profile lead')
                control('/start_profile')
                joins = [pool.submit(request,url,tokens[:n],32) for n in (33,97,257)]
                for f in joins: f.result()
                lead.result()
            control('/stop_profile')
        receipt['status'] = 'PASS'
    except BaseException as exc:
        receipt.update(status='FAIL',error=repr(exc))
        raise
    finally:
        if server.poll() is None:
            os.killpg(server.pid,signal.SIGTERM)
            try:
                server.wait(timeout=20 if cancelled else 90)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid,signal.SIGKILL); server.wait(timeout=20)
        receipt['server_exit'] = server.returncode
        path.write_text(json.dumps(receipt,indent=2))
        log.close()


if __name__ == '__main__':
    main()
