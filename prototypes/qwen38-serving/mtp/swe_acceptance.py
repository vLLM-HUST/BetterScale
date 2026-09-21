"""Frozen MTP2 SWE acceptance; source capsules stay immutable, no tool execution."""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request

from service_probe import request
from swe_service import replay


def command_for(source, context=8192):
    command = json.loads((source / 'receipt.json').read_text())['command'][:]
    assert json.loads(command[command.index('--speculative-config') + 1]) == {
        'method': 'mtp', 'num_speculative_tokens': 2}
    command[0] = sys.executable
    command[command.index('--max-model-len') + 1] = str(context)
    if '--profiler-config' in command:
        index = command.index('--profiler-config')
        del command[index:index + 2]
    command.append('--enable-prompt-tokens-details')
    return command


def main():
    p = argparse.ArgumentParser()
    p.add_argument('source', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--arm', choices=['baseline', 'candidate'], required=True)
    p.add_argument('--repeat', type=int, choices=[0, 1], required=True)
    a = p.parse_args()
    root = a.output
    root.mkdir(parents=True, exist_ok=False)
    trace = json.loads((root.parent.parent / 'trace.json').read_text())
    sessions = trace['sessions']
    assert len(sessions) == 8
    assert all(0 < len(c['prompt_ids']) + c['output_tokens'] <= 8192
               for s in sessions for c in s['calls'])
    command = command_for(a.source)
    port = command[command.index('--port') + 1]
    url = f'http://127.0.0.1:{port}'
    receipt = dict(status='STARTING', arm=a.arm, command=command, rounds=[],
        devices=os.environ['ASCEND_RT_VISIBLE_DEVICES'], prefix_caching=True,
        cache_start='empty before each cohort', mtp_tokens=2,
        source_capsule=str(a.source),
        scope='Original-history closed-loop SWE inputs and recorded output budgets; no tools or task-accuracy claim. Native baseline includes required SD/V1 ABI bridge; candidate includes lookahead APC and device continuation. No profiler in timing.')
    path = root / 'receipt.json'
    def save():
        path.write_text(json.dumps(receipt, indent=2) + '\n')
    save()
    def cancel(signum, frame):
        raise KeyboardInterrupt(f'cancelled by signal {signum}')
    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, cancel)
    with (root / 'server.log').open('w') as log:
        env = os.environ.copy()
        env.update(VLLM_SERVER_DEV_MODE='1', MTP_PROFILE='0', CAPSULE=str(root))
        if a.arm == 'baseline':
            env.pop('LD_PRELOAD', None)
        server = subprocess.Popen(command, env=env, stdout=log,
                                  stderr=subprocess.STDOUT, start_new_session=True)
        try:
            deadline = time.monotonic() + 1200
            while True:
                if server.poll() is not None:
                    raise RuntimeError(f'server exited {server.returncode}')
                try:
                    with urllib.request.urlopen(url + '/health', timeout=2):
                        break
                except Exception:
                    if time.monotonic() > deadline:
                        raise TimeoutError('model startup')
                    time.sleep(2)
            # Whole first turns warm all used inference paths, not just a tiny decode.
            with concurrent.futures.ThreadPoolExecutor(8) as pool:
                list(pool.map(lambda s: request(url, s['calls'][0]['prompt_ids'],
                                                s['calls'][0]['output_tokens']), sessions))
            for c in ((1, 2, 4, 8) if a.repeat == 0 else (8, 4, 2, 1)):
                with urllib.request.urlopen(urllib.request.Request(
                        url + '/reset_prefix_cache', method='POST'), timeout=60) as response:
                    assert response.status == 200
                cohort = replay(url, sessions, c)
                hits = [r['usage']['prompt_tokens_details']['cached_tokens']
                        for r in cohort['requests']]
                assert sum(hits) > 0, 'APC has no observed reuse'
                # First admitted call cannot reuse an uncleared preceding cohort.
                assert any(r['usage']['prompt_tokens_details']['cached_tokens'] == 0
                           for r in cohort['requests'] if r['turn'] == 0)
                cohort['cached_prompt_tokens'] = sum(hits)
                receipt['rounds'].append(cohort)
                save()
                print(a.arm, a.repeat, 'C', c, 'tokens/s',
                      cohort['output_tokens'] / cohort['elapsed_s'], flush=True)
            with urllib.request.urlopen(url + '/metrics', timeout=10) as response:
                receipt['metrics'] = [l for l in response.read().decode().splitlines()
                    if not l.startswith('#') and any(k in l for k in
                        ('spec_decode_num_', 'prefix_cache_hits_total', 'prefix_cache_queries_total'))]
            receipt['status'] = 'PASS'
        except BaseException as exc:
            receipt.update(status='FAIL', error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            if server.poll() is None:
                os.killpg(server.pid, signal.SIGINT)
            try:
                server.wait(timeout=90)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait()
                receipt['shutdown_timeout'] = True
            receipt['server_exit_code'] = server.returncode
            if server.returncode != 0:
                receipt['status'] = 'FAIL'
            save()
    if receipt['status'] != 'PASS':
        raise RuntimeError('service qualification failed')


if __name__ == '__main__':
    main()
