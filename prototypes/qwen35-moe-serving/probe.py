"""Bounded real-MTP and 256K HTTP qualification, not a performance benchmark."""
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


def server_command(args):
    command = [sys.executable, '-m', 'vllm.entrypoints.cli.main', 'serve', args.model,
        '--host', '127.0.0.1', '--port', str(args.port), '--served-model-name', 'qwen35-moe',
        '--tensor-parallel-size', '2', '--distributed-executor-backend', 'mp',
        '--worker-cls', args.worker, '--dtype', 'bfloat16', '--kv-cache-dtype', 'auto',
        '--max-model-len', '262144', '--max-num-seqs', str(getattr(args, 'max_num_seqs', 8)),
        '--max-num-batched-tokens', '8192',
        '--gpu-memory-utilization', str(getattr(args, 'gpu_memory_utilization', 0.90)), '--seed', '17', '--enable-prefix-caching',
        '--mamba-cache-mode', 'align', '--enable-prompt-tokens-details', '--async-scheduling',
        '--shutdown-timeout', '60', '--additional-config', '{"enable_cpu_binding":false}',
        '--limit-mm-per-prompt', '{"image":0,"video":0}', '--compilation-config',
        json.dumps({'cudagraph_mode': 'FULL_AND_PIECEWISE', 'cudagraph_capture_sizes': [3,6,12,24],
                    'max_cudagraph_capture_size': 24}),
        '--speculative-config', json.dumps({'method': 'mtp', 'num_speculative_tokens': 2})]
    if getattr(args, 'candidate_full', False):
        # Leave space for a2048-token APC block alongside live MTP rows.
        command[command.index('--max-num-batched-tokens')+1] = '4096'
        command[command.index('--compilation-config')+1] = json.dumps({
            'cudagraph_mode': 'FULL',
            'cudagraph_capture_sizes': [3,6,12,16,24,32,64,128,256,512,1024,1536,2048,4096],
            'max_cudagraph_capture_size': 4096})
        command += ['--scheduler-cls', 'apc_boundary.BoundaryScheduler']
    if getattr(args, 'max_num_seqs', 8) == 16:
        index = command.index('--compilation-config')+1
        config = json.loads(command[index])
        config['cudagraph_capture_sizes'] = sorted(set(config['cudagraph_capture_sizes']+([40,48] if getattr(args, 'candidate_full', False) else [48])))
        config['max_cudagraph_capture_size'] = max(config['cudagraph_capture_sizes'])
        command[index] = json.dumps(config)
    if getattr(args, 'max_num_batched_tokens', None) is not None:
        command[command.index('--max-num-batched-tokens')+1] = str(args.max_num_batched_tokens)
    if getattr(args, 'kv_cache_memory_bytes', None) is not None:
        command += ['--kv-cache-memory-bytes', str(args.kv_cache_memory_bytes)]
    return command


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--worker', default='native_worker.Worker')
    parser.add_argument('--port', type=int, default=32281)
    parser.add_argument('--candidate-full', action='store_true')
    parser.add_argument('--max-num-seqs', type=int, choices=(8,16), default=8)
    parser.add_argument('--max-num-batched-tokens', type=int)
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.90)
    parser.add_argument('--kv-cache-memory-bytes', type=int)
    parser.add_argument('--concurrency', type=int, choices=(4,16), default=4)
    parser.add_argument('--raw-stress', action='store_true',
                        help='Retain raw, forced-length stress and strict equality; not normal chat quality')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    output = args.output
    url = f'http://127.0.0.1:{args.port}'
    receipt = {'status': 'STARTED', 'requests': [], 'scope': 'functional, not benchmark',
               'speculation': 'real acceptance, never forced', 'context_tokens': 262144,
               'completion_policy': 'raw forced-length' if args.raw_stress else 'chat retrieval with EOS'}
    def save():
        (output / 'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    def cancel(signum, frame):
        raise KeyboardInterrupt(f'cancelled by {signum}')
    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, cancel)
    def get(path):
        with urllib.request.urlopen(url+path, timeout=5) as response:
            return response.read().decode()
    def post(path, payload):
        req = urllib.request.Request(url+path, data=json.dumps(payload).encode(),
                                     headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=3600) as response:
            data = response.read()
            return json.loads(data) if data else None
    command = server_command(args)
    receipt['command'] = command
    save()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if args.raw_stress:
        # Exact lengths use model-tokenizer text tokens, not random vocabulary IDs.
        filler = tokenizer.encode(' The archive contains ordinary historical records.', add_special_tokens=False)
        prefix = tokenizer.encode('Read the archive and answer the question at its end.\n', add_special_tokens=False)
        suffix = tokenizer.encode('\nQuestion: What is two plus three? Answer briefly: ', add_special_tokens=False)
        def prompt(length):
            n = length-len(prefix)-len(suffix)
            return prefix + (filler*((n+len(filler)-1)//len(filler)))[:n] + suffix
    else:
        # A model-native chat prompt, unlike the retained raw-completion stress case.
        # Place a unique record in the middle, not in the final question.
        marker = 'cobalt-seven-42'
        rendered = tokenizer.apply_chat_template([{'role':'user', 'content':
            'Read these records. Find the access code.\nBEGIN_RECORDS\n<FILLER>\nEND_RECORDS\n'
            'Reply with only the access code, without explanation.'}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        before, after = rendered.split('<FILLER>')
        prefix = tokenizer.encode(before, add_special_tokens=False)
        suffix = tokenizer.encode(after, add_special_tokens=False)
        filler = tokenizer.encode(' The archive contains ordinary historical records.', add_special_tokens=False)
        key = tokenizer.encode('\nThe access code is '+marker+'.\n', add_special_tokens=False)
        def prompt(length):
            n = length-len(prefix)-len(suffix)-len(key)
            left = n//2
            pad = (filler*((n+len(filler)-1)//len(filler)))[:n]
            return prefix + pad[:left] + key + pad[left:] + suffix
    def completion(tokens, name):
        started = time.monotonic()
        result = post('/v1/completions', {'model':'qwen35-moe', 'prompt':tokens,
            'max_tokens':64, 'temperature':0, 'ignore_eos':args.raw_stress, 'logprobs':5})
        (output / (name+'.json')).write_text(json.dumps(result, ensure_ascii=False)+'\n')
        assert result['usage']['prompt_tokens'] == len(tokens), result['usage']
        if args.raw_stress:
            assert result['usage']['completion_tokens'] == 64, result['usage']
            assert result['choices'][0]['finish_reason'] == 'length'
        else:
            assert 0 < result['usage']['completion_tokens'] <= 64, result['usage']
            assert result['choices'][0]['finish_reason'] == 'stop'
            assert result['choices'][0]['text'].strip() == marker, (
                'incorrect retrieval', name, result['choices'][0]['text'])
        return {'name':name, 'usage':result['usage'], 'seconds':time.monotonic()-started,
                'text':result['choices'][0]['text']}
    server = None
    try:
        with (output / 'server.log').open('w') as log:
            server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                      env=dict(os.environ, CAPSULE=str(output.resolve())),
                                      start_new_session=True)
        deadline = time.monotonic()+1800
        while time.monotonic() < deadline:
            if server.poll() is not None:
                raise RuntimeError(f'server exited {server.returncode}')
            try:
                get('/health')
                break
            except OSError:
                time.sleep(2)
        else:
            raise TimeoutError('server startup exceeded1800s')
        (output/'metrics-before.txt').write_text(get('/metrics'))
        chat = post('/v1/chat/completions', {'model':'qwen35-moe', 'messages':[
            {'role':'user','content':'What is two plus three? Reply with only the number.'}],
            'temperature':0, 'max_tokens':64, 'chat_template_kwargs':{'enable_thinking':False}})
        (output/'chat.json').write_text(json.dumps(chat,ensure_ascii=False)+'\n')
        assert '5' in chat['choices'][0]['message']['content'], chat['choices']
        for length in (8193, 32769, 131073, 262080):
            post('/reset_prefix_cache', {})
            tokens = prompt(length)
            cold = completion(tokens, f'cold-{length}')
            receipt['requests'].append(cold); save()
            warm = completion(tokens, f'warm-{length}')
            receipt['requests'].append(warm)
            cached = warm['usage'].get('prompt_tokens_details',{}).get('cached_tokens',0)
            assert cached > 0, ('no warm cache reuse', length)
            # Equality is a retained observation; divergence requires investigation,
            # not automatic dismissal as acceptable numerical noise.
            warm['same_text_as_cold'] = cold['text'] == warm['text']
            save()
            assert warm['same_text_as_cold'], ('cold/warm continuation differs', length)
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            rows = list(pool.map(lambda i:completion(prompt(4097+i*31),f'concurrent-{i}'),range(args.concurrency)))
        receipt['requests'].extend(rows)
        metrics = get('/metrics'); (output/'metrics-after.txt').write_text(metrics)
        import re
        def counter(name):
            return sum(float(m.group(1)) for m in re.finditer(
                rf'^vllm:{name}(?:\{{[^\n]*\}})?\s+([\d.eE+\-]+)',metrics,re.M))
        receipt['draft_tokens'] = counter('spec_decode_num_draft_tokens_total')
        receipt['accepted_tokens'] = counter('spec_decode_num_accepted_tokens_total')
        assert receipt['draft_tokens'] > 0 and receipt['accepted_tokens'] > 0, 'No actual MTP evidence'
        receipt['status'] = 'PASS'
    except BaseException as error:
        receipt.update(status='FAIL',error=f'{type(error).__name__}: {error}')
        raise
    finally:
        if server is not None:
            if server.poll() is None:
                os.killpg(server.pid,signal.SIGINT)
                try: server.wait(timeout=90)
                except subprocess.TimeoutExpired:
                    os.killpg(server.pid,signal.SIGTERM)
                    try: server.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        os.killpg(server.pid,signal.SIGKILL); server.wait()
                    receipt.update(status='FAIL',shutdown_timeout=True)
            receipt['server_exit_code'] = server.returncode
            if server.returncode != 0: receipt['status']='FAIL'
        save()
    if receipt['status'] != 'PASS': raise RuntimeError('qualification failed')


if __name__ == '__main__':
    main()
