"""CPU execution of unchanged donor running-grant AST, NOT a serving trace.

No vLLM/Torch/NPU imports. Allocation always succeeds; encoder, preemption,
connector and async completion are outside this controlled fixture. The
real waiting loop is NOT executed: we record its token-budget eligibility
and call the original alignment helper on the remaining grant.
"""
import argparse
import ast
import contextlib
import copy
import json
import time
from pathlib import Path
from types import SimpleNamespace as NS


def extract(path):
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Scheduler')
    schedule = copy.deepcopy(next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'schedule'))
    align = copy.deepcopy(next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_mamba_block_aligned_split'))
    # Preserve the complete leading statements and running loop verbatim.
    loops = [(i, n) for i, n in enumerate(schedule.body)
             if isinstance(n, ast.While) and 'self.running' in ast.unparse(n.test)]
    assert len(loops) == 1
    i, loop = loops[0]
    schedule.body = schedule.body[:i+1] + ast.parse(
        'return dict(grants=num_scheduled_tokens, remaining=token_budget)\n').body
    schedule.returns = align.returns = None
    for fn in (schedule, align):
        for arg in fn.args.args:
            arg.annotation = None
    module = ast.fix_missing_locations(ast.Module(body=[schedule, align], type_ignores=[]))
    env = {'time': time, 'PauseState': NS(PAUSED_ALL=1, UNPAUSED=0),
           'record_function_or_nullcontext': lambda _: contextlib.nullcontext()}
    exec(compile(module, str(path), 'exec'), env)
    return env['schedule'], env['_mamba_block_aligned_split'], [loop.lineno, loop.end_lineno]


def request(key, prompt, computed=0, decode=False):
    return NS(request_id=key, num_prompt_tokens=prompt, num_tokens=prompt,
              num_tokens_with_spec=prompt, num_computed_tokens=computed,
              num_output_placeholders=0, max_tokens=128,
              next_decode_eligible_step=0, is_prefill_chunk=not decode,
              has_encoder_inputs=False, spec_token_ids=[])


def fixture(schedule, align, threshold=0, decode=False, aligned=True):
    long = request('long', 143971)
    running = [long]
    if decode:
        d = request('decode', 100, 100, True)
        d.num_tokens_with_spec = 103
        running.insert(0, d)
    s = NS(current_step=0, max_num_scheduled_tokens=4096, _pause_state=0,
           max_num_encoder_input_tokens=0, prefill_capacity_bound=False,
           running=running, scheduler_config=NS(long_prefill_token_threshold=threshold),
           max_model_len=262144, num_sampled_tokens_per_step=1,
           need_mamba_block_aligned_split=aligned, num_lookahead_tokens=2,
           cache_config=NS(block_size=2048), use_eagle=False,
           kv_cache_manager=NS(new_step_starts=lambda: None, allocate_slots=lambda *a, **k: object()))
    s._mamba_block_aligned_split = lambda *a, **k: align(s, *a, **k)
    short = request('short', 29399)
    steps = []
    for step in range(1, 100):
        before = long.num_computed_tokens
        out = schedule(s)
        remaining = out['remaining']
        # Waiting request cache-hit boundary is 26624. This is only its first
        # prospective grant, not an implementation of admission/publication.
        raw = min(29399-26624, remaining)
        grant = align(s, short, raw, num_new_local_computed_tokens=26624) if raw and aligned else raw
        row = dict(step=step, long_computed_before=before, **out,
                   waiting_budget_eligible=remaining > 0,
                   short_raw_grant=raw, short_aligned_grant=grant)
        steps.append(row)
        if grant:
            break
        for r in running:
            grant = out['grants'].get(r.request_id, 0)
            r.num_computed_tokens += grant
            if not r.is_prefill_chunk:
                # Model a ready 3-token decode at each next scheduling step.
                r.num_tokens = r.num_computed_tokens
                r.num_tokens_with_spec = r.num_computed_tokens + 3
        if not out['grants']:
            break
    return dict(threshold=threshold, decode_first=decode, mamba_alignment=aligned,
                first_short_positive_grant_step=next((r['step'] for r in steps if r['short_aligned_grant']), None),
                steps=steps)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scheduler', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    schedule, align, lines = extract(a.scheduler)
    cases = {name: fixture(schedule, align, cap, decode, aligned) for name, cap, decode, aligned in
             [('uncapped',0,False,True), ('cap_2048',2048,False,True),
              ('uncapped_decode',0,True,True), ('cap_2048_decode',2048,True,True),
              ('noalign_uncapped_decode',0,True,False),
              ('noalign_cap_2048_decode',2048,True,False)]}
    assert cases['uncapped']['first_short_positive_grant_step'] == 36
    assert cases['cap_2048']['first_short_positive_grant_step'] == 1
    assert cases['cap_2048_decode']['steps'][0]['remaining'] == 2045
    assert cases['cap_2048_decode']['steps'][0]['short_aligned_grant'] == 0
    assert cases['noalign_uncapped_decode']['steps'][0]['remaining'] == 0
    assert cases['noalign_cap_2048_decode']['steps'][0]['short_aligned_grant'] == 2045
    result = dict(scope='CPU extracted running-grant AST; no actual historical steps or NPU latency',
                  source=str(a.scheduler.resolve()), running_loop_lines=lines, cases=cases)
    a.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:dict(first_short_grant=v['first_short_positive_grant_step'],
                          first_step=v['steps'][0]) for k,v in cases.items()}, indent=2))


if __name__ == '__main__':
    main()
