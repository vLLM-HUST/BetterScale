"""Execute complete pinned donor schedule + three grant seams on CPU doubles.

Real schedule source; fake request/KV/encoder/output and completion updates.
This checks routing and token accounting, not device numerics or allocator fit.
"""

import argparse
import ast
from collections import deque
from contextlib import nullcontext
import json
from pathlib import Path
from types import SimpleNamespace as NS
import time

from betterscale.models.qwen35.fair_schedule import bind
from betterscale.models.qwen35.prefill_round_robin import PrefillRoundRobin, prepare


class Queue(deque):
    def peek_request(self):
        return self[0]

    def pop_request(self):
        return self.popleft()

    def prepend_request(self, r):
        self.appendleft(r)

    def prepend_requests(self, rs):
        self.extendleft(reversed(rs))


class Status:
    WAITING = NS(name="WAITING")
    PREEMPTED = NS(name="PREEMPTED")
    RUNNING = NS(name="RUNNING")


class Request:
    def __init__(self, key, prompt, computed=0, decode=False):
        self.request_id = key
        self.num_prompt_tokens = prompt
        self.num_computed_tokens = computed
        self.num_tokens = prompt
        self.num_tokens_with_spec = prompt
        self.num_output_placeholders = 0
        self.max_tokens = 128
        self.next_decode_eligible_step = 0
        self.is_prefill_chunk = not decode
        self.has_encoder_inputs = False
        self.spec_token_ids = []
        self.lora_request = None
        self.status = Status.RUNNING if computed or decode else Status.WAITING
        self.mm_features = []
        self.prefill_stats = None
        self.all_token_ids = [0] * prompt
        self.cache_salt = None
        self.skip_reading_prefix_cache = False
        self.use_structured_output = False

    def __hash__(self):
        return id(self)


def load(path, cls, name, env):
    node = next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == cls
    )
    method = next(
        n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == name
    )
    module = ast.Module(body=[method], type_ignores=[])
    exec(compile(module, str(path), "exec"), env)
    return env[name]


def scheduler(path, *, patch=True):
    blocks = NS(get_block_ids=lambda: ([1],))
    manager = NS(
        new_step_starts=lambda: None,
        allocate_slots=lambda *a, **k: blocks,
        get_computed_blocks=lambda r: (blocks, 26624 if r.request_id == "short" else 0),
        empty_kv_cache_blocks=blocks,
        get_blocks=lambda r: blocks,
        get_num_common_prefix_blocks=lambda r: [0],
    )
    s = NS(
        current_step=0,
        max_num_scheduled_tokens=4096,
        _pause_state=0,
        max_num_encoder_input_tokens=0,
        prefill_capacity_bound=False,
        running=[],
        waiting=Queue(),
        skipped_waiting=Queue(),
        requests={},
        scheduler_config=NS(
            long_prefill_token_threshold=0, enable_chunked_prefill=True
        ),
        max_model_len=262144,
        num_sampled_tokens_per_step=1,
        need_mamba_block_aligned_split=False,
        num_lookahead_tokens=2,
        kv_cache_manager=manager,
        lora_config=None,
        policy="fcfs",
        max_num_running_reqs=16,
        num_waiting_for_streaming_input=0,
        connector=None,
        has_mamba_layers=False,
        ec_connector=None,
        num_spec_tokens=2,
        dynamic_sd_lookup=None,
        is_encoder_decoder=False,
        scheduler_reserve_full_isl=False,
        log_stats=False,
        _inflight_prefills=set(),
        kv_cache_config=NS(kv_cache_groups=[0]),
        use_v2_model_runner=False,
        prev_step_scheduled_req_ids=set(),
        needs_kv_cache_zeroing=False,
        reset_preempted_req_ids=set(),
        finished_req_ids=set(),
        defer_block_free=True,
        sched_step_seq=0,
        processed_step_seq=0,
        encoder_cache_manager=NS(get_freed_mm_hashes=lambda: []),
        _prefill_round_robin=PrefillRoundRobin(),
        _waiting_for_resident=lambda r: False,
        residents=NS(
            offer=lambda tokens, *a, **k: NS(warm=len(tokens) == 29399, seat=0),
            seats=[NS(cursor=26624)],
        ),
        pp_size=1,
    )
    s._is_blocked_waiting_status = lambda status: False
    s._select_waiting_queue_for_scheduling = lambda: s.skipped_waiting or s.waiting
    s._make_cached_request_data = lambda *a: None

    def update(out):
        for key, n in out.num_scheduled_tokens.items():
            r = s.requests[key]
            r.num_computed_tokens += n
            r.is_prefill_chunk = r.num_computed_tokens < r.num_prompt_tokens

    s._update_after_schedule = update
    env = dict(
        time=time,
        PauseState=NS(PAUSED_ALL=1, UNPAUSED=0),
        RequestStatus=Status,
        SchedulerOutput=lambda **kw: NS(**kw),
        Request=Request,
        create_request_queue=lambda _: Queue(),
        record_function_or_nullcontext=lambda _: nullcontext(),
        NewRequestData=NS(from_request=lambda r, *a: NS(req_id=r.request_id)),
        SchedulingPolicy=NS(PRIORITY="priority"),
    )
    native = load(path, "Scheduler", "schedule", env)
    return s, bind(native) if patch else native


def run(path, patch):
    s, step = scheduler(path, patch=patch)
    d = Request("decode", 100, 100, True)
    d.num_tokens_with_spec = 103
    d.status = Status.RUNNING
    long = Request("long", 143971)
    long.status = Status.RUNNING
    short = Request("short", 29399)
    s.running = [d, long]
    s.waiting.append(short)
    s.requests = {r.request_id: r for r in [d, long, short]}
    rows = []
    for i in range(4):
        if patch:
            prepare(s)
        out = step(s)
        rows.append(
            dict(
                step=i + 1,
                grants=out.num_scheduled_tokens,
                waiting=[r.request_id for r in s.waiting]
                + [r.request_id for r in s.skipped_waiting],
            )
        )
        assert out.total_num_scheduled_tokens <= 4096
        assert out.num_scheduled_tokens["decode"] == 3
        d.num_tokens = d.num_computed_tokens
        d.num_tokens_with_spec = d.num_computed_tokens + 3
        # Fake sampled outputs only once prefill completes; model/async numerics
        # are outside the fixture. Keep completed short request ready for decode.
        if short.num_computed_tokens >= short.num_prompt_tokens:
            short.num_tokens = short.num_computed_tokens
            short.num_tokens_with_spec = short.num_computed_tokens + 3
    return rows


def padded_waiting(path):
    # Native waiting order differs from the rotating grant order. A hot single
    # token may become a three-token MTP row before our cap: never cut it to one.
    s, step = scheduler(path)
    d = Request("decode", 100, 100, True)
    d.num_tokens_with_spec = 103
    d.status = Status.RUNNING
    hot, bulk = Request("short", 26625), Request("bulk", 4092)
    s.running = [d]
    s.waiting.extend([hot, bulk])
    s.requests = {r.request_id: r for r in (d, hot, bulk)}
    s.residents.offer = lambda tokens, *a, **k: NS(warm=len(tokens) == 26625, seat=0)
    s._prefill_round_robin.ring = deque(["bulk", "short"])
    prepare(s)
    out = step(s)
    assert out.num_scheduled_tokens == {"decode": 3, "bulk": 4092}
    assert hot in s.waiting or hot in s.skipped_waiting
    return dict(
        grants=out.num_scheduled_tokens, unused=4096 - out.total_num_scheduled_tokens
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scheduler", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--ascend-balance", type=Path)
    a = p.parse_args()
    old = run(a.scheduler, False)
    new = run(a.scheduler, True)
    assert all("short" not in r["grants"] for r in old)
    assert new[1]["grants"]["short"] == 2775
    result = dict(
        scope=__doc__,
        baseline=old,
        round_robin=new,
        padded_waiting=padded_waiting(a.scheduler),
    )
    if a.ascend_balance:
        _, native = scheduler(a.scheduler, patch=False)
        wrapper = load(
            a.ascend_balance,
            "BalanceScheduler",
            "schedule",
            {"Scheduler": NS(schedule=native), "SchedulerOutput": object},
        )
        adapted = bind(wrapper, balance_enabled=False)
        assert adapted.__code__.co_filename == str(a.scheduler)
        result["ascend_disabled_wrapper"] = "source-pinned forwarding resolved"
    a.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
