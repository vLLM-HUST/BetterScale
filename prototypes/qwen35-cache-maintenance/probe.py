"""One-card real-weight roundtrip; execute only inside selected-device lease."""

import json
import os
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer
from betterscale.live.llm.qwen35.bootstrap import open_model
from betterscale.live.llm.qwen35.scheduler import Scheduler
from betterscale.live.runtime.host_state import HostStateKey
from maintenance import CacheMaintenance, CopyWorker

run = Path(os.environ["CAPSULE"])
model = str(run / "model")
tok = AutoTokenizer.from_pretrained(model, local_files_only=True)
prompts = [
    tok.apply_chat_template(
        [
            {
                "role": "user",
                "content": ("A harmless filler sentence. " * n)
                + "\nReply with exactly ORCHID-7319 and nothing else.",
            }
        ],
        enable_thinking=False,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=False,
    )
    for n in (23, 31, 41)
]
receipt = {
    "model_revision": "2fc06364715b967f1860aea9cf38778875588b17",
    "route": "owned Qwen research root, TP1; not native35B async",
    "passed": False,
}


def drain(scheduler):
    results = {}
    for _ in range(800):
        if not scheduler.busy:
            return results
        results.update(scheduler.tick())
        if scheduler.maintenance.busy and not scheduler.active:
            scheduler.maintenance.wait(30)
    raise AssertionError("bounded scheduler progress exhausted")


def views(root, worker, seat):
    resident = root.residents_table.seats[seat]
    result = {}
    for name, state in worker.states:
        ids = [seat] if state.domain is root.residents else resident.pages
        span = state.physical_blocks_per_logical_block
        for ordinal, block in enumerate(ids):
            start = state.leading_physical_blocks + block * span
            result[name, ordinal] = state.tensor[start : start + span]
    return result


with open_model(
    model,
    context_tokens=8192,
    execution_seats=2,
    resident_seats=3,
    token_pages=128,
    paged_attention=True,
    prefill_tokens=256,
) as root:
    worker = CopyWorker(root, host_bytes=256 << 20)
    cache = CacheMaintenance(root.residents_table, worker)
    scheduler = Scheduler(root, maintenance=cache)
    try:
        scheduler.submit("A", prompts[0], 9, cache_salt="session-A")
        a = drain(scheduler)["A"]
        table = root.residents_table
        source = a["seat"]
        original_pages = list(table.seats[source].pages)
        original_tokens = list(table.seats[source].tokens)
        expected = {
            k: v.cpu().contiguous().view(torch.uint8).clone()
            for k, v in views(root, worker, source).items()
        }
        key = HostStateKey("A", 1)
        scheduler.submit("B", prompts[1], 9)
        scheduler.tick()  # another request is active on a different seat
        store = cache.store(source, key)
        assert source not in table.idle_indices and key not in cache.host
        assert table.seats[source].pages == original_pages
        started = time.monotonic()
        b = drain(scheduler)["B"]
        assert key in cache.host and not table.seats[source].tokens
        receipt["store_and_other_request_seconds"] = time.monotonic() - started
        receipt["host_payload_bytes"] = worker.backend.committed_bytes
        scheduler.submit("C", prompts[2], 9)
        c = drain(scheduler)["C"]
        assert c["seat"] == source  # original seat and pages really overwritten
        destination = next(
            i for i in table.idle_indices if i not in (source, b["seat"])
        )
        epoch_before = root.continuation.resident_epoch.tensor[destination].item()
        load = cache.load(key, destination)
        assert (
            destination not in table.idle_indices
            and not table.seats[destination].tokens
        )
        # No model waves are necessary to get a completion/wakeup.
        waves = scheduler.stats["waves"]
        cache.wait(30)
        assert scheduler.stats["waves"] == waves and not table.seats[destination].tokens
        scheduler.tick()  # receipt publication belongs to scheduler
        assert table.seats[destination].tokens == original_tokens
        restored_views = views(root, worker, destination)
        assert expected.keys() == restored_views.keys()
        assert all(
            torch.equal(value, restored_views[k].cpu().contiguous().view(torch.uint8))
            for k, value in expected.items()
        )
        assert (
            root.continuation.resident_epoch.tensor[destination].item() > epoch_before
        )
        new_pages = list(table.seats[destination].pages)
        assert new_pages != original_pages
        continuation = original_tokens + tok.encode(
            "\n" + "Another filler sentence. " * 20 + " Reply with exactly PINE-427."
        )
        scheduler.submit("resume-A", continuation, 9, cache_salt="session-A")
        warm = drain(scheduler)["resume-A"]
        assert warm["seat"] == destination and warm["cached_tokens"] == len(
            original_tokens
        )
        for i in table.idle_indices:
            table.evict(i)
        cold = root.generate(continuation, 9)
        # Independent target-only, nonchunked prefill oracle, as qualified before.
        for i in table.idle_indices:
            table.evict(i)
        widths = root.prefill_widths
        root.prefill_widths = ()
        oracle = root.generate(continuation, 9, speculative=False)
        root.prefill_widths = widths
        assert warm["token_ids"] == cold["token_ids"] == oracle["token_ids"]
        # Cancel store/load after enqueue: receipt must still precede recycling.
        cancel_seat = oracle["seat"]
        cancel_key = HostStateKey("cancel", 1)
        n = cache.store(cancel_seat, cancel_key)
        cache.cancel(n)
        assert table.seats[cancel_seat].io_owner == n
        cache.wait()
        scheduler.tick()
        assert table.seats[cancel_seat].tokens and cancel_key not in cache.host
        n = cache.load(key, cancel_seat)
        cache.cancel(n)
        assert table.seats[cancel_seat].io_owner == n
        cache.wait()
        scheduler.tick()
        assert (
            not table.seats[cancel_seat].tokens
            and table.seats[cancel_seat].io_owner is None
        )
        receipt.update(
            passed=True,
            source_seat=source,
            destination_seat=destination,
            original_pages=original_pages,
            restored_pages=new_pages,
            raw_views_compared=len(expected),
            cached_tokens=warm["cached_tokens"],
            warm=warm,
            cold=cold,
            target_only=oracle,
            receipts=cache.receipts,
            scheduler=scheduler.snapshot(),
            idle_completion=True,
            cancelled_store_and_load=True,
        )
        print("CACHE_ROUNDTRIP_PASS", json.dumps(receipt), flush=True)
    finally:
        scheduler.close()
        receipt["host_bytes_after_close"] = worker.backend.committed_bytes
        (run / "result.json").write_text(json.dumps(receipt, indent=2) + "\n")
