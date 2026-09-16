"""Native startup/observation; candidate owns the complete execution loop."""

import json
import os
import time
from pathlib import Path
import torch
from vllm_ascend.worker.worker import NPUWorker


class ServingWorker(NPUWorker):
    def begin_native_profile(self, steps):
        from profiling import ProfileWindow
        from vllm.distributed import get_tp_group

        self.window = ProfileWindow(
            Path(os.environ["SERVING_OUTPUT"]) / "native-profile",
            get_tp_group().rank_in_group,
            steps,
        )
        r = self.model_runner
        original = r.execute_model

        def observed(*args, **kwargs):
            result = original(*args, **kwargs)
            self.window.step()
            return result

        self.native_execute = original
        r.execute_model = observed

    def end_native_profile(self):
        self.window.close()
        self.model_runner.execute_model = self.native_execute

    def reset_memory_peaks(self):
        torch.npu.synchronize()
        torch.npu.reset_peak_memory_stats()

    def memory_receipt(self):
        torch.npu.synchronize()
        return dict(
            allocated=torch.npu.memory_allocated(),
            reserved=torch.npu.memory_reserved(),
            peak_allocated=torch.npu.max_memory_allocated(),
            peak_reserved=torch.npu.max_memory_reserved(),
        )

    def run_serving(self, sessions, rounds, chunks, profile_steps):
        import torch.distributed as dist
        from vllm.distributed import get_tp_group
        from vllm.v1.sample.metadata import SamplingMetadata
        from vllm.v1.sample.logits_processor import LogitsProcessors
        from livemodule import LiveRuntime, live_runtime
        from livemodule.arch.ascend.runtime.aclgraph import ACLGraphBackend
        from live_root import AdoptedStateBackend, ModelBundle
        from root import ServingRoot
        from cache import PrefixLeases
        from reactor import Reactor
        from scheduler import SessionScheduler
        from profiling import ProfileWindow
        from storage import backing_views
        from boundary import forbid_runner_execution

        r = self.model_runner
        group = get_tp_group()
        rank = group.rank_in_group
        out = Path(os.environ["SERVING_OUTPUT"])
        path = out / f"candidate-rank{rank}.json"
        receipt = dict(status="STARTED", rounds=[])
        path.write_text(json.dumps(receipt))
        assert (
            r.parallel_config.data_parallel_size == 1
            and r.parallel_config.pipeline_parallel_size == 1
        )
        n = len(sessions)
        maxlen = r.max_model_len
        bs = r.input_batch.block_table.block_tables[0].block_size
        pools = backing_views(r)
        model = r.model
        while hasattr(model, "unwrap"):
            model = model.unwrap()
        sampling = SamplingMetadata(
            temperature=None,
            all_greedy=True,
            all_random=False,
            top_p=None,
            top_k=None,
            generators={},
            max_num_logprobs=None,
            no_penalties=True,
            prompt_token_ids=None,
            frequency_penalties=torch.zeros(n, device=r.device),
            presence_penalties=torch.zeros(n, device=r.device),
            repetition_penalties=torch.ones(n, device=r.device),
            output_token_ids=[[] for _ in range(n)],
            allowed_token_ids_mask=None,
            bad_words_token_ids={},
            logitsprocs=LogitsProcessors(),
        )
        names = tuple(r.kv_cache_config.kv_cache_groups[0].layer_names)
        table = torch.zeros(
            (1, (maxlen + bs - 1) // bs), dtype=torch.int32, device=r.device
        )
        bundle = ModelBundle(
            model,
            r.sampler,
            sampling,
            r.vllm_config,
            names,
            table,
            pools,
            r.device,
            bs,
            r.attn_backend,
            maxlen,
        )
        backend = AdoptedStateBackend(
            r.device, memory_budget_bytes=sum(x.numel() for x in pools) + 1024**3
        )
        runtime = LiveRuntime(
            device=r.device,
            state_backend=backend,
            graph_backend=ACLGraphBackend(device=r.device),
        )
        root = None
        window = None
        try:
            # Donor scheduler has no requests. Adopt the entire arena exclusively;
            # native APC metadata is not used by this independently owned allocator.
            with forbid_runner_execution(r) as calls:
                for pool in pools:
                    pool.zero_()
                with live_runtime(runtime):
                    root = ServingRoot(
                        bundle, backend, residents=n, max_length=maxlen, chunks=chunks
                    )
                activation = time.perf_counter()
                root.activate()
                torch.npu.synchronize()
                receipt["activation_s"] = time.perf_counter() - activation
                cache = PrefixLeases(r.kv_cache_config, maxlen)
                generations = [0] * n
                for repeat in range(rounds):
                    dist.barrier(group=group.cpu_group)
                    torch.npu.synchronize()
                    torch.npu.reset_peak_memory_stats()
                    reactor = Reactor(root)
                    scheduler = SessionScheduler(
                        cache,
                        sessions,
                        slots=n,
                        chunks=chunks,
                        table_width=table.shape[1],
                        generations=generations,
                    )
                    window = ProfileWindow(
                        out / "owned-profile",
                        rank,
                        profile_steps if repeat == 0 else 0,
                        warmup_steps=int(os.environ.get("PROFILE_WARMUP_STEPS", "0")),
                    )
                    hit_before = cache.hit_tokens
                    before = root.forward_calls
                    launches_before = (
                        root.static_attention.launch_calls
                        if root.static_attention
                        else 0
                    )
                    start = time.perf_counter()
                    while not scheduler.done:
                        while (plan := scheduler.next_plan()) is not None:
                            reactor.submit(plan)
                            window.step()
                        if not reactor.pending:
                            raise RuntimeError(
                                "no runnable work: finite KV capacity exhausted"
                            )
                        plan, rows = reactor.receive()
                        quorum = [None] * group.world_size
                        dist.all_gather_object(quorum, rows, group=group.cpu_group)
                        scheduler.receive(plan, quorum)
                    torch.npu.synchronize()
                    elapsed = time.perf_counter() - start
                    window.close()
                    window = None
                    assert root.active_invocation_count == 0 and not cache.live
                    assert root.forward_calls == before and not calls
                    if root.static_attention:
                        assert root.static_attention.launch_calls == launches_before
                    entry = dict(
                        repeat=repeat,
                        elapsed_s=elapsed,
                        records=scheduler.records,
                        output_tokens=sum(
                            x["output_tokens"] for x in scheduler.records
                        ),
                        hit_tokens=cache.hit_tokens - hit_before,
                        peak_used_blocks=cache.peak_used_blocks,
                        deferred_releases=cache.deferred_releases,
                        waves=scheduler.sequence,
                        memory=self.memory_receipt(),
                        events=scheduler.events,
                    )
                    receipt["rounds"].append(entry)
                    path.write_text(json.dumps(receipt))
                receipt.update(
                    status="PASS",
                    runner_calls=calls,
                    graphs=len(root.frames),
                    forward_calls=root.forward_calls,
                    static_fia=(
                        dict(
                            protocol=(
                                "native-host-wave"
                                if hasattr(root.static_attention, "prepare")
                                else "static-non-fd"
                            ),
                            bootstrap_calls=root.static_attention.bootstrap_calls,
                            launch_calls=root.static_attention.launch_calls,
                            plans=len(root.static_attention.plans),
                            wave_plans=getattr(root.static_attention, "wave_plans", 0),
                            dispatches=dict(
                                getattr(root.static_attention, "dispatches", {})
                            ),
                        )
                        if root.static_attention
                        else None
                    ),
                    scope="concurrent closed-loop original-history replay; native APC leases; whole-wave N+2",
                )
                root.close()
                if root.static_attention:
                    root.static_attention.close()
                root = None
        except BaseException as exc:
            receipt.update(status="FAIL", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            if window is not None:
                window.close()
            path.write_text(json.dumps(receipt))
        return dict(status=receipt["status"], rank=rank, path=str(path))
