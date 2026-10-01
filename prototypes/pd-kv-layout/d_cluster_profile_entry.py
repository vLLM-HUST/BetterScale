"""Observation-only D6 worker: host dispatch receipts and bounded CANN capture."""
import json
import time
from pathlib import Path
from contextlib import nullcontext
from ep6_state_entry import Worker as BaseWorker

class Worker(BaseWorker):
    def compile_or_warm_up_model(self):
        result = super().compile_or_warm_up_model()
        import os
        directory = os.environ.get("BETTERSCALE_FIA_METADATA_AUDIT")
        if directory:
            from fia_padding_observer import install
            install(directory)
        return result

    def d_probe_arm(self, output, label, batch, capture):
        if getattr(self, "_probe", None) is not None:
            raise RuntimeError("Previous observation not retired")
        from vllm.distributed import get_ep_group
        self._probe = dict(output=output, label=label, batch=batch, capture=capture,
                           rank=get_ep_group().rank_in_group, rows=[], steady=0,
                           profiler=None, closed=False)
        return self._probe["rank"]

    def execute_model(self, scheduler_output):
        p = getattr(self, "_probe", None)
        if p is None:
            return super().execute_model(scheduler_output)
        counts = dict(scheduler_output.num_scheduled_tokens)
        steady = len(counts) == p["batch"] and all(v == 1 for v in counts.values())
        if steady:
            p["steady"] += 1
        if p["capture"] and steady and p["steady"] == 24:
            import torch
            import torch_npu
            torch.npu.synchronize()  # Capture boundary only; excluded from timing.
            p["profiler"] = torch_npu.profiler.profile(
                activities=[torch_npu.profiler.ProfilerActivity.CPU,
                            torch_npu.profiler.ProfilerActivity.NPU],
                record_shapes=False, profile_memory=False, with_stack=False,
                experimental_config=torch_npu.profiler._ExperimentalConfig(
                    profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
                    export_type=torch_npu.profiler.ExportType.Db),
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                    str(Path(p["output"]) / "profile"),
                    worker_name=f'rank{p["rank"]}', analyse_flag=False))
            p["profiler"].start()
        active = p["profiler"] is not None and not p["closed"]
        if active:
            import torch
            context = torch.profiler.record_function(f'D_DECODE_STEP_{p["steady"]}')
        else:
            context = nullcontext()
        begin = time.perf_counter_ns()
        with context:
            result = super().execute_model(scheduler_output)
        end = time.perf_counter_ns()
        p["rows"].append(dict(begin_ns=begin, end_ns=end, steady=steady,
                              decode_index=p["steady"], profiled=active,
                              scheduled=counts))
        if active and p["steady"] >= 39:
            import torch
            torch.npu.synchronize()
            p["profiler"].stop()
            p["closed"] = True
        return result

    def d_probe_finish(self):
        p = self._probe
        if p["profiler"] is not None and not p["closed"]:
            import torch
            torch.npu.synchronize()
            p["profiler"].stop()
            p["closed"] = True
        import torch
        memory = dict(allocated=torch.npu.memory_allocated(),
                      reserved=torch.npu.memory_reserved(),
                      free_total=list(torch.npu.mem_get_info()))
        report = {k:v for k,v in p.items() if k != "profiler"}
        report["memory"] = memory
        path = Path(p["output"]) / f'{p["label"]}-rank{p["rank"]}.json'
        path.write_text(json.dumps(report, indent=2))
        self._probe = None
        if p["capture"] and not p["closed"]:
            raise RuntimeError("Did not reach the requested steady capture window")
        return dict(rank=p["rank"], steps=len(p["rows"]), memory=memory)
