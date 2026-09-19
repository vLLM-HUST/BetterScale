"""Observe native DP2/EP2 with asymmetric P/D work; no execution patches."""
import json
import os
from pathlib import Path
import torch
from vllm_ascend.worker.worker import NPUWorker


class ObserveWorker(NPUWorker):
    def compile_or_warm_up_model(self):
        result = super().compile_or_warm_up_model()
        self.observing = False
        self.pending = []
        runner = self.model_runner
        original_execute = runner.execute_model
        original_forward = runner._model_forward
        self.scheduled = {}

        def execute(scheduler_output, *args, **kwargs):
            self.scheduled = dict(scheduler_output.num_scheduled_tokens)
            return original_execute(scheduler_output, *args, **kwargs)

        def forward(*args, **kwargs):
            if not self.observing:
                return original_forward(*args, **kwargs)
            from vllm.forward_context import get_forward_context
            ctx = get_forward_context()
            batch = runner.input_batch
            reqs = list(batch.req_ids[:batch.num_reqs])
            row = dict(index=len(self.pending), scheduled=self.scheduled.copy(),
                       req_ids=reqs, mode=str(ctx.cudagraph_runtime_mode),
                       descriptor=str(ctx.batch_descriptor),
                       local_computed=[int(x) for x in batch.num_computed_tokens_cpu[:batch.num_reqs]])
            start, end = [torch.npu.Event(enable_timing=True) for _ in range(2)]
            start.record()
            result = original_forward(*args, **kwargs)
            end.record()
            self.pending.append((row, start, end))
            return result

        runner.execute_model = execute
        runner._model_forward = forward
        return result

    def start_window(self, label, profile=False):
        from vllm.distributed import get_ep_group
        self.root = Path(os.environ['SHARED_EP_OUTPUT']) / label
        self.root.mkdir(parents=True, exist_ok=True)
        self.rank = self.model_runner.parallel_config.data_parallel_rank
        self.pending = []
        self.profiler = None
        if profile:
            import torch_npu
            self.profiler = torch_npu.profiler.profile(
                activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
                record_shapes=False, profile_memory=False, with_stack=False,
                experimental_config=torch_npu.profiler._ExperimentalConfig(profiler_level=torch_npu.profiler.ProfilerLevel.Level1),
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(self.root/'profile'),worker_name=f'rank{self.rank}',analyse_flag=False))
            self.profiler.start()
        self.observing = True
        p = self.model_runner.parallel_config
        receipt = dict(dp_rank=self.rank, tp=p.tensor_parallel_size,
                       ep_rank=get_ep_group().rank_in_group, ep_size=get_ep_group().world_size,
                       device=str(self.model_runner.device))
        (self.root/f'manifest-rank{self.rank}.json').write_text(json.dumps(receipt,indent=2))
        return receipt

    def stop_window(self):
        torch.npu.synchronize()
        self.observing = False
        if self.profiler:
            self.profiler.stop()
        rows = []
        for row, start, end in self.pending:
            row['forward_ms'] = start.elapsed_time(end)
            rows.append(row)
        (self.root/f'forwards-rank{self.rank}.json').write_text(json.dumps(rows,indent=2))
        return dict(count=len(rows))
