"""Freeze a real Qwen prefill invocation and replay its native FULL graph.

The extra ACL DMA is OUTSIDE the graph on an independent stream. No scheduler,
attention, MoE, metadata-update or collective implementation is replaced.
"""

import ctypes as C
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from vllm.config import CUDAGraphMode
from vllm.distributed import get_ep_group
from vllm.forward_context import get_forward_context
from vllm_ascend.worker.worker import NPUWorker


class DMAWorker(NPUWorker):
    def compile_or_warm_up_model(self):
        result = super().compile_or_warm_up_model()
        self._dma_armed = False
        runner = self.model_runner
        original = runner._model_forward

        def forward(*args, **kwargs):
            result = original(*args, **kwargs)
            if self._dma_armed:
                self._dma_armed = False
                self.measure(result, lambda: original(*args, **kwargs))
            return result

        runner._model_forward = forward
        return result

    def arm_dma_probe(self):
        self._dma_armed = True
        return {"device": str(self.model_runner.device)}

    @torch.inference_mode()
    def measure(self, output, replay):
        runner = self.model_runner
        ctx = get_forward_context()
        rank = runner.parallel_config.data_parallel_rank
        root = Path(os.environ["DMA_PROBE_OUTPUT"])
        root.mkdir(parents=True, exist_ok=True)
        assert ctx.cudagraph_runtime_mode == CUDAGraphMode.FULL
        assert ctx.batch_descriptor.num_tokens == 4096, ctx.batch_descriptor
        assert runner.input_batch.num_reqs == 1
        assert isinstance(output, torch.Tensor)
        entry = runner.model.concrete_aclgraph_entries[ctx.batch_descriptor]
        assert entry.aclgraph is not None
        graph = entry.aclgraph
        main = torch.npu.current_stream()
        dma = torch.npu.Stream()
        cpu_group = get_ep_group().cpu_group
        acl = C.CDLL("libascendcl.so")
        for name, types in {
            "aclrtMallocHost": [C.POINTER(C.c_void_p), C.c_size_t],
            "aclrtFreeHost": [C.c_void_p],
            "aclrtMemcpyAsync": [
                C.c_void_p,
                C.c_size_t,
                C.c_void_p,
                C.c_size_t,
                C.c_int,
                C.c_void_p,
            ],
        }.items():
            f = getattr(acl, name)
            f.argtypes, f.restype = types, C.c_int

        def call(name, *args):
            rc = getattr(acl, name)(*args)
            if rc:
                raise RuntimeError(f"{name} returned {rc}")

        def barrier():
            torch.npu.synchronize()
            dist.barrier(group=cpu_group)

        def tensors(value):
            if isinstance(value, torch.Tensor):
                yield value
            elif isinstance(value, dict):
                for v in value.values():
                    yield from tensors(v)
            elif isinstance(value, (tuple, list)):
                for v in value:
                    yield from tensors(v)

        barrier()
        # A second same-state replay is the ordinary compute-only oracle.
        replay()
        torch.npu.synchronize()
        reference = output.clone()
        unique = {}
        for tensor in tensors(runner.kv_caches):
            storage = tensor.untyped_storage()
            key = (storage.data_ptr(), storage.nbytes())
            if key not in unique:
                view = torch.empty(0, dtype=torch.uint8, device=tensor.device).set_(
                    storage, 0, (storage.nbytes(),), (1,)
                )
                unique[key] = (view, view.clone())
        assert unique
        barrier()
        maxbytes = 4 * 1024**3
        source = torch.full((maxbytes,), 37, dtype=torch.uint8, device=runner.device)
        destination = torch.zeros_like(source)
        hp = C.c_void_p()
        call("aclrtMallocHost", C.byref(hp), maxbytes)
        C.memset(hp, 37, maxbytes)
        barrier()
        manifest = dict(
            rank=rank,
            device=str(runner.device),
            model="Qwen3-30B-A3B",
            mode=str(ctx.cudagraph_runtime_mode),
            tokens=4096,
            requests=1,
            graph_entries=len(runner.model.concrete_aclgraph_entries),
            kv_bytes=sum(k[1] for k in unique),
            allocated=torch.npu.memory_allocated(),
            reserved=torch.npu.memory_reserved(),
            dma_max_bytes=maxbytes,
            scope="fixed real prefill graph; graph-external DMA; no scheduler or sampler timing",
        )
        (root / f"manifest-rank{rank}.json").write_text(json.dumps(manifest, indent=2))
        log = (root / f"records-rank{rank}.jsonl").open("w", buffering=1)
        try:
            sustained = os.environ.get("DMA_SUSTAINED") == "1"
            directions = (
                [("d2d_local", 3)]
                if sustained
                else [("h2d", 1), ("d2h", 2), ("d2d_local", 3)]
            )
            for direction, kind in directions:
                dst, src = {
                    1: (destination.data_ptr(), hp.value),
                    2: (hp.value, source.data_ptr()),
                    3: (destination.data_ptr(), source.data_ptr()),
                }[kind]
                for amount in ((16, 64) if sustained else (256, 1024, 4096)):
                    repeat_copies = amount if sustained else 1
                    mib = 4096 if sustained else amount
                    size = mib * 1024**2
                    for active in (("both",) if sustained else ("rank0", "both")):
                        participating = rank == 0 or active == "both"
                        for trial in range(3):
                            modes = ["compute", "copy", "serial", "overlap"]
                            if trial % 2:
                                modes.reverse()
                            for mode in modes:
                                barrier()
                                begin, end, ms, me, cs, ce = [
                                    torch.npu.Event(enable_timing=True)
                                    for _ in range(6)
                                ]
                                begin.record(main)
                                if mode == "serial":
                                    ms.record(main)
                                    replay()
                                    me.record(main)
                                if mode != "compute":
                                    with torch.npu.stream(dma):
                                        dma.wait_event(
                                            me if mode == "serial" else begin
                                        )
                                        cs.record(dma)
                                        if participating:
                                            for _ in range(repeat_copies):
                                                call(
                                                    "aclrtMemcpyAsync",
                                                    dst,
                                                    size,
                                                    src,
                                                    size,
                                                    kind,
                                                    dma.npu_stream,
                                                )
                                        ce.record(dma)
                                if mode in ("compute", "overlap"):
                                    ms.record(main)
                                    replay()
                                    me.record(main)
                                if mode != "compute":
                                    main.wait_event(ce)
                                end.record(main)
                                end.synchronize()
                                maxerr = 0.0
                                exact = True
                                if mode != "copy":
                                    exact = torch.equal(output, reference)
                                    if not exact:
                                        maxerr = (output - reference).abs().max().item()
                                        assert torch.allclose(
                                            output, reference, rtol=0.001, atol=0.001
                                        ), maxerr
                                if mode != "compute" and participating:
                                    # Sample all pages plus both ends; whole transfer is
                                    # independently covered by the earlier byte oracle.
                                    if kind == 2:
                                        assert (
                                            C.c_ubyte.from_address(
                                                hp.value + size - 1
                                            ).value
                                            == 37
                                        )
                                        assert all(
                                            C.c_ubyte.from_address(hp.value + i).value
                                            == 37
                                            for i in range(0, size, 4096)
                                        )
                                    else:
                                        assert bool(
                                            (destination[:size:4096].cpu() == 37).all()
                                        )
                                        assert destination[size - 1].item() == 37
                                row = dict(
                                    rank=rank,
                                    direction=direction,
                                    mib=mib,
                                    active=active,
                                    trial=trial,
                                    mode=mode,
                                    total_ms=begin.elapsed_time(end),
                                    compute_ms=(
                                        ms.elapsed_time(me) if mode != "copy" else None
                                    ),
                                    copy_ms=(
                                        cs.elapsed_time(ce)
                                        if mode != "compute"
                                        else None
                                    ),
                                    participating=participating,
                                    copies=repeat_copies,
                                    total_copy_bytes=size * repeat_copies,
                                    output_exact=exact,
                                    max_abs_error=maxerr,
                                )
                                log.write(json.dumps(row) + "\n")
                        # KV writes repeated at identical positions must be idempotent.
                        assert all(
                            torch.equal(view, ref) for view, ref in unique.values()
                        ), "KV differs"
                        print(
                            f"DMA_PROBE_PASS rank={rank} {direction} {mib}MiB {active}",
                            flush=True,
                        )
            import torch_npu

            profiler = torch_npu.profiler.profile(
                activities=[
                    torch_npu.profiler.ProfilerActivity.CPU,
                    torch_npu.profiler.ProfilerActivity.NPU,
                ],
                record_shapes=False,
                profile_memory=False,
                with_stack=False,
                experimental_config=torch_npu.profiler._ExperimentalConfig(
                    profiler_level=torch_npu.profiler.ProfilerLevel.Level1
                ),
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                    str(root / "profile"), worker_name=f"rank{rank}", analyse_flag=False
                ),
            )
            barrier()
            profiler.start()
            profile_directions = (
                [("compute", 0), ("d2d_sustained", 3)]
                if sustained
                else [("compute", 0), ("h2d", 1), ("d2h", 2), ("d2d_local", 3)]
            )
            for direction, kind in profile_directions:
                barrier()
                with torch.profiler.record_function(f"dma_probe::{direction}"):
                    begin, done = torch.npu.Event(), torch.npu.Event()
                    begin.record(main)
                    if kind:
                        dst, src = {
                            1: (destination.data_ptr(), hp.value),
                            2: (hp.value, source.data_ptr()),
                            3: (destination.data_ptr(), source.data_ptr()),
                        }[kind]
                        with torch.npu.stream(dma):
                            dma.wait_event(begin)
                            copy_size = maxbytes if sustained else 1024**3
                            for _ in range(64 if sustained else 1):
                                call(
                                    "aclrtMemcpyAsync",
                                    dst,
                                    copy_size,
                                    src,
                                    copy_size,
                                    kind,
                                    dma.npu_stream,
                                )
                            done.record(dma)
                    replay()
                    if kind:
                        main.wait_event(done)
                    torch.npu.synchronize()
            profiler.stop()
            (root / f"passed-rank{rank}.json").write_text(
                json.dumps(dict(pass_all=True, kv_exact=True))
            )
        finally:
            barrier()
            log.close()
            call("aclrtFreeHost", hp)
