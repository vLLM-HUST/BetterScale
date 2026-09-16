"""EP2 BF16 fused DFC control; native operator, NZ weights, bounded FULL graphs."""

import argparse
import json
import os
from pathlib import Path
import statistics
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu
import profile_capture

H, M, E, K = 2048, 768, 128, 8


def worker(rank, out):
    torch.set_num_threads(2)
    torch.npu.set_device(rank)
    torch_npu.npu.config.allow_internal_format = True
    extension = os.environ.get("DFC_EXTENSION")
    if extension:
        # Match the A2 provider ABI: its BF16 API has an extra xActiveMask.
        torch.ops.load_library(extension)
    else:
        from vllm_ascend.utils import enable_custom_op

        enable_custom_op()
    dist.init_process_group(
        "hccl", init_method="tcp://127.0.0.1:43760", rank=rank, world_size=2
    )
    group = dist.distributed_c10d._get_default_group()
    # Materialize the communicator before passing its name into an MC2 op.
    bootstrap = torch.ones(1, device="npu")
    dist.all_reduce(bootstrap)
    torch.npu.synchronize()
    comm = group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
    torch.manual_seed(742)
    # Shared random base plus expert-dependent powers-of-two scaling: an
    # independent per-route oracle can detect owner/routing errors without
    # gathering the entire weight tensor between devices.
    base_up = (torch.randn(H, 2 * M) * 0.01).to(torch.bfloat16).npu()
    base_down = (torch.randn(M, H) * 0.01).to(torch.bfloat16).npu()
    factors = torch.tensor([0.5 if e % 2 == 0 else 1.0 for e in range(E)], device="npu")
    w1 = (base_up.unsqueeze(0) * factors[rank * 64 : (rank + 1) * 64, None, None]).to(
        torch.bfloat16
    )
    w2 = base_down.unsqueeze(0).repeat(64, 1, 1)
    weights1 = [torch_npu.npu_format_cast(w1, 29)]
    weights2 = [torch_npu.npu_format_cast(w2, 29)]
    address_sets = int(os.environ.get("DFC_WEIGHT_SETS", "1"))
    assert address_sets in (1, 2)
    catalogs = [(weights1, weights2)]
    if address_sets == 2:
        catalogs.append(
            (
                [torch_npu.npu_format_cast(w1, 29).clone()],
                [torch_npu.npu_format_cast(w2, 29).clone()],
            )
        )
        assert catalogs[0][0][0].data_ptr() != catalogs[1][0][0].data_ptr()
    del w1, w2
    scale1 = [torch.ones((64, 2 * M), dtype=torch.int64, device="npu")]
    scale2 = [torch.ones((64, H), dtype=torch.int64, device="npu")]
    results = []
    for pattern in ("balanced", "hot8"):
        for rows in (1, 16, 32):
            torch.manual_seed(180 + rank + rows)
            x = (torch.randn(rows, H) * 0.1).to(torch.bfloat16).npu()
            if pattern == "balanced":
                ids = (torch.arange(rows * 8).reshape(rows, 8) + rank * 8) % 128
            else:
                ids = torch.tensor([0, 1, 2, 3, 64, 65, 66, 67]).expand(rows, 8).clone()
            ids = ids.to(torch.int32).npu()
            probs = torch.full((rows, K), 1 / K, device="npu", dtype=torch.float32)
            output = torch.empty_like(x)
            counts = torch.zeros((64,), device="npu", dtype=torch.int32)

            def body(catalog=0):
                torch.ops._C_ascend.dispatch_ffn_combine(
                    x=x,
                    weight1=catalogs[catalog][0],
                    weight2=catalogs[catalog][1],
                    expert_idx=ids,
                    scale1=scale1,
                    scale2=scale2,
                    bias1=[],
                    bias2=[],
                    probs=probs,
                    group=comm,
                    max_output_size=512,
                    out=output,
                    expert_token_nums=counts,
                )

            # Same BF16 intermediate rounding as the native GMM chain.
            partials = []
            for factor in (0.5, 1.0):
                up = x @ (base_up * factor).to(torch.bfloat16)
                partials.append(torch_npu.npu_swiglu(up) @ base_down)
            chosen = torch.stack([partials[0], partials[1]], dim=1)
            parity = (ids % 2).long()
            expected = (
                chosen.gather(1, parity[:, :, None].expand(-1, -1, H))
                .float()
                .mean(1)
                .to(torch.bfloat16)
            )
            for _ in range(3):
                body()
            torch.npu.synchronize()
            torch.testing.assert_close(output, expected, rtol=0.02, atol=2e-5)
            relative = (
                torch.linalg.vector_norm(output.float() - expected.float())
                / torch.linalg.vector_norm(expected.float()).clamp_min(1e-12)
            ).item()
            assert relative < 0.01, relative
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                for step in range(2):
                    body(step % address_sets)
            for _ in range(3):
                graph.replay()
            torch.npu.synchronize()
            torch.testing.assert_close(output, expected, rtol=0.02, atol=2e-5)
            timings = []
            for _ in range(5):
                dist.barrier()
                torch.npu.synchronize()
                start = torch.npu.Event(enable_timing=True)
                end = torch.npu.Event(enable_timing=True)
                start.record()
                for _ in range(64):
                    graph.replay()
                end.record()
                end.synchronize()
                timings.append(start.elapsed_time(end) * 1000 / 128)
            results.append(
                dict(
                    rank=rank,
                    weight_address_sets=address_sets,
                    rows_per_source=rows,
                    pattern=pattern,
                    global_tokens=rows * 2,
                    capacity=512,
                    median_us=statistics.median(timings),
                    trials_us=timings,
                    relative_l2=relative,
                    exact=torch.equal(output, expected),
                )
            )
            Path(out, f"rank{rank}.json").write_text(json.dumps(results, indent=2))
            # Capture only a warm broad-hit case; oracle, capture and timing
            # trials stay outside the profiler so its overhead is not a result.
            if (
                pattern == "balanced"
                and rows == 32
                and os.environ.get("DEVICE_SERVICE_PROFILE")
            ):
                dist.barrier()
                profiler = profile_capture.start(f"dfc{rank}")
                for _ in range(8):
                    graph.replay()
                torch.npu.synchronize()
                profile_capture.stop(profiler)
            graph.reset()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    mp.spawn(worker, args=(args.out,), nprocs=2, join=True)
    print("DFC_EP2_PASS", flush=True)
