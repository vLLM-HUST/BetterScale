"""Two-rank native Qwen MoE seam: random weights, not a model-quality claim."""
import argparse
from datetime import timedelta
import json
import os
import time
from pathlib import Path
import torch
import torch_npu
import vllm_ascend.vllm_ascend_C  # register the pinned native MoE operators


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--arm", choices=("serial", "donor", "early"), required=True)
    p.add_argument("--tokens", type=int, default=48)
    p.add_argument("--profile", action="store_true")
    p.add_argument("--graph", action="store_true")
    p.add_argument("--full-forward", action="store_true")
    p.add_argument("--tp", type=int, choices=(1, 2), default=1)
    p.add_argument("--peer-tokens", type=int)
    p.add_argument("--iterations", type=int, default=30)
    a = p.parse_args()
    rank = int(os.environ["RANK"])
    assert int(os.environ["WORLD_SIZE"]) == 2
    assert torch.npu.device_count() == 2
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    torch.set_num_threads(2)
    from vllm.config import VllmConfig, ModelConfig, ParallelConfig, SchedulerConfig, set_current_vllm_config
    from vllm.distributed import init_distributed_environment, initialize_model_parallel
    from vllm.distributed import destroy_model_parallel, destroy_distributed_environment
    from vllm_ascend.ascend_config import init_ascend_config
    from vllm_ascend.distributed.parallel_state import init_ascend_model_parallel, destroy_ascend_model_parallel
    from vllm_ascend.ascend_forward_context import MoECommType, set_mc2_tokens_capacity
    from vllm_ascend.ops.fused_moe.moe_comm_method import get_moe_comm_method
    from vllm_ascend.ops.fused_moe.fused_moe import AscendMoERunner
    from vllm.model_executor.layers.fused_moe.layer import FusedMoE
    from vllm.model_executor.models.qwen3_next import Qwen3NextMLP
    from vllm.model_executor.layers.linear import ReplicatedLinear
    from vllm.forward_context import set_forward_context, get_forward_context

    dp = 2 // a.tp
    counts = [a.tokens, a.peer_tokens if a.peer_tokens is not None else a.tokens]
    if a.tp == 2:
        assert counts[0] == counts[1] and a.full_forward
    local_tokens = counts[rank] if dp == 2 else a.tokens
    parallel = ParallelConfig(tensor_parallel_size=a.tp, data_parallel_size=dp,
        data_parallel_rank=rank // a.tp, data_parallel_size_local=dp,
        distributed_executor_backend="external_launcher", enable_expert_parallel=True)
    from transformers import Qwen3NextConfig
    config_dir = a.output / f"model-config-rank{rank}"
    hf = Qwen3NextConfig(hidden_size=2048, num_experts=256, num_experts_per_tok=8,
                        moe_intermediate_size=512, shared_expert_intermediate_size=512)
    hf.architectures = ["Qwen3NextForCausalLM"]
    hf.save_pretrained(config_dir)
    config = VllmConfig(parallel_config=parallel,
        model_config=ModelConfig(model=str(config_dir), skip_tokenizer_init=True,
                                 dtype="bfloat16", enforce_eager=True, max_model_len=8192),
        scheduler_config=SchedulerConfig(is_encoder_decoder=False,
                                        max_model_len=8192, max_num_seqs=16,
                                        max_num_batched_tokens=4096),
        additional_config=dict(enable_cpu_binding=False,
                               multistream_overlap_shared_expert=a.arm != "serial",
                               betterscale_shared_expert_overlap=a.arm == "early"))
    with set_current_vllm_config(config):
        init_ascend_config(config)
        init_distributed_environment(2, rank, "env://", rank, "hccl", timedelta(seconds=90))
        initialize_model_parallel(a.tp, backend="hccl")
        init_ascend_model_parallel(parallel)
        set_mc2_tokens_capacity(config, 16, 3)
        from betterscale.models.qwen35.moe_overlap import configure
        configure(config)
        torch.set_default_dtype(torch.bfloat16)
        with torch.device("npu"):
            gate = ReplicatedLinear(2048, 1, bias=False, params_dtype=torch.bfloat16)
            shared = Qwen3NextMLP(2048, 512, "silu", reduce_results=False, expert_gate=gate)
            model = FusedMoE(num_experts=256, top_k=8, hidden_size=2048,
                intermediate_size=512, params_dtype=torch.bfloat16,
                shared_experts=shared, prefix="overlap_probe", runner_cls=AscendMoERunner)
        with torch.no_grad():
            torch.manual_seed(101 + rank)
            for value in model.parameters():
                value.normal_(0, .02)
            if a.full_forward:
                # Shared experts are DP replicas and TP shards; the scalar gate
                # is replicated even across TP ranks.
                torch.manual_seed(401 + (rank if a.tp == 2 else 0))
                for value in shared.parameters():
                    value.normal_(0, .02)
                torch.manual_seed(501)
                gate.weight.normal_(0, .02)
        # Donor conversion/weight-layout path, not a manually rewritten GMM.
        model._quant_method.process_weights_after_loading(model.routed_experts)
        x = torch.randn(local_tokens, 2048, device="npu", dtype=torch.bfloat16) * .1
        logits = torch.randn(local_tokens, 256, device="npu", dtype=torch.float32)
        with set_forward_context(None, config, num_tokens=local_tokens,
                num_tokens_across_dp=torch.tensor(counts[:dp], dtype=torch.int32)):
            ctx = get_forward_context()
            ctx.moe_comm_type = MoECommType.ALLGATHER
            ctx.moe_comm_method = get_moe_comm_method(MoECommType.ALLGATHER)
            ctx.max_tokens_across_dp = max(counts)
            ctx.in_profile_run = False
            ctx.flash_comm_v1_enabled = False
            ctx.eplb_heat_collection_status = False
            def forward():
                if a.full_forward:
                    ctx.moe_layer_index = 0  # one model wave per leaf call
                    return model(hidden_states=x, router_logits=logits)
                shared_out, routed_out = model.shared_forward_impl(x, logits)
                return shared_out + routed_out

            with torch.no_grad():
                graph = None
                if a.graph:
                    for _ in range(3):
                        forward()
                    torch.npu.synchronize()
                    graph = torch.npu.NPUGraph()
                    with torch.npu.graph(graph):
                        graph_output = forward()
                    torch.npu.synchronize()

                def execute():
                    if graph is None:
                        return forward()
                    graph.replay()
                    return graph_output

                outputs = []
                # Changing inputs/routing protects against accidental stale results.
                for wave in range(4):
                    torch.manual_seed(701 + (rank if dp == 2 else 0) * 10 + wave)
                    x.copy_(torch.randn_like(x) * .1)
                    logits.copy_(torch.randn_like(logits))
                    result = execute()
                    torch.npu.synchronize()
                    assert torch.isfinite(result).all()
                    outputs.append(result.cpu())
                    if graph is not None:
                        reference = forward()
                        torch.npu.synchronize()
                        torch.testing.assert_close(outputs[-1], reference.cpu(), rtol=0, atol=0)
                torch.save(outputs, a.output / f"rank{rank}-outputs.pt")
                for _ in range(5):
                    execute()
                torch.npu.synchronize()
                begin = time.perf_counter()
                for _ in range(a.iterations):
                    execute()
                torch.npu.synchronize()
                wall_ms = (time.perf_counter() - begin) * 1000 / a.iterations
                if a.profile:
                    from torch_npu.profiler import profile, ProfilerActivity, tensorboard_trace_handler
                    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.NPU],
                            on_trace_ready=tensorboard_trace_handler(str(a.output / f"profile-rank{rank}")),
                            record_shapes=True) as prof:
                        for _ in range(5):
                            execute()
                            prof.step()
                        torch.npu.synchronize()
        a.output.mkdir(parents=True, exist_ok=True)
        (a.output / f"rank{rank}.json").write_text(json.dumps(dict(
            arm=a.arm, shape=list(result.shape), finite=True, wall_ms=wall_ms,
            iterations=a.iterations, graph=a.graph, tp=a.tp, full_forward=a.full_forward,
            scope="native MoE leaf; full-engine qualification separate")))
        print("native MoE wiring ready", rank, a.arm, flush=True)
        destroy_ascend_model_parallel()
        destroy_model_parallel()
        destroy_distributed_environment()


if __name__ == "__main__":
    main()
