"""Admitted Qwen35B TP2: owned full-model graphs versus owned eager target."""
import json
import os
from pathlib import Path
from datetime import timedelta
import torch
import torch_npu  # noqa: F401
from vllm.engine.arg_utils import EngineArgs
from vllm.config import set_current_vllm_config
from vllm.distributed import init_distributed_environment, initialize_model_parallel
from vllm_ascend.utils import enable_custom_op
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
from betterscale.live import LiveRuntime, TorchStateBackend, live_runtime
from betterscale.live.arch.ascend.graph import ACLGraphBackend
from betterscale.live.llm.qwen35 import Geometry, Capacity
from betterscale.live.llm.qwen35.loader import load_models
from betterscale.live.llm.qwen35.execution import QwenExecutionRoot
from betterscale.live.llm.qwen35.graphs import QwenLiveLLMRoot

run = Path(os.environ['CAPSULE'])
rank = int(os.environ['RANK'])
torch.npu.set_device(rank)
assert enable_custom_op()
init_device_properties_triton()
torch.set_default_dtype(torch.bfloat16)
model = Path('/workspace/models/Qwen3.5-35B-A3B')
config = EngineArgs(model=str(model), dtype='bfloat16', tensor_parallel_size=2,
    max_model_len=4096, max_num_seqs=2, enable_expert_parallel=False,
    enforce_eager=True, enable_prefix_caching=False,
    speculative_config={'method':'mtp', 'num_speculative_tokens':2}).create_engine_config()
with set_current_vllm_config(config):
    init_distributed_environment(world_size=2, rank=rank, local_rank=rank, backend='hccl',
        distributed_init_method='env://', timeout=timedelta(seconds=180))
    initialize_model_parallel(tensor_model_parallel_size=2, backend='hccl')
    device = f'npu:{rank}'
    target, draft = load_models(config, model, device)
    print(json.dumps({'rank':rank, 'weights_covered':True,
                     'moe_type':type(target.model.layers[0].mlp.experts).__name__}), flush=True)
    geometry = Geometry.from_config(json.loads((model/'config.json').read_text()), tensor_parallel_size=2)
    capacity = Capacity(2, 2, token_pages=32)

    def make(kind):
        with live_runtime(LiveRuntime(device=device,
            state_backend=TorchStateBackend(device, memory_budget_bytes=1 << 30),
            graph_backend=ACLGraphBackend(device=device) if kind is QwenLiveLLMRoot else None)):
            return kind(geometry, capacity, target, draft, greedy_only=True)

    prompt = [9707, 11, 358, 1079, 264, 1786, 13]
    eager = make(QwenExecutionRoot); eager.activate()
    baseline = eager.generate(prompt, 12, speculative=False)
    print(json.dumps({'rank':rank, 'eager':baseline}), flush=True)
    eager.close()
    root = make(QwenLiveLLMRoot); root.activate()
    assert len(list(root.named_graphs())) == 4
    assert all(g.prepared and not g.metadata.requires_forward_replay for _,g in root.named_graphs())
    result = root.generate(prompt, 12)
    assert result['token_ids'] == baseline['token_ids'], ('mtp/graph-vs-eager', result, baseline)
    b = root.generate([785, 279, 3363, 374], 4)
    assert b['seat'] == 1
    continuation = prompt + result['token_ids'] + [271]
    warm = root.generate(continuation, 8)
    assert warm['cached_tokens'] == 19 and warm['seat'] == 0
    root.close()
    eager.activate()
    cold = eager.generate(continuation, 8, speculative=False)
    assert warm['token_ids'] == cold['token_ids'], ('warm-vs-cold',warm,cold)
    eager.close()
    receipt = {'rank':rank, 'model':str(model), 'baseline':baseline, 'live':result,
        'unrelated':b, 'warm':warm, 'cold':cold, 'graphs':4,
        'max_allocated':torch.npu.max_memory_allocated(), 'passed':True}
    (run/f'rank{rank}.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt), flush=True)
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()
