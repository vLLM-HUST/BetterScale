"""Bounded donor loader seam: no Worker, runner, State allocation or graphs."""
import json, os
from pathlib import Path
from datetime import timedelta
import torch
import torch_npu
from safetensors import safe_open
from vllm.engine.arg_utils import EngineArgs
from vllm.config import set_current_vllm_config
from vllm.distributed import init_distributed_environment, initialize_model_parallel
from vllm.model_executor.model_loader.utils import initialize_model, process_weights_after_loading
from vllm.model_executor.models.qwen3_5 import Qwen3_5ForCausalLM
from vllm.model_executor.models.qwen3_5_mtp import Qwen3_5MTP
from vllm_ascend.utils import enable_custom_op

assert enable_custom_op()
torch.npu.set_device(0)
torch.set_default_dtype(torch.bfloat16)
model_path=Path('/workspace/models/Qwen3.5-0.8B')
config=EngineArgs(model=str(model_path), dtype='bfloat16', max_model_len=4096,
                  max_num_seqs=2, enforce_eager=True, enable_prefix_caching=False,
                  speculative_config={'method':'mtp','num_speculative_tokens':2}).create_engine_config()
with set_current_vllm_config(config):
    init_distributed_environment(world_size=1,rank=0,local_rank=0,backend='hccl',
        distributed_init_method='file://'+str(Path(os.environ['CAPSULE'])/'distributed-init'),timeout=timedelta(seconds=90))
    initialize_model_parallel(backend='hccl')
    with torch.device('npu'):
        target=initialize_model(config,model_class=Qwen3_5ForCausalLM)
        draft=initialize_model(config,model_class=Qwen3_5MTP)
    draft.model.embed_tokens=target.model.embed_tokens
    draft.lm_head=target.lm_head
    files=sorted(set(json.loads((model_path/'model.safetensors.index.json').read_text())['weight_map'].values()))
    def weights(draft=False):
        for filename in files:
            with safe_open(str(model_path/filename),framework='pt',device='cpu') as reader:
                for name in reader.keys():
                    if name.startswith('model.language_model.'):
                        if draft and 'embed_tokens' not in name:continue
                        yield name.replace('model.language_model.','model.',1),reader.get_tensor(name)
                    elif draft and name.startswith('mtp.'):
                        yield name,reader.get_tensor(name)
    loaded_target=target.load_weights(weights())
    loaded_draft=draft.load_weights(weights(True))
    process_weights_after_loading(target, config.model_config, torch.device('npu'))
    process_weights_after_loading(draft, config.model_config, torch.device('npu'))
    target.eval();draft.eval()
    caches=[]
    for root_name,model in [('target',target),('draft',draft)]:
        for name,module in model.named_modules():
            if hasattr(module,'kv_cache'):
                value=module.kv_cache
                def numel(x):
                    if isinstance(x,torch.Tensor):return x.numel()
                    if isinstance(x,(tuple,list)):return sum(numel(y) for y in x)
                    assert x is None,(name,type(x))
                    return 0
                assert numel(value)==0,(root_name,name,'native cache already allocated')
                caches.append(root_name+'.'+name)
    receipt={'loaded_target':len(loaded_target),'loaded_draft':len(loaded_draft),
             'cache_consumers':caches,'unallocated':True,
             'shared_embedding':target.model.embed_tokens is draft.model.embed_tokens,
             'shared_lm_head':target.lm_head is draft.lm_head,
             'scope':'weights and empty numerical consumers only; no forward or State integration'}
    print(json.dumps(receipt),flush=True)
    Path(os.environ['CAPSULE'],'loader-receipt.json').write_text(json.dumps(receipt,indent=2))
    torch.npu.synchronize()
