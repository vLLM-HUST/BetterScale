"""Experimental native EP6 worker: source admission and SD/V1 ABI only."""
from vllm_ascend.worker.worker import NPUWorker
class Worker(NPUWorker):
 def __init__(self,vllm_config,*args,**kwargs):
  from betterscale.models.qwen import check_runtime
  from betterscale.models.qwen35.mamba_abi import install
  p=vllm_config.parallel_config;hf=vllm_config.model_config.hf_text_config
  assert (p.tensor_parallel_size,p.data_parallel_size,p.pipeline_parallel_size,p.enable_expert_parallel) in ((2,3,1,True),(2,1,1,False))
  assert hf.num_experts==256 and hf.model_type=='qwen3_5_moe_text'
  assert vllm_config.model_config.quantization is None and vllm_config.speculative_config is None
  check_runtime('qwen35_pins.json');install()
  super().__init__(vllm_config,*args,**kwargs)
