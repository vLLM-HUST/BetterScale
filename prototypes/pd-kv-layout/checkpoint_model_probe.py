"""Actual TP2 target-only export/drop/new-seat import/continuation gate."""
import argparse,json,time
from pathlib import Path
import checkpoint_entry
from betterscale.models.qwen35 import CAPTURE_SIZES
from vllm import LLM,SamplingParams
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
 a=p.parse_args();a.output.mkdir(exist_ok=False,parents=True)
 model=LLM(model='/data/shared_models/modelscope_cache/Qwen/Qwen3.5-35B-A3B',tensor_parallel_size=2,
  distributed_executor_backend='mp',worker_cls='checkpoint_entry.Worker',dtype='bfloat16',kv_cache_dtype='auto',
  max_model_len=8192,max_num_seqs=16,max_num_batched_tokens=4096,gpu_memory_utilization=.9,seed=17,
  enable_prefix_caching=True,mamba_cache_mode='align',async_scheduling=True,
  additional_config={'enable_cpu_binding':False,'using_live_runtime':True},limit_mm_per_prompt={'image':0,'video':0},
  compilation_config=dict(cudagraph_mode='FULL',cudagraph_capture_sizes=CAPTURE_SIZES,max_cudagraph_capture_size=4096),
  speculative_config=dict(method='mtp',num_speculative_tokens=2),scheduler_cls='checkpoint_entry.Scheduler',
  kv_cache_memory_bytes=8<<30,generation_config='vllm')
 client=model.llm_engine.engine_core
 salt='target-checkpoint-vertical'
 prompt=('Observation: the blue key opens the north gate; the red key opens the south gate.\n'*220)+'\nQuestion: Which key opens the north gate?\nAnswer:'
 def generate(name,prompt,salt,n):
  inp=dict(prompt=prompt,cache_salt=salt) if isinstance(prompt,str) else dict(prompt_token_ids=prompt,cache_salt=salt)
  result=model.generate([inp],SamplingParams(temperature=0,max_tokens=n,ignore_eos=True),use_tqdm=False)[0]
  row=dict(prompt_token_ids=result.prompt_token_ids,token_ids=list(result.outputs[0].token_ids),text=result.outputs[0].text,cached=result.num_cached_tokens)
  (a.output/(name+'.json')).write_text(json.dumps(row,indent=2));print(name,row['cached'],flush=True);return row
 try:
  first=generate('prefix',prompt,salt,64);tokens=first['prompt_token_ids']+first['token_ids']
  start=time.monotonic();payload=client.call_utility('pd_export_target',tokens,salt)
  exported=time.monotonic()-start
  old_seat=payload['header']['seat'];old_blocks=payload['header']['blocks']
  removed=client.call_utility('pd_drop_target',salt);assert removed==[old_seat]
  start=time.monotonic();installed=client.call_utility('pd_import_target',payload,salt);imported=time.monotonic()-start
  warm=generate('restored',tokens,salt,64);cold=generate('cold',tokens,salt+'-cold',64)
  receipt=dict(equal=warm['token_ids']==cold['token_ids'],cached=warm['cached'],cursor=payload['header']['cursor'],
      export_seconds=exported,import_seconds=imported,old_seat=old_seat,new_seat=installed['seat'],
      old_blocks=old_blocks,new_blocks=installed['blocks'],workers=installed['workers'],draft_state_transferred=False)
  (a.output/'receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt),flush=True)
  assert receipt['equal'] and receipt['cached']==receipt['cursor'] and cold['cached']==0
  assert installed['seat']!=old_seat or installed['blocks']!=old_blocks,'Restore reused identical physical addresses'
 finally:model.llm_engine.engine_core.shutdown()

if __name__=='__main__':main()
