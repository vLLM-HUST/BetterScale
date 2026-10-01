"""Real DP3/TP2/EP6 State gate: skew, idle peers, exact warm/cold target continuation."""
import argparse,json,multiprocessing as mp,os,time
from pathlib import Path

def run(rank,output,barrier):
 os.environ.update(VLLM_DP_RANK=str(rank),VLLM_DP_RANK_LOCAL=str(rank),VLLM_DP_SIZE='3',VLLM_DP_MASTER_IP='127.0.0.1',VLLM_DP_MASTER_PORT='29673')
 from vllm.plugins import load_general_plugins
 load_general_plugins()
 import ep6_state_entry
 from betterscale.models.qwen35 import CAPTURE_SIZES
 from vllm import LLM,SamplingParams
 model=LLM(model='/data/shared_models/modelscope_cache/Qwen/Qwen3.5-35B-A3B',tensor_parallel_size=2,
  enable_expert_parallel=True,all2all_backend='flashinfer_all2allv',distributed_executor_backend='mp',
  worker_cls='ep6_state_entry.Worker',dtype='bfloat16',max_model_len=8192,max_num_seqs=16,max_num_batched_tokens=4096,
  seed=17,enable_prefix_caching=True,mamba_cache_mode='align',async_scheduling=True,
  additional_config={'enable_cpu_binding':False,'using_live_runtime':True},limit_mm_per_prompt={'image':0,'video':0},
  compilation_config=dict(cudagraph_mode='FULL',cudagraph_capture_sizes=CAPTURE_SIZES,max_cudagraph_capture_size=4096),
  speculative_config=dict(method='mtp',num_speculative_tokens=2),scheduler_cls='ep6_state_entry.Scheduler',
  kv_cache_memory_bytes=8<<30,generation_config='vllm')
 barrier.wait(timeout=600)
 def generate(name,tokens,salt,n):
  inp=dict(prompt=tokens,cache_salt=salt) if isinstance(tokens,str) else dict(prompt_token_ids=tokens,cache_salt=salt)
  r=model.generate([inp],SamplingParams(temperature=0,max_tokens=n,ignore_eos=True),use_tqdm=False)[0]
  d=dict(prompt_token_ids=r.prompt_token_ids,token_ids=list(r.outputs[0].token_ids),cached=r.num_cached_tokens,text=r.outputs[0].text)
  (output/f'rank{rank}-{name}.json').write_text(json.dumps(d,indent=2));print('STATE_PHASE',rank,name,d['cached'],flush=True)
  barrier.wait(timeout=180);return d
 try:
  salt=f'ep6-state-{rank}'
  prompt=('Observation: the blue key opens the north gate; the red key opens the south gate.\n'*[220,12,1][rank])+'\nQuestion: Which key opens the north gate?\nAnswer:'
  first=generate('prefix',prompt,salt,[64,16,8][rank]);tokens=first['prompt_token_ids']+first['token_ids']
  warm=generate('warm',tokens,salt,32)
  cold=generate('cold',tokens,salt+'-cold',32)
  receipt=dict(equal=warm['token_ids']==cold['token_ids'],cached=warm['cached'],cursor=len(tokens)-1,cold_cached=cold['cached'])
  (output/f'rank{rank}-receipt.json').write_text(json.dumps(receipt,indent=2))
  assert receipt['equal'] and receipt['cached']==receipt['cursor'] and cold['cached']==0,receipt
  barrier.wait(timeout=120)
 finally:model.llm_engine.engine_core.shutdown()

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 ctx=mp.get_context('spawn');barrier=ctx.Barrier(3);children=[ctx.Process(target=run,args=(r,a.output,barrier),name=f'ep6-state-{r}') for r in range(3)]
 for c in children:c.start()
 deadline=time.monotonic()+1000
 try:
  while any(c.is_alive() for c in children):
   for c in children:c.join(timeout=.5)
   if any(c.exitcode not in (None,0) for c in children):raise RuntimeError([(c.name,c.exitcode) for c in children])
   if time.monotonic()>deadline:raise TimeoutError('EP6 State qualification deadline')
  (a.output/'complete.json').write_text(json.dumps(dict(status='passed',scope='DP3 TP2 EP6 target-only State skew/idle warm-cold equality; not P/D',exitcodes=[c.exitcode for c in children]),indent=2))
 finally:
  barrier.abort()
  for c in children:
   if c.is_alive():c.terminate()
  for c in children:c.join(timeout=10)
