"""Three native TP2 clients sharing EP6: bounded four-layer dummy compatibility.

Not BetterScale State/FULL or model quality qualification. No MTP in this gate.
"""
import argparse,json,multiprocessing as mp,os,time
from pathlib import Path

def four_layers(config):
 # Pinned parser first calls overrides on a model-type-only placeholder.
 if not hasattr(config,'text_config'):return config
 text=config.get_text_config();text.num_hidden_layers=4;text.layer_types=text.layer_types[:4]
 return config

def run(rank,output,barrier,real,control):
 os.environ.update(VLLM_DP_RANK=str(rank),VLLM_DP_RANK_LOCAL=str(rank),VLLM_DP_SIZE='1' if control else '3',
     VLLM_DP_MASTER_IP='127.0.0.1',VLLM_DP_MASTER_PORT='29671',VLLM_ASCEND_ENABLE_FLASHCOMM1='0')
 from vllm import LLM,SamplingParams
 model=LLM(model='/data/shared_models/modelscope_cache/Qwen/Qwen3.5-35B-A3B',
     load_format='auto' if real else 'dummy',**({} if real else {'hf_overrides':four_layers}),tensor_parallel_size=2,enable_expert_parallel=not control,
     dtype='bfloat16',quantization=None,max_model_len=2048,max_num_batched_tokens=2048,max_num_seqs=2,
     kv_cache_memory_bytes=2<<30,enable_prefix_caching=False,skip_tokenizer_init=True,
     enforce_eager=True,async_scheduling=False,seed=601,worker_cls='ep6_native_worker.Worker',
     limit_mm_per_prompt={'image':0,'video':0},additional_config={'enable_cpu_binding':False})
 barrier.wait(timeout=300)
 outputs=[]
 if real:
  from transformers import AutoTokenizer
  tokenizer=AutoTokenizer.from_pretrained('/data/shared_models/modelscope_cache/Qwen/Qwen3.5-35B-A3B')
  code=['MARBLE','COBALT','ORCHID'][rank]
 for wave,lengths in enumerate(([32+rank*13],[256 if rank==0 else 17])):
  if real:
   filler='The ledger contains routine observations about weather and roads. '* (80 if wave and rank==0 else 3)
   text=f'Remember this session code: {code}.\n'+filler+f'\nReply with only the session code, without punctuation or explanation.'
   tokens=tokenizer.apply_chat_template([{'role':'user','content':text}],tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=False)
   assert isinstance(tokens,list) and tokens and all(isinstance(t,int) for t in tokens)
   prompts=[dict(prompt_token_ids=tokens)];lengths=[len(tokens)]
  else:prompts=[dict(prompt_token_ids=[17+rank]*n) for n in lengths]
  result=model.generate(prompts,SamplingParams(temperature=0,max_tokens=16 if real else 8,ignore_eos=not real,detokenize=False,stop_token_ids=[tokenizer.eos_token_id] if real else None))
  rows=[list(r.outputs[0].token_ids) for r in result]
  record=dict(lengths=lengths,outputs=rows)
  if real:
   record.update(text=tokenizer.decode(rows[0],skip_special_tokens=True),expected=code)
   record['correct']=record['text'].strip()==code
  else:assert all(len(r)==8 for r in rows)
  outputs.append(record);barrier.wait(timeout=180)
 (output/f'rank{rank}.json').write_text(json.dumps(outputs,indent=2));barrier.wait(timeout=60)
 model.llm_engine.engine_core.shutdown()

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--real',action='store_true');p.add_argument('--control',action='store_true');a=p.parse_args();a.output.mkdir(exist_ok=False,parents=True)
 size=1 if a.control else 3
 ctx=mp.get_context('spawn');barrier=ctx.Barrier(size);children=[ctx.Process(target=run,args=(r,a.output,barrier,a.real,a.control),name=f'ep6-dp{r}') for r in range(size)]
 for child in children:child.start()
 deadline=time.monotonic()+600
 try:
  while any(c.is_alive() for c in children):
   for c in children:c.join(timeout=.5)
   if any(c.exitcode not in (None,0) for c in children):raise RuntimeError([(c.name,c.exitcode) for c in children])
   if time.monotonic()>deadline:raise TimeoutError('EP6 dummy deadline')
  correct=not a.real or all(row['correct'] for r in range(size) for row in json.loads((a.output/f'rank{r}.json').read_text()))
  (a.output/'complete.json').write_text(json.dumps(dict(status='passed' if correct else 'quality-failed',scope=('40-layer real-weight six retrievals' if a.real else 'four-layer dummy')+(' native TP2 no EP control' if a.control else ' native DP3TP2EP6')+' eager no MTP; no State/PD/FULL claim',exitcodes=[c.exitcode for c in children]),indent=2))
  assert correct,'EP6 retrieval failure; inspect rank receipts'
 finally:
  barrier.abort()
  for c in children:
   if c.is_alive():c.terminate()
  for c in children:c.join(timeout=10)
