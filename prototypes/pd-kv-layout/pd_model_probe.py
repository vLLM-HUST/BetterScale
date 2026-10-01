"""Single-host P2/D6 target-only handoff through real Mooncake DRAM.

Quiescent session turns, one controller/SQLite directory, no overlap or production
availability claim. D attention owners are fixed per session; MTP never transfers.
"""
import argparse,json,multiprocessing as mp,os,time,traceback
from pathlib import Path
ROOT=Path('/workspace/betterscale-pd-runtime')
MODEL='/data/shared_models/modelscope_cache/Qwen/Qwen3.5-35B-A3B'

def worker(role,connection,output):
 import sys
 is_p=role=='P';rank=0 if is_p else int(role[1:])
 package=ROOT/('candidate-package-6-no-draft' if is_p else 'candidate-package-7-ep6-state')
 runtime=ROOT/('owned-runtime' if is_p else 'ep6-runtime')
 sys.path[:0]=[str(package),str(runtime)]
 os.environ['PYTHONPATH']=f'{package}:{runtime}:'+os.environ.get('PYTHONPATH','')
 os.environ.update(ASCEND_RT_VISIBLE_DEVICES='0,1' if is_p else '2,3,4,5,6,7',
  HCCL_IF_BASE_PORT='29635' if is_p else '29675',VLLM_DP_RANK=str(rank),VLLM_DP_RANK_LOCAL=str(rank),
  VLLM_DP_SIZE='1' if is_p else '3',VLLM_DP_MASTER_IP='127.0.0.1',VLLM_DP_MASTER_PORT='29673')
 from vllm.plugins import load_general_plugins
 load_general_plugins()
 if is_p:import checkpoint_entry
 else:import ep6_state_entry
 from betterscale.models.qwen35 import CAPTURE_SIZES
 from vllm import LLM,SamplingParams
 entry='checkpoint_entry' if is_p else 'ep6_state_entry'
 model=None
 try:
  model=LLM(model=MODEL,tensor_parallel_size=2,enable_expert_parallel=not is_p,all2all_backend='flashinfer_all2allv',
   distributed_executor_backend='mp',worker_cls=entry+'.Worker',dtype='bfloat16',max_model_len=8192,
   max_num_seqs=16,max_num_batched_tokens=4096,seed=17,enable_prefix_caching=True,mamba_cache_mode='align',async_scheduling=True,
   additional_config={'enable_cpu_binding':False,'using_live_runtime':True},limit_mm_per_prompt={'image':0,'video':0},
   compilation_config=dict(cudagraph_mode='FULL',cudagraph_capture_sizes=CAPTURE_SIZES,max_cudagraph_capture_size=4096),
   speculative_config=dict(method='mtp',num_speculative_tokens=2),scheduler_cls=entry+'.Scheduler',
   kv_cache_memory_bytes=8<<30,generation_config='vllm')
  core=model.llm_engine.engine_core;connection.send(('ready',role))
  while True:
   op,args=connection.recv()
   if op=='stop':break
   if op=='generate':
    tokens=args['tokens'];salt=args['salt'];n=args['n']
    inp=dict(prompt=tokens,cache_salt=salt) if isinstance(tokens,str) else dict(prompt_token_ids=tokens,cache_salt=salt)
    r=model.generate([inp],SamplingParams(temperature=0,max_tokens=n,ignore_eos=True),use_tqdm=False)[0]
    result=dict(prompt_token_ids=r.prompt_token_ids,token_ids=list(r.outputs[0].token_ids),text=r.outputs[0].text,cached=r.num_cached_tokens)
   elif op=='wake':result=core.call_utility('pd_start_wave')
   elif op=='export':result=core.call_utility('pd_export_target',args['tokens'],args['salt'])
   elif op=='drop':result=core.call_utility('pd_drop_target',args['salt'])
   elif op=='import':result=core.call_utility('pd_import_target',args['payload'],args['salt'])
   elif op=='append':result=args['tokens']+model.get_tokenizer().encode(args['text'],add_special_tokens=False)
   else:raise ValueError(op)
   connection.send(('ok',result))
 except BaseException:
  error=traceback.format_exc();(output/(role+'-failure.txt')).write_text(error)
  try:connection.send(('error',error))
  except (BrokenPipeError,EOFError):pass
  raise
 finally:
  if model is not None:model.llm_engine.engine_core.shutdown()
  connection.close()


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 from dram_store_fixture import dram_store
 from model_store import publish,load,split
 from model_checkpoint import IDENTITY
 from session import Directory,Objects
 ctx=mp.get_context('spawn');children={};pipes={};stage='startup';receipts=[]
 def receive(role,timeout=240):
  pipe=pipes[role]
  if not pipe.poll(timeout):raise TimeoutError(f'{stage}: {role} response deadline')
  status,data=pipe.recv()
  if status=='error':raise RuntimeError(f'{role}: {data}')
  return data
 def call(role,op,**kwargs):
  pipes[role].send((op,kwargs));return receive(role)
 def generate(role,name,tokens,salt,n):
  if role=='P':result=call(role,'generate',tokens=tokens,salt=salt,n=n)
  else:
   # Send the real request first; it can wait in collectives until peers wake.
   pipes[role].send(('generate',dict(tokens=tokens,salt=salt,n=n)))
   peers=[r for r in pipes if r.startswith('D') and r!=role]
   for peer in peers:pipes[peer].send(('wake',{}))
   for peer in peers:receive(peer)
   result=receive(role)
  (a.output/(name+'.json')).write_text(json.dumps(result,indent=2));print('PD_PHASE',role,name,result['cached'],flush=True)
  return result
 try:
  with dram_store(a.output/'store',55401) as stores:
   directory=Directory(a.output/'directory.sqlite');objects={'P':Objects(stores[0]),'D':Objects(stores[1])}
   for role in ('P','D0','D1','D2'):
    parent,child=ctx.Pipe();process=ctx.Process(target=worker,args=(role,child,a.output),name='pd-'+role)
    process.start();child.close();pipes[role]=parent;children[role]=process
   deadline=time.monotonic()+900
   for role in pipes:assert receive(role,max(1,deadline-time.monotonic()))==role
   print('PD_ALL_ENGINES_READY',flush=True)
   def handoff(source,dest,tokens,salt,lease):
    start=time.monotonic();payload=call(source,'export',tokens=tokens,salt=salt)
    streams=tuple(sorted(split(payload)[0]));source_objects=objects['P' if source=='P' else 'D'];dest_objects=objects['P' if dest=='P' else 'D']
    key=publish(directory,source_objects,lease,payload,dest)
    assert call(source,'drop',salt=salt),'Expected source resident retirement'
    del payload
    restored=load(dest_objects,key,IDENTITY,streams)
    installed=call(dest,'import',payload=restored,salt=salt)
    row=dict(source=source,destination=dest,cursor=len(tokens)-1,seconds=time.monotonic()-start,
      seat=installed['seat'],draft_state_transferred=False)
    receipts.append(row);print('PD_HANDOFF',json.dumps(row),flush=True)
   for rank,repeats in enumerate((220,12,1)):
    owner=f'D{rank}';sid=f'pd-session-{rank}';salt=sid;directory.create(sid,IDENTITY,'P')
    stage=sid+' first prefill';lease=directory.claim(sid,'P',IDENTITY)
    prompt=('Observation: the blue key opens the north gate; the red key opens the south gate.\n'*repeats)+'\nQuestion: Which key opens the north gate?\nAnswer:'
    first=generate('P',sid+'-prefill',prompt,salt,1);tokens=first['prompt_token_ids']+first['token_ids']
    stage=sid+' P to D';handoff('P',owner,tokens,salt,lease)
    lease=directory.claim(sid,owner,IDENTITY);stage=sid+' first decode'
    decoded=generate(owner,sid+'-decode',tokens,salt,16)
    assert decoded['cached']==len(tokens)-1
    tokens+=decoded['token_ids'];stage=sid+' D to P';handoff(owner,'P',tokens,salt,lease)
    lease=directory.claim(sid,'P',IDENTITY);stage=sid+' second prefill'
    prompt=call('P','append',tokens=tokens,text='\nWhich key opens the south gate?\nAnswer:')
    second=generate('P',sid+'-prefill2',prompt,salt,1)
    assert second['cached']==len(tokens)-1
    tokens=second['prompt_token_ids']+second['token_ids'];handoff('P',owner,tokens,salt,lease)
    lease=directory.claim(sid,owner,IDENTITY);stage=sid+' second decode'
    warm=generate(owner,sid+'-decode2',tokens,salt,16)
    cold=generate(owner,sid+'-cold2',tokens,salt+'-cold',16)
    row=dict(session=sid,owner=owner,equal=warm['token_ids']==cold['token_ids'],cached=warm['cached'],cursor=len(tokens)-1,cold_cached=cold['cached'])
    (a.output/(sid+'-receipt.json')).write_text(json.dumps(row,indent=2))
    assert row['equal'] and row['cached']==row['cursor'] and cold['cached']==0,row
    stage=sid+' final checkpoint';handoff(owner,'P',tokens+warm['token_ids'],salt,lease)
   (a.output/'complete.json').write_text(json.dumps(dict(status='passed',scope='P2 D6 actual-model DRAM P-D-P-D-P target-only handoff, three fixed attention owners; no DMA overlap/production HA claim',handoffs=receipts),indent=2))
 except BaseException:
  (a.output/'failure.json').write_text(json.dumps(dict(stage=stage,error=traceback.format_exc(),handoffs=receipts),indent=2));raise
 finally:
  for role,pipe in pipes.items():
   try:pipe.send(('stop',{}))
   except (BrokenPipeError,EOFError,OSError):pass
  deadline=time.monotonic()+60
  for process in children.values():process.join(timeout=max(0,deadline-time.monotonic()))
  for process in children.values():
   if process.is_alive():process.terminate()
  for process in children.values():process.join(timeout=10)
  for pipe in pipes.values():pipe.close()

if __name__=='__main__':main()
