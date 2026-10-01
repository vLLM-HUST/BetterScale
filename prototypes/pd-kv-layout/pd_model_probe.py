"""Single-host P2/D6 target-only handoff through real Mooncake DRAM.

Quiescent session turns, one controller/SQLite directory, no overlap or production
availability claim. D attention owners are fixed per session; MTP never transfers.
"""
import argparse,json,multiprocessing as mp,os,time,traceback,uuid
from pathlib import Path
ROOT=Path('/workspace/betterscale-pd-runtime')
MODEL=os.environ.get('BETTERSCALE_MODEL_PATH','/data/shared_models/modelscope_cache/Qwen/Qwen3.5-35B-A3B')

def prepare_worker(role,*,native_async=False):
 import sys
 is_p=role=='P';rank=0 if is_p or role=='D' else int(role[1:])
 package=ROOT/('candidate-package-6-no-draft' if is_p else 'candidate-package-8-fia-padding')
 if not is_p and os.environ.get('BETTERSCALE_PD_D_PACKAGE'):package=Path(os.environ['BETTERSCALE_PD_D_PACKAGE'])
 runtime=ROOT/('owned-runtime' if is_p else 'ep6-runtime')
 sys.path[:0]=[str(package),str(runtime)]
 os.environ['PYTHONPATH']=f'{package}:{runtime}:'+os.environ.get('PYTHONPATH','')
 os.environ.update(ASCEND_RT_VISIBLE_DEVICES='0,1' if is_p else '2,3,4,5,6,7',
  HCCL_IF_BASE_PORT='29635' if is_p else '29675',VLLM_DP_RANK=str(rank),VLLM_DP_RANK_LOCAL=str(rank),
  VLLM_DP_SIZE='1' if is_p else '3',VLLM_DP_MASTER_IP='127.0.0.1',VLLM_DP_MASTER_PORT='29673')
 if native_async:
  for key in ('VLLM_DP_RANK','VLLM_DP_RANK_LOCAL','VLLM_DP_SIZE','VLLM_DP_MASTER_IP','VLLM_DP_MASTER_PORT'):os.environ.pop(key,None)
 return is_p

def engine_options(is_p):
 from vllm.plugins import load_general_plugins
 load_general_plugins()
 if is_p:import checkpoint_entry
 else:import ep6_state_entry
 from betterscale.models.qwen35 import CAPTURE_SIZES
 entry='checkpoint_entry' if is_p else 'ep6_state_entry'
 return dict(model=MODEL,tensor_parallel_size=2,enable_expert_parallel=not is_p,all2all_backend='flashinfer_all2allv',
   distributed_executor_backend='mp',worker_cls=entry+'.Worker',dtype='bfloat16',max_model_len=8192,
   max_num_seqs=16,max_num_batched_tokens=4096,seed=17,enable_prefix_caching=True,mamba_cache_mode='align',async_scheduling=True,
   additional_config={'enable_cpu_binding':False,'using_live_runtime':True},limit_mm_per_prompt={'image':0,'video':0},
   compilation_config=dict(cudagraph_mode='FULL',cudagraph_capture_sizes=CAPTURE_SIZES,max_cudagraph_capture_size=4096),
   speculative_config=dict(method='mtp',num_speculative_tokens=2),scheduler_cls=entry+'.Scheduler',
   kv_cache_memory_bytes=8<<30,generation_config='vllm')

def worker(role,connection,output):
 is_p=prepare_worker(role)
 options=engine_options(is_p)
 from vllm import LLM,SamplingParams
 model=None
 try:
  model=LLM(**options)
  core=model.llm_engine.engine_core;connection.send(('ready',role))
  while True:
   op,args=connection.recv()
   if op=='stop':break
   if op=='generate':
    tokens=args['tokens'];salt=args['salt'];n=args['n']
    inp=dict(prompt=tokens,cache_salt=salt) if isinstance(tokens,str) else dict(prompt_token_ids=tokens,cache_salt=salt)
    r=model.generate([inp],SamplingParams(temperature=0,max_tokens=n,ignore_eos=True,logprobs=5 if args.get('diagnostic') else None),use_tqdm=False)[0]
    result=dict(prompt_token_ids=r.prompt_token_ids,token_ids=list(r.outputs[0].token_ids),text=r.outputs[0].text,cached=r.num_cached_tokens,logprobs=[{str(k):v.logprob for k,v in row.items()} for row in r.outputs[0].logprobs] if r.outputs[0].logprobs is not None else None)
   elif op=='wake':result=core.call_utility('pd_start_wave')
   elif op=='export':result=core.call_utility('pd_export_retired',args['tokens'],args['salt'],args.get('dense_start',0),args.get('stream_store'))
   elif op=='wait-export':result=core.call_utility('pd_wait_export',args['transfer_id'])
   elif op=='finish-export':result=core.call_utility('pd_finish_export',args['transfer_id'])
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
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--incremental',action='store_true');p.add_argument('--streamed',action='store_true');p.add_argument('--async-export',action='store_true');p.add_argument('--serialize-export',action='store_true');p.add_argument('--verify-transfer',action='store_true');p.add_argument('--native-async',action='store_true');p.add_argument('--import-failure-probe',action='store_true');p.add_argument('--stream-import',action='store_true');p.add_argument('--direct-checkpoint',action='store_true');p.add_argument('--no-export-activity',action='store_true');p.add_argument('--dp-finish-sync',type=int,choices=(1,4,8,32),default=32);p.add_argument('--logprobs',action='store_true');p.add_argument('--cold-controls',type=int,choices=range(4),default=0);a=p.parse_args();a.async_export=a.async_export or a.serialize_export;a.streamed=a.streamed or a.async_export;a.incremental=a.incremental or a.streamed;a.output.mkdir(parents=True,exist_ok=False)
 os.environ['BETTERSCALE_PD_FINISH_SYNC_STEPS']=str(a.dp_finish_sync)
 if a.dp_finish_sync!=32 and not a.native_async:raise ValueError('Cadence experiment requires native actor')
 if a.no_export_activity and not a.async_export:raise ValueError('No-activity arm requires async export')
 if a.direct_checkpoint and not (a.stream_import and a.async_export):raise ValueError('Direct checkpoint requires async export and streamed ingress')
 if a.stream_import and not a.native_async:raise ValueError('Stream import currently qualified through native async actor only')
 if a.import_failure_probe and not a.native_async:raise ValueError('Import failure probe requires native async actor')
 from dram_store_fixture import dram_store
 from model_store import publish,publish_streamed,load
 from model_checkpoint import IDENTITY
 from session import Directory,Objects
 ctx=mp.get_context('spawn');children={};pipes={};stage='startup';receipts=[]
 def receive(role,timeout=240):
  pipe=pipes['D' if a.native_async and role.startswith('D') else role]
  if not pipe.poll(timeout):raise TimeoutError(f'{stage}: {role} response deadline')
  status,data=pipe.recv()
  if status=='error':raise RuntimeError(f'{role}: {data}')
  return data
 def call(role,op,**kwargs):
  if a.native_async:
   kwargs['owner']=0 if role=='P' else int(role[1:])
  pipes['D' if a.native_async and role.startswith('D') else role].send((op,kwargs));return receive(role)
 def generate(role,name,tokens,salt,n):
  if role=='P' or a.native_async:result=call(role,'generate',tokens=tokens,salt=salt,n=n,diagnostic=a.verify_transfer or a.logprobs)
  else:
   # Send the real request first; it can wait in collectives until peers wake.
   pipes[role].send(('generate',dict(tokens=tokens,salt=salt,n=n,diagnostic=a.verify_transfer or a.logprobs)))
   peers=[r for r in pipes if r.startswith('D') and r!=role]
   for peer in peers:pipes[peer].send(('wake',{}))
   for peer in peers:receive(peer)
   result=receive(role)
  (a.output/(name+'.json')).write_text(json.dumps(result,indent=2));print('PD_PHASE',role,name,result['cached'],flush=True)
  return result
 try:
  with dram_store(a.output/'store',55401) as stores:
   directory=Directory(a.output/'directory.sqlite');objects={'P':Objects(stores[0]),'D':Objects(stores[1])}
   if a.native_async:
    from native_async_pool import worker as pool_worker
   else:pool_worker=worker
   for role in (('P','D') if a.native_async else ('P','D0','D1','D2')):
    parent,child=ctx.Pipe();process=ctx.Process(target=pool_worker,args=(role,child,a.output),name='pd-'+role)
    process.start();child.close();pipes[role]=parent;children[role]=process
   deadline=time.monotonic()+900
   for role in pipes:assert receive(role,max(1,deadline-time.monotonic()))==role
   print('PD_ALL_ENGINES_READY',flush=True)
   def handoff(source,dest,tokens,salt,lease):
    start=time.monotonic();source_objects=objects['P' if source=='P' else 'D'];dest_objects=objects['P' if dest=='P' else 'D']
    phases={};phase_start=start
    def mark(name):
     nonlocal phase_start
     now=time.monotonic();phases[name]=now-phase_start;phase_start=now
    dense_start=json.loads(source_objects.get(lease.base))['cursor'] if a.incremental and lease.base is not None else 0
    port=55421 if source=='P' else 55423+2*int(source[1:])
    stream_store=dict(prefix=f'pd/{lease.session}/{lease.epoch}/{uuid.uuid4().hex}',ports=[port,port+1],asynchronous=a.async_export,verify_transfer=a.verify_transfer,direct_checkpoint=a.direct_checkpoint) if a.streamed else None
    mark('prepare')
    payload=call(source,'export',tokens=tokens,salt=salt,dense_start=dense_start,stream_store=stream_store)
    mark('export_begin')
    activity=None
    if a.async_export:
     ticket=payload
     assert call(source,'drop',salt=salt),'Expected hot eviction while native export pins remain'
     if a.serialize_export:call(source,'wait-export',transfer_id=ticket['transfer_id'])
     mark('hot_drop_and_optional_wait')
     if not a.no_export_activity:
      activity_start=time.monotonic()
      other=generate(source,f'{salt}-export-activity-{lease.epoch}',
        'The access code is MARBLE. What is the access code? Answer:',salt+'-activity-'+str(lease.epoch),8)
      activity=[activity_start,time.monotonic()]
     mark('injected_request')
     payload=call(source,'finish-export',transfer_id=ticket['transfer_id'])
     assert all(n>=1 for n in payload['page_pin_receipt']['refs_at_release'])
    mark('finish_export')
    checkpoint_exports=[s['checkpoint_store'] for s in payload['shards']] if a.direct_checkpoint else []
    control=payload['header'].get('dp_control')
    if a.native_async and source.startswith('D'):
     assert control is not None and control['finish_sync_steps']==a.dp_finish_sync,control
    stream_receipts=[s['dense_store'] for s in payload['shards']] if a.streamed else []
    dense_bytes=sum(s['dense_bytes'] for s in stream_receipts) if a.streamed else sum(len(data[k]['data']) for shard in payload['shards'] for data in shard['layers'].values() if set(data)=={'key','value'} for k in ('key','value'))
    assert dense_bytes==(len(tokens)-1-dense_start)*20480
    key=(publish_streamed if a.streamed else publish)(directory,source_objects,lease,payload,dest)
    streams=tuple(json.loads(source_objects.get(key))['streams'])
    if not a.async_export:assert call(source,'drop',salt=salt),'Expected source resident retirement'
    del payload
    mark('store_publish')
    if a.stream_import:
     from model_stream_import import load_plan
     port=55431 if dest=='P' else 55433+2*int(dest[1:])
     restored=load_plan(dest_objects,key,IDENTITY,streams,[port,port+1],verify=a.verify_transfer)
    else:restored=load(dest_objects,key,IDENTITY,streams)
    mark('restore_plan')
    inject=a.import_failure_probe and not receipts
    installed=call(dest,'import-failure-probe' if inject else 'import',payload=restored,salt=salt)
    mark('import_rpc')
    if inject:(a.output/'import-failure-recovery.json').write_text(json.dumps(installed['failure_probe'],indent=2))
    row=dict(source=source,destination=dest,cursor=len(tokens)-1,seconds=time.monotonic()-start,
      seat=installed['seat'],phase_seconds=phases,source_dp_control=control,draft_state_transferred=False,dense_start=dense_start,dense_bytes=dense_bytes,host_activity_interval=activity,stream_pipeline=[{k:v for k,v in s.items() if k not in ('chunks','streams')} for s in stream_receipts])
    if a.stream_import:row['receiver_streams']=[r['dense_import'] for r in installed['workers']]
    if a.direct_checkpoint:row.update(checkpoint_exports=checkpoint_exports,checkpoint_imports=[r['checkpoint_import'] for r in installed['workers']])
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
    controls=[]
    for control in range(a.cold_controls):
     result=generate(owner,sid+f'-cold-control-{control}',tokens,salt+f'-cold-control-{control}',16)
     assert result['cached']==0
     controls.append(dict(equal_to_first_cold=result['token_ids']==cold['token_ids'],equal_to_warm=result['token_ids']==warm['token_ids']))
    row=dict(session=sid,owner=owner,equal=warm['token_ids']==cold['token_ids'],cached=warm['cached'],cursor=len(tokens)-1,cold_cached=cold['cached'],cold_controls=controls)
    (a.output/(sid+'-receipt.json')).write_text(json.dumps(row,indent=2))
    assert row['equal'] and row['cached']==row['cursor'] and cold['cached']==0,row
    stage=sid+' final checkpoint';handoff(owner,'P',tokens+warm['token_ids'],salt,lease)
   (a.output/'complete.json').write_text(json.dumps(dict(status='passed',incremental_d2h=a.incremental,streamed_d2h_store=a.streamed,async_retired_export=a.async_export,serialize_export=a.serialize_export,verify_transfer=a.verify_transfer,native_async_frontend=a.native_async,streamed_store_h2d=a.stream_import,direct_checkpoint=a.direct_checkpoint,export_activity=not a.no_export_activity,dp_finish_sync=a.dp_finish_sync,logprobs=a.logprobs or a.verify_transfer,cold_controls=a.cold_controls,scope='P2 D6 actual-model DRAM P-D-P-D-P target-only handoff, three fixed attention owners; no compute overlap/production HA claim',handoffs=receipts),indent=2))
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
