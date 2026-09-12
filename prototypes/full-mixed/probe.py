"""Small dummy DSV4 model with real attention dimensions, all compression families."""
import argparse,json,time,os
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--output',type=Path,required=True)
p.add_argument('--mode',choices=['FULL','FULL_DECODE_ONLY','NONE'],default='FULL')
p.add_argument('--tp',type=int,default=2)
p.add_argument('--spec',action='store_true')
p.add_argument('--budget',type=int,default=256)
p.add_argument('--rounds',type=int,default=1)
p.add_argument('--real',action='store_true')
p.add_argument('--kv-gib',type=float)
p.add_argument('--policy-study',action='store_true')
p.add_argument('--profile-after',action='store_true')
p.add_argument('--output-tokens',type=int,default=64)
p.add_argument('--cpu-qli',action='store_true')
p.add_argument('--verify-qli',action='store_true')
p.add_argument('--draft-graph',action='store_true')
p.add_argument('--replay-study',action='store_true')
p.add_argument('--ordered-replay',action='store_true')
p.add_argument('--decode-study',action='store_true')
p.add_argument('--profile',action='store_true')
a=p.parse_args()
assert sum((a.decode_study,a.replay_study,a.policy_study))<=1
a.output.mkdir(parents=True,exist_ok=True)
from vllm import LLM,SamplingParams
config=dict(model=os.environ.get('PROBE_MODEL','/data/shared_models/DeepSeek-V4-Flash-0731-w8a8'),load_format='dummy',
 tensor_parallel_size=a.tp,enable_expert_parallel=a.tp>1,quantization='ascend',dtype='bfloat16',
 hf_overrides=dict(num_hidden_layers=4,compress_ratios=[0,0,4,128],n_routed_experts=8,num_hash_layers=0,num_nextn_predict_layers=0),
 max_model_len=max(2048,a.budget*2),max_num_batched_tokens=a.budget,max_num_seqs=4,block_size=128,
 gpu_memory_utilization=.35,num_gpu_blocks_override=max(4096,a.budget*4),enable_prefix_caching=False,skip_tokenizer_init=True,seed=123,
 worker_extension_cls='extension.FullMixedProbeWorker',
 compilation_config=dict(cudagraph_mode=a.mode,cudagraph_capture_sizes=[24 if a.spec else 8,a.budget],max_cudagraph_capture_size=a.budget),
 additional_config=dict(ascend_compilation_config=dict(enable_npugraph_ex=True,enable_static_kernel=False),
 enable_cpu_binding=False,enable_dsa_cp=True,multistream_overlap_shared_expert=True))
if a.spec and not a.real:
 from fixture import install_dummy_draft_config
 install_dummy_draft_config()
 config['hf_overrides'].update(num_nextn_predict_layers=1,dspark_target_layer_ids=[1,2,3])
if a.spec:
 config['speculative_config']=dict(method='dspark',num_speculative_tokens=5,enforce_eager=True)
if a.real:
 config['load_format']='auto'
 config.pop('hf_overrides')
 config.pop('num_gpu_blocks_override')
 config['gpu_memory_utilization']=.85
if a.kv_gib is not None:
 config['kv_cache_memory_bytes']=int(a.kv_gib*1024**3)
 config.pop('num_gpu_blocks_override',None)
(a.output/'config.json').write_text(json.dumps(config,indent=2,default=lambda x:x.__module__+"."+x.__name__))
(a.output/'protocol.json').write_text(json.dumps(dict(
 patch=os.environ.get('FULL_MIXED_PATCH','1'),oracle=os.environ.get('FULL_MIXED_ORACLE','graph'),
 shadow=os.environ.get('FULL_MIXED_SHADOW','0'),hccl_deterministic=os.environ.get('HCCL_DETERMINISTIC'),
 devices=os.environ.get('ASCEND_RT_VISIBLE_DEVICES'),real_weights=a.real,rounds=a.rounds,
 ordered_replay=a.ordered_replay,cpu_qli=a.cpu_qli,verify_qli=a.verify_qli,draft_graph=a.draft_graph,
 draft_shadow=os.environ.get('DRAFT_GRAPH_SHADOW','0'),policy_study=a.policy_study,decode_study=a.decode_study,replay_study=a.replay_study,profile=a.profile,profile_after=a.profile_after,output_tokens=a.output_tokens),indent=2))
os.environ['FULL_MIXED_OUTPUT']=str(a.output)
llm=LLM(**config)
if os.environ.get('FULL_MIXED_SHADOW')=='1':llm.collective_rpc('enable_shadow')
if a.ordered_replay:llm.collective_rpc('set_ordered_replay',args=(True,))
if a.draft_graph:llm.collective_rpc("enable_exact_draft_graph")
if a.cpu_qli:llm.collective_rpc("set_cpu_qli",args=(True,a.verify_qli))
results=[]
if a.decode_study or a.replay_study or a.policy_study:
 llm.generate([dict(prompt_token_ids=[17]*64)]*4,SamplingParams(temperature=0,max_tokens=8,ignore_eos=True,detokenize=False))
 if not (a.replay_study or a.policy_study):llm.collective_rpc('start_decode_observation',args=(a.profile,))
cohorts = ([[64]*4]*a.rounds if (a.decode_study or a.replay_study or a.policy_study) else ([[64,64,64,64],[129,17],[7,128,33],[256,23,5,9],[513,257,17],[1025],[a.budget+17,19]] * a.rounds))
for phase_index,lengths in enumerate(cohorts):
 if a.policy_study:
  policy=phase_index%3
  llm.collective_rpc("set_ordered_replay",args=(policy>0,))
  llm.collective_rpc("set_cpu_qli",args=(policy>0,False))
  llm.collective_rpc("enable_exact_draft_graph",args=(policy==2,))
  llm.generate([dict(prompt_token_ids=[17+i]*64) for i in range(4)],SamplingParams(temperature=0,max_tokens=8,ignore_eos=True,detokenize=False))
  llm.collective_rpc("start_decode_observation",args=(False,f"phase{phase_index}"))
 if a.replay_study:
  llm.collective_rpc("set_ordered_replay",args=(bool(phase_index%2),))
  llm.collective_rpc("start_decode_observation",args=(False,f"phase{phase_index}"))
 prompts=[dict(prompt_token_ids=[17+i]*n) for i,n in enumerate(lengths)]
 t=time.monotonic();out=llm.generate(prompts,SamplingParams(temperature=0,max_tokens=a.output_tokens if (a.decode_study or a.replay_study or a.policy_study) else 8,ignore_eos=True,detokenize=False))
 results.append(dict(phase=phase_index,policy=(phase_index%3 if a.policy_study else bool(phase_index%2) if a.replay_study else None),lengths=lengths,elapsed=time.monotonic()-t,outputs=[x.outputs[0].token_ids for x in out]))
 if a.replay_study or a.policy_study:llm.collective_rpc('stop_decode_observation')
 (a.output/'partial.json').write_text(json.dumps(results,indent=2))
if a.decode_study:llm.collective_rpc('stop_decode_observation')
if a.profile_after:
 llm.collective_rpc('start_decode_observation',args=(True,'profile'))
 llm.generate([dict(prompt_token_ids=[17+i]*64) for i in range(4)],SamplingParams(temperature=0,max_tokens=16,ignore_eos=True,detokenize=False))
 llm.collective_rpc('stop_decode_observation')
receipts=llm.collective_rpc('graph_receipt')
(a.output/'result.json').write_text(json.dumps(dict(status='COMPLETED_NOT_NUMERICALLY_QUALIFIED',results=results,receipts=receipts),indent=2))

if a.mode == 'FULL':
 for rank in receipts:
  entries=[x for w in rank['wrappers'] for x in w['entries']]
  assert any(x['tokens']>=a.budget and x['captured'] and x['replays']>0 for x in entries), 'large FULL bucket absent or never replayed'
