"""Small dummy DSV4 model with real attention dimensions, all compression families."""
import argparse,json,time,os
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--mode',choices=['FULL','FULL_DECODE_ONLY','NONE'],default='FULL');p.add_argument('--tp',type=int,default=2);p.add_argument('--spec',action='store_true');p.add_argument('--budget',type=int,default=256);p.add_argument('--rounds',type=int,default=1);a=p.parse_args()
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
if a.spec:
 from fixture import install_dummy_draft_config
 install_dummy_draft_config()
 config['hf_overrides'].update(num_nextn_predict_layers=1,dspark_target_layer_ids=[1,2,3])
 config['speculative_config']=dict(method='dspark',num_speculative_tokens=5,enforce_eager=True)
(a.output/'config.json').write_text(json.dumps(config,indent=2,default=lambda x:x.__module__+"."+x.__name__))
os.environ['FULL_MIXED_OUTPUT']=str(a.output)
llm=LLM(**config)
if os.environ.get('FULL_MIXED_SHADOW')=='1':llm.collective_rpc('enable_shadow')
results=[]
for lengths in ([[64,64,64,64],[129,17],[7,128,33],[256,23,5,9],[513,257,17],[1025],[a.budget+17,19]] * a.rounds):
 prompts=[dict(prompt_token_ids=[17+i]*n) for i,n in enumerate(lengths)]
 t=time.monotonic();out=llm.generate(prompts,SamplingParams(temperature=0,max_tokens=8,ignore_eos=True,detokenize=False))
 results.append(dict(lengths=lengths,elapsed=time.monotonic()-t,outputs=[x.outputs[0].token_ids for x in out]))
 (a.output/'partial.json').write_text(json.dumps(results,indent=2))
receipts=llm.collective_rpc('graph_receipt')
(a.output/'result.json').write_text(json.dumps(dict(status='COMPLETED_NOT_NUMERICALLY_QUALIFIED',results=results,receipts=receipts),indent=2))

if a.mode == 'FULL':
 for rank in receipts:
  entries=[x for w in rank['wrappers'] for x in w['entries']]
  assert any(x['tokens']>=a.budget and x['captured'] and x['replays']>0 for x in entries), 'large FULL bucket absent or never replayed'
