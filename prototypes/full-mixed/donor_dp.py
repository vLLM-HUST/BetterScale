"""Bounded native offline DP+EP study, following the pinned upstream DP example.

Each subprocess is a native model client, not an autonomous agent. It receives a
fixed request shard; native donor owns scheduling, dummy waves and collectives.
"""
import json
import multiprocessing as mp
import os
from pathlib import Path
import time


def rank_main(args,dp_rank,barrier):
    os.environ.update(VLLM_DP_RANK=str(dp_rank),VLLM_DP_RANK_LOCAL=str(dp_rank),
                     VLLM_DP_SIZE=str(args.donor_dp),VLLM_DP_MASTER_IP='127.0.0.1',
                     VLLM_DP_MASTER_PORT='30651',DONOR_DP_OUTPUT=str(args.output))
    os.environ['VLLM_ASCEND_ENABLE_FLASHCOMM1']='1' if args.tp>1 else '0'
    os.environ['DONOR_PINGPONG']='1' if args.pingpong else '0'
    os.environ['DONOR_PINGPONG_SHADOW']='1' if args.pingpong_shadow else '0'
    from vllm import LLM,SamplingParams
    if not args.real:
        from fixture import install_dummy_draft_config
        install_dummy_draft_config()
    single_card = args.donor_dp == args.tp == 1
    local_seats=(4 if args.tp > 1 else 2) if not args.real else 16//args.donor_dp
    capture_sizes=[6,12] if args.tp==1 else [24,48,96]
    if args.dp_full:
        assert args.tp == 1
        # Include the smallest K5-aligned bucket covering the unchanged budget.
        # Runner aligns down sizes that exceed max_num_batched_tokens, so the
        # budget itself is explicitly K5-aligned for the candidate/control pair.
        assert args.budget % 6 == 0
        capture_sizes += [n for n in (132, 264, 516, args.budget) if 12 < n <= args.budget]
        capture_sizes = sorted(set(capture_sizes))
    config=dict(model=os.environ['PROBE_MODEL'],load_format='auto' if args.real else 'dummy',
                tensor_parallel_size=args.tp,enable_expert_parallel=True,quantization='ascend',dtype='bfloat16',
                max_model_len=16384,max_num_batched_tokens=args.budget,max_num_seqs=local_seats,
                kv_cache_memory_bytes=int((args.kv_gib or 3)*1024**3),enable_prefix_caching=False,
                skip_tokenizer_init=True,seed=123,block_size=128,
                worker_extension_cls='dp_full.DPFullWorker' if args.dp_full else 'donor_dp_worker.DonorDPWorker',
                compilation_config=dict(cudagraph_mode='FULL' if args.dp_full else 'FULL_DECODE_ONLY',
                    cudagraph_capture_sizes=capture_sizes,max_cudagraph_capture_size=capture_sizes[-1]),
                additional_config=dict(ascend_compilation_config=dict(enable_npugraph_ex=True,enable_static_kernel=False),
                    enable_cpu_binding=False,enable_dsa_cp=args.tp>1,multistream_overlap_shared_expert=True))
    if args.spec:
        config['speculative_config']=dict(method='dspark',num_speculative_tokens=5,enforce_eager=True)
    if not args.real:
        config['hf_overrides']=dict(num_hidden_layers=4,compress_ratios=[0,0,4,128],n_routed_experts=8,
                                   num_hash_layers=0,num_nextn_predict_layers=1,dspark_target_layer_ids=[1,2,3])
    (args.output/f'dp{dp_rank}-config.json').write_text(json.dumps(config,indent=2))
    llm=LLM(**config)
    if args.pingpong:
        llm.collective_rpc('enable_pingpong')
    if args.pingpong_sources:
        assert args.pingpong
        llm.collective_rpc('enable_pingpong_sources')
    if args.pingpong_continuous:
        assert args.pingpong and args.pingpong_sources
        llm.collective_rpc('enable_pingpong_continuous')
    if args.pingpong_shadow:
        assert args.pingpong and not args.dp_shadow
        llm.collective_rpc('enable_pingpong_shadow')
    if args.dp_shadow:
        assert args.dp_full
        llm.collective_rpc('enable_dp_shadow')
    rows=[]
    def wave(label,lengths,output,profile=False,observe=True):
        barrier.wait(timeout=600)
        if observe:llm.collective_rpc('start_window',args=(label,profile))
        barrier.wait(timeout=120)
        started=time.monotonic()
        result=llm.generate([dict(prompt_token_ids=[17+(dp_rank*len(lengths)+i)%4]*n) for i,n in enumerate(lengths)],
                            SamplingParams(temperature=0,max_tokens=output,ignore_eos=True,detokenize=False))
        elapsed=time.monotonic()-started
        # Keep native engines alive after this rank finishes: other EP ranks
        # may still need its dummy/collective service while draining.
        barrier.wait(timeout=600)
        if observe:llm.collective_rpc('stop_window')
        record=dict(label=label,dp_rank=dp_rank,lengths=lengths,started_monotonic=started,elapsed=elapsed,
                    outputs=[list(x.outputs[0].token_ids) for x in result],profile=profile)
        rows.append(record);(args.output/f'dp{dp_rank}-results.json').write_text(json.dumps(rows,indent=2))
        barrier.wait(timeout=120)
    wave('warmup',[64]*local_seats,16,observe=False)
    if args.real:
        for repeat in range(2):
            wave(f'decode{repeat}',[128]*local_seats,128)
            wave(f'prefill{repeat}',[4096]*(8//args.donor_dp),16)
            wave(f'skew{repeat}',[8192 if dp_rank*(8//args.donor_dp)+i==0 else 256 for i in range(8//args.donor_dp)],16)
    elif not args.real:
        wave('dummybalanced',[128]*local_seats,16)
        wave('dummyskew',[args.budget*2+17 if dp_rank==0 else 32]*(1 if single_card else 8//args.donor_dp),16)
    if args.profile_after:
        wave('profiledecode',[128]*local_seats,64,profile=True)
        wave('profileskew',[8192 if dp_rank*(8//args.donor_dp)+i==0 else 256 for i in range(8//args.donor_dp)],32,profile=True)
    if args.quality_requests:
        assert args.real and args.donor_dp == 8 and args.tp == 1 and not args.dp_shadow
        from quality import run_dp
        run_dp(llm,args.output,args.quality_requests,dp_rank,args.donor_dp,barrier)
    receipt=llm.collective_rpc('donor_receipt')
    (args.output/f'dp{dp_rank}-receipt.json').write_text(json.dumps(receipt,indent=2))
    barrier.wait(timeout=120)
    time.sleep(1)  # upstream offline-DP drain convention


def main(args):
    assert args.spec, "This bounded DP comparison is a K5 study"
    assert not any((args.n2,args.split_draft,args.ordered_replay,args.cpu_qli,args.draft_graph,args.cross_step_bounds))
    assert (args.donor_dp*args.tp==8 and args.donor_dp in (1,2,4,8)) or (
        args.tp == 1 and args.donor_dp in (1,2,4) and args.dp_full and not args.real)
    args.output.mkdir(parents=True,exist_ok=True)
    ctx=mp.get_context('spawn');barrier=ctx.Barrier(args.donor_dp)
    processes=[ctx.Process(target=rank_main,args=(args,i,barrier),name=f'donor-dp-client-{i}') for i in range(args.donor_dp)]
    for process in processes:process.start()
    while any(p.is_alive() for p in processes):
        failed=[p for p in processes if p.exitcode not in (None,0)]
        if failed:
            barrier.abort()
            raise RuntimeError(f'DP client failed: {[(p.name,p.exitcode) for p in failed]}')
        for p in processes:p.join(timeout=.2)
    assert all(p.exitcode==0 for p in processes)
    if args.quality_requests:
        requests = json.loads(Path(args.quality_requests).read_text())
        results = [r for i in range(args.donor_dp) for r in json.loads((args.output/f'dp{i}-quality.json').read_text())]
        assert len(results) == 32
        (args.output/'quality-inputs.json').write_text(json.dumps(requests))
        (args.output/'quality-result.json').write_text(json.dumps(dict(status='COMPLETED_UNSCORED',
            scope='Retained 32 OpenCompass LongBench English retrieval items; native DP8',requests=results),indent=2))
    (args.output/'dp-completion.json').write_text(json.dumps(dict(dp=args.donor_dp,tp=args.tp,exitcodes=[p.exitcode for p in processes]),indent=2))


if __name__ == '__main__':
    import sys
    from types import SimpleNamespace
    args=SimpleNamespace(**json.loads(sys.argv[1]))
    args.output=Path(args.output)
    main(args)
