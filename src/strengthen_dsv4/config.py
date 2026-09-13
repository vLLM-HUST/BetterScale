"""The bounded launch contract. No vLLM imports, GPU work or global mutation."""
from pathlib import Path

PROFILES = ('baseline', 'optimized')
PATCH_IDS = ('compat-lcm', 'target-full', 'ordered-replay', 'cpu-qli',
             'private-draft-banks', 'split-draft-context', 'stable-receipt-cut')


def engine_options(model, artifacts, profile='optimized', kv_gib=12):
    if profile not in PROFILES:
        raise ValueError(f'Unknown patch profile: {profile}')
    if not 0 < kv_gib <= 64:
        raise ValueError('KV GiB must be positive and at most64 per card')
    return dict(
        model=str(model), tensor_parallel_size=8, enable_expert_parallel=True,
        quantization='ascend', dtype='bfloat16', max_model_len=15104,
        max_num_batched_tokens=4128, max_num_seqs=4,
        kv_cache_memory_bytes=int(kv_gib * 1024**3), enable_prefix_caching=False,
        block_size=128, seed=123, worker_cls='strengthen_dsv4.worker.Worker',
        speculative_config=dict(method='dspark', num_speculative_tokens=5, enforce_eager=True),
        compilation_config=dict(cudagraph_mode='FULL' if profile=='optimized' else 'FULL_DECODE_ONLY',
                                cudagraph_capture_sizes=[24,4128], max_cudagraph_capture_size=4128),
        additional_config=dict(
            ascend_compilation_config=dict(enable_npugraph_ex=True, enable_static_kernel=False),
            enable_cpu_binding=False, enable_dsa_cp=True, multistream_overlap_shared_expert=True,
            strengthen_dsv4=dict(profile=profile, artifacts=str(Path(artifacts).resolve()))))


def validate_worker_config(config):
    """Reject unqualified combinations before allocating model weights."""
    p=config.parallel_config; s=config.scheduler_config; m=config.model_config
    spec=config.speculative_config; extra=config.additional_config
    policy=extra.get('strengthen_dsv4', {})
    profile=policy.get('profile')
    if profile not in PROFILES or not policy.get('artifacts'):
        raise ValueError('Use the strengthen-dsv4 launcher or engine_options()')
    checks={
        'TP8, DP1, PP1, EP': (p.tensor_parallel_size,p.data_parallel_size,p.pipeline_parallel_size,p.enable_expert_parallel)==(8,1,1,True),
        'DCP1, PCP1': p.decode_context_parallel_size==1 and p.prefill_context_parallel_size==1,
        'DSV4 Flash43 layers /4096 hidden /256 experts':
            m.hf_config.model_type=='deepseek_v4' and
            (m.hf_config.num_hidden_layers,m.hf_config.hidden_size,m.hf_config.n_routed_experts)==(43,4096,256),
        'four seats /4128 budget /<=15104 context':
            s.max_num_seqs==4 and s.max_num_batched_tokens==4128 and m.max_model_len<=15104,
        'DSpark K5 with native eager drafter': spec is not None and spec.method=='dspark' and spec.num_speculative_tokens==5 and spec.enforce_eager,
        'standard rejection': spec is not None and spec.rejection_sample_method=='standard',
        'DSACP enabled': extra.get('enable_dsa_cp') is True,
        'prefix caching disabled': not config.cache_config.enable_prefix_caching,
        'real W8A8 model': config.load_config.load_format=='auto' and m.quantization=='ascend',
        'native scheduler': s.scheduler_cls is None,
        'explicit graph mode': str(config.compilation_config.cudagraph_mode)==('FULL' if profile=='optimized' else 'FULL_DECODE_ONLY'),
    }
    failed=[name for name,passed in checks.items() if not passed]
    if failed:
        raise ValueError('Outside the qualified strengthen-dsv4 envelope: '+ '; '.join(failed))
    return policy
