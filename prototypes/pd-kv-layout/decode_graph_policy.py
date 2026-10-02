"""Explicit D-only small-graph experiment; not a generic admission relaxation.

D may execute only the uncomputed tail of an already-restored checkpoint plus
native MTP proposals. P keeps the qualified mixed/prefill graph bank. A cold or
bulk-prefill request reaching D is an ownership error, not a fallback workload.
"""
KEYS=(3,6,12,24,40,48)


def capacity(tokens, requests, scheduled, computed, prompts):
    if (not 1<=requests<=16 or len(scheduled)!=requests
            or len(computed)!=requests or len(prompts)!=requests
            or any(not 1<=int(n)<=3 for n in scheduled)
            or sum(map(int,scheduled))!=tokens):
        raise ValueError("decode-only graph requires1..16 bounded tail/verify rows")
    if any(int(c)<int(p)-1 for c,p in zip(computed,prompts)):
        raise ValueError("bulk prefill reached decode-only owner before State restore")
    return next(k for k in KEYS if k>=tokens)


def qualify(config):
    p=config.parallel_config
    if (p.tensor_parallel_size,p.data_parallel_size,p.enable_expert_parallel)!=(2,4,True):
        raise ValueError("decode-only graph experiment requires DP4TP2EP8")
    if (config.scheduler_config.max_num_seqs!=16
            or set(config.compilation_config.cudagraph_capture_sizes)!=set(KEYS)
            or str(config.compilation_config.cudagraph_mode)!="FULL"):
        raise ValueError("decode-only graph configuration differs from admitted keys")
    # The experimental capsules already own the DP4/EP8 admission override.
    # Narrow their graph-key qualification before Worker.validate, rather than
    # disabling validation or touching P's independent process.
    from betterscale.models import qwen35
    qwen35.CAPTURE_SIZES=KEYS


def install():
    from betterscale.patches.qwen_gdn import graphs
    def choose(runner,tokens,requests,scheduled):
        if getattr(runner,"_elastic_dummy",False):
            return tokens  # native startup/profile and synchronized idle shape
        batch=runner.input_batch
        return capacity(tokens,requests,scheduled,
            batch.num_computed_tokens_cpu_tensor[:requests],
            batch.num_prompt_tokens_cpu_tensor[:requests])
    graphs.capacity=choose
