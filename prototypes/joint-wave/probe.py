"""One-card native model oracle; not a throughput or whole-scheduler test."""
import json
import os
from pathlib import Path

from fixture import install_dummy_draft_config

install_dummy_draft_config()
from vllm import LLM, SamplingParams

output = Path(os.environ['JOINT_WAVE_OUTPUT'])
output.mkdir(parents=True, exist_ok=True)
config = dict(
    model='/data/shared_models/DeepSeek-V4-Flash-0731-w8a8',
    load_format='dummy', tensor_parallel_size=1, quantization='ascend',
    dtype='bfloat16', enforce_eager=True, max_model_len=1024,
    max_num_batched_tokens=132, max_num_seqs=1, block_size=128,
    kv_cache_memory_bytes=256 * 1024**2,
    enable_prefix_caching=False, skip_tokenizer_init=True, seed=123,
    async_scheduling=False,
    hf_overrides=dict(num_hidden_layers=4, compress_ratios=[0, 0, 4, 128],
                      n_routed_experts=8, num_hash_layers=0,
                      num_nextn_predict_layers=1, dspark_target_layer_ids=[1, 2, 3]),
    speculative_config=dict(method='dspark', num_speculative_tokens=5, enforce_eager=True),
    worker_extension_cls='worker.JointWaveProbeWorker',
    additional_config=dict(enable_cpu_binding=False, enable_dsa_cp=False,
                           multistream_overlap_shared_expert=True),
)
(output / 'config.json').write_text(json.dumps(config, indent=2))
llm = LLM(**config)
llm.collective_rpc('enable_joint_oracle')
results = llm.generate([dict(prompt_token_ids=[17] * 128)],
                       SamplingParams(temperature=0, max_tokens=32,
                                      ignore_eos=True, detokenize=False))
receipts = llm.collective_rpc('joint_receipt')
(output / 'result.json').write_text(json.dumps(dict(
    status='PASS', receipts=receipts,
    outputs=[list(r.outputs[0].token_ids) for r in results],
    scope='TP1 four-layer dummy native whole-wave same-state oracle; not service qualification',
), indent=2))
