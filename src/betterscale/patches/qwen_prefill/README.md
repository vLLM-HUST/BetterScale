# Qwen hybrid FULL prefill (independent opt-in)

Historical native-layout single-prefill implementation. The unified public Worker
no longer selects this route; no-MTP Qwen uses owned GDN, while native MTP2 uses
weight packing only. Commands and measurements below describe the old entry,
not current deployment instructions. See `../../models/README.md`.

Entry: `betterscale.qwen_worker.Worker`, not the DSV4 Worker. This bounded path
keeps native scheduling, model, sampling, decode graphs, and mixed-batch fallback.
It does not claim to replace the runner with the separate LiveInference reactor.

Single-request prefills round **up** to16/32/64/128/256/512/1024/1536/2048.
GDN metadata tensors retain stable addresses. Convolution uses the real endpoint;
virtual recurrent tokens have zero q/k/v/g/beta. FIA's empty virtual request must
not become a second GDN state row. Request count is part of graph identity; capture
uses one prefill without reducing the eight live scheduling seats.

Admission is intentionally narrow: pinned donor, Qwen3.5-text architecture matching
the local Qwen3.8-27B weights, BF16, TP2/DP1/PP1, no EP/context parallelism, text-only,
8seats,2048token budget, context<=8192, async native scheduler, APC off.
Without speculation it enables the FULL prefill path described here. With native
MTP2 it instead leaves prefill/decode/draft execution native and packs the48 fixed
GDN convolution weights once, removing their repeated device transposes. These
are separate routes: FULL-prefill plus MTP remains experimental. APC is not admitted.
Changing physical padding can perturb BF16 trajectories even without graph capture.

Historical prototype evidence (not a fresh package benchmark): same-process512-token
prefill ABBA TTFT359.645→178.366ms. Profile rank0 compute union~118ms unchanged,
model body405.864→162.522ms, uncovered compute/comm247.872→4.675ms; peer waiting also
falls. Profile overhead is not HTTP latency. Long2048-prefill compute hid the gain.
`padded-full3` then exercised6lengths512/513/1024/1536/2048/2051 and mixedC4/C8;
all36single-request32-token continuations matched its unpadded control. This is not
population-wide accuracy, bitwise equality for every input, or general mixed FULL.

Native invocation (supply your model path/environment/KV budget):

```bash
vllm serve "$MODEL" --worker-cls betterscale.qwen_worker.Worker \
  --tensor-parallel-size 2 --distributed-executor-backend mp --dtype bfloat16 \
  --max-model-len 8192 --max-num-seqs 8 --max-num-batched-tokens 2048 \
  --async-scheduling --no-enable-prefix-caching \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  --compilation-config '{"cudagraph_mode":"FULL","cudagraph_capture_sizes":[1,2,4,8,16,32,64,128,256,512,1024,1536,2048],"max_cudagraph_capture_size":2048}'
```

No debug RPC, profiler, custom allocator, environment mutation, automatic donor
upgrade, or native DSV4 patches are installed by this entry. Weight layout changes
only in the explicit native MTP2 route; logical values/Parameter identity are retained.
This source addition is not a PyPI release.

Frozen public-entry acceptance `package2` (source `ebe3725`) passed12 C1 requests
(two rounds of the six lengths above) plus C4/C8 mixed cohorts. All12 C1
continuations matched the qualified prototype. Subsequent changes add donor pins and a separate MTP2 route; the non-speculative
metadata, dispatch and padding code are unchanged.70CPU tests pass. Fresh local wheel/sdist delivery is checked separately;
no installed runtime or published package was replaced.

For native MTP2 + convolution-weight packing, use the same common TP2/text/async/APC-off
settings, omit the FULL capture configuration, and supply:

```bash
--speculative-config '{"method":"mtp","num_speculative_tokens":2}'
```

Keep native FULL_AND_PIECEWISE and its default capture set (maximum24tokens for
8seats). Do not use the large non-speculative FULL prefill capture set with MTP.
The native MTP route does not install this directory's FULL metadata hooks.

Native MTP2 route qualification: `conv-mtp2-2` removes48weightTranspose kernels
per targetstep on bothranks; a short native MTP2 control confirms target-body means
36.253→35.367ms (rank0),36.268→35.445ms (rank1), four exact graphs each. This is
~.8-.9ms saved per targetstep, not per accepted output token. E2E concurrent
throughput varies across runs; no stable C4/C8 percentage is promised.
`package-mtp2` passes both cohorts at C1/C4/C8; C1texts match the prototype.

**Use a dedicated native compile cache for the text-only MTP route**, e.g. set
`VLLM_CACHE_ROOT="$HOME/.cache/vllm-betterscale-qwen-mtp2-text"` before launching.
The pinned donor reused a multimodal tensor-input AOT artifact for text-only
`inputs_embeds=None` and failed at startup with `NoneType.size`. A fresh isolated
cache passes. Do not delete a shared cache or silently modify donor files. This
is an upstream cache/signature limitation, not fixed by the weight-layout patch.
