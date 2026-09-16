# Qwen hybrid FULL prefill (independent opt-in)

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
8seats,2048token budget, context<=8192, async native scheduler, APC off, no MTP.
Do not silently enable speculative decoding or prefix caching. Both remain research
fronts; the native MTP2 configuration is a separate useful serving alternative.
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
upgrade, weight-layout change, or native DSV4 patches are installed by this entry.
This source addition is not a PyPI release.
