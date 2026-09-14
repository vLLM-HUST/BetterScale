# BetterScale

**Less host waiting. More device execution.**

BetterScale is a modular execution-optimization package for **vLLM Ascend**.
It adds supported FULL graph paths, ordered replay, and device-side progress and
metadata preparation for DeepSeek V4 Flash. One native Worker entry composes the
patches; it does not replace the serving engine or configure your environment.

[Measured results and mechanisms](https://vllm-hust.sage.org.ai/betterscale.html)

## Install

Use your **existing, working Ascend serving environment**, with Python 3.12+:

```bash
python -m pip install --no-deps vllm-betterscale==0.3.0
```

The package deliberately does not install or upgrade vLLM, vLLM-Ascend, torch-npu,
CANN, model weights, or kernels. Install those through your normal Ascend deployment.

This release is pinned to **vLLM 0.25.1**, **vLLM-Ascend 0.25.1rc1**, and
**torch-npu 2.10.0.post2**. The measured environment uses CANN 9.0.1 and eight
Ascend 910B2 cards connected with HCCS. Worker initialization verifies both versions
and selected upstream source files; a different build may be rejected even if its
version string matches. Do not disable those checks to force an unqualified runtime.

If you previously installed the private `strengthen-dsv4` distribution, stop your
service and uninstall that distribution before installing this one. The new package
retains its implementation namespace, so the two distributions must not be installed
together. The old Worker import remains an alias-compatible entry.

## Start

Keep your native serving arguments and add:

```text
--worker-cls betterscale.worker.Worker
```

This release supports two bounded, single-node configurations—not arbitrary models,
shapes or parallel layouts. The examples below use local model weights at
`/models/DeepSeek-V4-Flash`; replace that path with your Ascend W8A8 checkpoint.
CANN/HCCL and device visibility remain your existing environment's responsibility.
The loopback binding is deliberate; use your normal authentication, TLS and access
controls before exposing the service outside the host.

### TP8 + EP, DSpark K5

Four active requests; token budget 4128; context up to 15104. DSACP is enabled,
DCP/PCP remain 1, and prefix caching is disabled.

```bash
vllm serve /models/DeepSeek-V4-Flash \
  --worker-cls betterscale.worker.Worker \
  --tensor-parallel-size 8 --enable-expert-parallel \
  --quantization ascend --dtype bfloat16 \
  --max-num-seqs 4 --max-num-batched-tokens 4128 --max-model-len 15104 \
  --kv-cache-memory-bytes 12884901888 --no-enable-prefix-caching \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5,"enforce_eager":true}' \
  --compilation-config '{"cudagraph_mode":"FULL","cudagraph_capture_sizes":[24,4128],"max_cudagraph_capture_size":4128}' \
  --additional-config '{"enable_dsa_cp":true,"multistream_overlap_shared_expert":true,"ascend_compilation_config":{"enable_npugraph_ex":true,"enable_static_kernel":false}}' \
  --host 127.0.0.1 --port 8000 --served-model-name dsv4
```

### TP1 / DP8 / EP8, DSpark K5

Two active requests per rank (16 total); local token budget 1026; context up to
16384. DSACP is disabled; target is FULL and draft remains native eager.

```bash
vllm serve /models/DeepSeek-V4-Flash \
  --worker-cls betterscale.worker.Worker \
  --tensor-parallel-size 1 --data-parallel-size 8 --enable-expert-parallel \
  --quantization ascend --dtype bfloat16 \
  --max-num-seqs 2 --max-num-batched-tokens 1026 --max-model-len 16384 \
  --kv-cache-memory-bytes 8589934592 --no-enable-prefix-caching \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5,"enforce_eager":true}' \
  --compilation-config '{"cudagraph_mode":"FULL","cudagraph_capture_sizes":[6,12,132,264,516,1026],"max_cudagraph_capture_size":1026}' \
  --additional-config '{"enable_dsa_cp":false,"multistream_overlap_shared_expert":true,"ascend_compilation_config":{"enable_npugraph_ex":true,"enable_static_kernel":false}}' \
  --host 127.0.0.1 --port 8000 --served-model-name dsv4
```

The 12 GiB / 8 GiB KV budgets above are tested examples, not memory reserved by the
package. Use your native KV budget appropriate to the workload and available HBM.
First encounters with some legal shapes may capture graphs; warmup is not a promise
that every request shape has already been captured.

## Verify and roll back

Check for the per-worker `strengthen-dsv4 rank=... READY patches=...` log entry and
the native health endpoint after model startup:

```bash
curl --fail http://127.0.0.1:8000/health
```

To roll back, stop the service and restart with your known-working native Worker
configuration. Do not hot-unpatch a live process. Some pinned K5/TP8 combinations
need BetterScale's alignment repair, so removing the Worker from exactly the same
command is not guaranteed to produce a runnable unpatched comparator.

## Results and scope

Matched K5 cycles improve by 20–21% in the TP draft-graph study, with a separate
10–12% incremental receipt-handling improvement. DP device preparation improves
matched cycle time by 15.4–15.9% against dual graph endpoints. Those are different
controlled studies, not additive percentages or stable whole-service throughput
claims. The retained OpenCompass LongBench English retrieval subset passes 32/32;
this is not the full OpenCompass suite. See the linked case study for all repeats,
configurations and boundaries.

The distribution contains Python source and upstream compatibility pins, not model
weights, datasets, CANN or donor binaries. It adapts Apache-2.0 upstream execution
paths; third-party notices and the license are included. Source included in the
package is public even while the development repository remains private.
