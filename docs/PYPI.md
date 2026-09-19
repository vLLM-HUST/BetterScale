# BetterScale

**Less host waiting. More device execution.**

BetterScale is a modular execution-optimization package for **vLLM Ascend**.
It adds supported FULL graph paths, ordered replay, and device-side progress and
metadata preparation for DeepSeek V4 Flash and dynamic mixed FULL graphs for Qwen27. One native Worker entry composes the
patches; it does not replace the serving engine or configure your environment.

[Measured results and mechanisms](https://vllm-hust.sage.org.ai/betterscale.html)

## Install

Use your **existing, working Ascend serving environment**, with Python 3.12+:

```bash
python -m pip install --no-deps vllm-betterscale==0.5.0
```

The package deliberately does not install or upgrade vLLM, vLLM-Ascend, torch-npu,
the CANN runtime, model weights, or the donor device operators. Install those through your normal
Ascend deployment. The Linux/aarch64 package includes a qualified HC-pre host-tiling
library for TP, selected privately without overwriting your donor installation.
Native CANN-licensed components remain solely for Ascend processors. PyPI ships
an sdist with that prebuilt library and the three qualified Qwen native libraries: pip builds the small Python wrapper locally,
without compiling operators. Install on Linux/aarch64 in the qualified environment.

This release is pinned to **vLLM 0.25.1**, **vLLM-Ascend 0.25.1rc1**, and
**torch-npu 2.10.0.post2**. The measured environment uses CANN 9.0.1 and eight
Ascend 910B2 cards connected with HCCS. Worker initialization verifies both versions
and selected upstream source files; a different build may be rejected even if its
version string matches. Do not disable those checks to force an unqualified runtime.

## Start Qwen3.8-27B · TP2

In the qualified Linux/aarch64 CANN 9.0.1 environment, choose two idle 910B2 cards
and replace the local model path:

```bash
python -m betterscale serve-qwen /models/Qwen3.8-27B --devices 0,1 --port 8000
```

No Git checkout, native compilation, library-path exports or second Worker are
needed. The package supplies GDN H/O, its graph-pool host adapter and the FIA
planner. The launcher sets the pre-startup preload, queue mode and AIV, then
executes the existing `betterscale.worker.Worker` service. Explicit
`BETTERSCALE_*_LIBRARY` overrides remain available but must match the packaged
artifact identities. It never rewrites installed donor files.

Configuration: BF16, TP2/DP1, no MTP, eight seats, 2048-token budget, 8192 context,
6 GiB KV per rank, FULL capacities 1/2/4/8/16/32/64/128/256/512/1024/1536/2048,
align-mode prefix caching, asynchronous scheduling, text-only input. Qwen pins
include vLLM source752a3a50 and Ascend9bf964cb; the package validates selected
source files as well as versions. Model weights and this pre-existing runtime
are still prerequisites; `pip install` does not provision them.

In another terminal:

```bash
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen27","prompt":"Hello","max_tokens":32,"temperature":0}'
```

The loopback endpoint is intentional. `--cache-dir` selects a dedicated compiler
cache (default `~/.cache/betterscale/qwen27`). The launcher does not reserve cards;
use your host's admission protocol. Restart with a fresh native service/state pool
to revert; never hot-unpatch GDN state. Historical Qwen measurements are retained
as source-result evidence, not a fresh performance claim for this packaging change.

## Start DSV4

Choose one complete command below. These are two bounded, single-node DSV4 configurations—not arbitrary models,
shapes or parallel layouts. The examples below use local model weights at
`/models/DeepSeek-V4-Flash`; replace that path with your Ascend W8A8 checkpoint.
CANN/HCCL and device visibility remain your existing environment's responsibility.
The loopback binding is deliberate; use your normal authentication, TLS and access
controls before exposing the service outside the host.

### TP8 + EP, DSpark K5

Four active requests; token budget 4128; context up to 524288. DSACP is enabled,
DCP/PCP remain 1, and native prefix caching is supported.

```bash
vllm serve /models/DeepSeek-V4-Flash \
  --worker-cls betterscale.worker.Worker \
  --tensor-parallel-size 8 --enable-expert-parallel \
  --quantization ascend --dtype bfloat16 --async-scheduling \
  --max-num-seqs 4 --max-num-batched-tokens 4128 --max-model-len 524288 \
  --enable-prefix-caching \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5,"enforce_eager":true}' \
  --compilation-config '{"cudagraph_mode":"FULL","cudagraph_capture_sizes":[24,4128],"max_cudagraph_capture_size":4128}' \
  --additional-config '{"enable_dsa_cp":true,"multistream_overlap_shared_expert":true,"ascend_compilation_config":{"enable_npugraph_ex":true,"enable_static_kernel":false}}' \
  --host 127.0.0.1 --port 8000 --served-model-name dsv4
```

### TP1 / DP8 / EP8, DSpark K5

Two active requests per rank (16 total); local token budget 1026; context up to
524288. DSACP is disabled; target is FULL and draft remains native eager.

```bash
vllm serve /models/DeepSeek-V4-Flash \
  --worker-cls betterscale.worker.Worker \
  --tensor-parallel-size 1 --data-parallel-size 8 --enable-expert-parallel \
  --quantization ascend --dtype bfloat16 --async-scheduling \
  --max-num-seqs 2 --max-num-batched-tokens 1026 --max-model-len 524288 \
  --enable-prefix-caching \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5,"enforce_eager":true}' \
  --compilation-config '{"cudagraph_mode":"FULL","cudagraph_capture_sizes":[6,12,132,264,516,1026],"max_cudagraph_capture_size":1026}' \
  --additional-config '{"enable_dsa_cp":false,"multistream_overlap_shared_expert":true,"ascend_compilation_config":{"enable_npugraph_ex":true,"enable_static_kernel":false}}' \
  --host 127.0.0.1 --port 8000 --served-model-name dsv4
```

The examples use automatic physical KV sizing. An explicitly supplied manual
byte budget remains available; see the capacity boundary below.
In 0.3.1, DP prepares its finite producer/metadata graph catalog before READY:
4 producer banks and 6 metadata entries per rank for the two-seat K5 configuration.
Unknown runtime metadata shapes fall back to native preparation instead of capturing
online. This changes startup preparation, not the model or KV contents. TP also prepares its bounded draft graph catalog before READY; runtime capture
is not part of the serving path.

## Verify and roll back

Check for the per-worker `BetterScale rank=... READY patches=...` log entry and
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
paths; third-party notices and the license are included. Source is available in the package and the
[public development repository](https://github.com/vLLM-HUST/BetterScale).
## DSV4 physical KV sizing and context capacity

Without `--kv-cache-memory-bytes`, BetterScale measures the resident target/draft
program in a shared graph pool and budgets actual free memory minus model,
non-graph/profile costs and **1 GiB/rank safety headroom**. It does not use the
native 90% fraction as its automatic ceiling. An explicit manual byte budget
still selects the native manual path. Do not run uncoordinated services on the
same cards: this is startup sizing, not dynamic memory arbitration.

Both supported layouts admit contexts up to **524,288 tokens**, including output.
This is a per-request ceiling, not a promise that every active seat can hold a
full-length context simultaneously. Native KV admission still queues requests.

- TP8: real-weight automatic KV **about 14.94 GiB/rank**, with about **0.97 GiB/rank**
  free after startup. The new startup composition passes all 32 retained retrieval
  questions. Dummy capacity gates complete a 448Ki-token input and reach **96.84%**
  KV occupancy under long-request pressure; at most three long requests run together.
- DP8: the retained real-weight physical-sizing gate assigns **about 7.9 GiB/rank**;
  eight 60K-input requests and a separate single 448Ki-input request complete.
  Long-history saturation and preemption recovery are not established by those runs.

Do not compare native “KV token capacity” across different context ceilings as
if it were a fixed-size token heap. Hybrid SWA/compressed-state accounting depends
on the horizon and prefill wave budget. APC is supported with native hybrid checkpoint alignment (4K at default block32).
DP caches remain engine-local; use session affinity or the native
`X-data-parallel-rank` routing header to return to the same engine.
The 3GiB diagnostic preemption failure is not claimed fixed.

## DSV4 prefix reuse qualification

TP8 and DP8 each pass cold32/32 and warm32/32 original retained retrieval
questions; all warm requests hit and cold/warm token outputs match. DP also
passes a16-request repeated cohort, two requests per engine. This enables the
existing native cache mechanism; no new cache algorithm or execution hook is added.
The native option can still be disabled. These are bounded reuse/quality checks,
not whole-suite certification or a preemption-recovery fix.

## 0.4.2 memory improvements

TP now clears each KV backing once and removes HC-pre’s fixed workspace floor.
On the qualified TP8 dummy FULL configuration, automatic KV budget increased by
about 186 MiB/rank (1.21%) and READY reserved memory fell 360 MiB/rank, with the
same 1 GiB safety reserve. These are capacity measurements, not new throughput
or real-weight quality results. DP native tiling is unchanged.
