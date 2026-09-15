# Qualify native prefix reuse with BetterScale FULL execution

Starting point: released0.4.0 / `80a75b5`, native vLLM `752a3a50`,
Ascend `9bf964cb`. Work is isolated from the published APC-off entry.

## Reuse the native mechanism

The pinned Ascend hybrid coordinator already handles C4/C128 logical-page hashes,
common hit boundaries, and propagates the EAGLE bit to same-spec sibling managers.
That propagation keeps SWA checkpoint retention consistent with read-side draft
lookahead. Do not copy these fixes into another patch or claim them as BetterScale
inventions. Physical block32 implies a4K common boundary, not the historical16K
from block128. Do not lower this boundary without a new partial-page protocol.

The shipped cross_step and DP producer cuts admit stable pure K5 only. A new
cache-hit request must first use native preparation; cached length alone must not
admit it as an already-established decode continuation. Once it becomes stable,
it may re-enter the existing fast path. Target FULL and TP draft metadata still
need an actual hit/resume gate; source plausibility is not qualification.

## First bounded gate

`runs/prefix-cache-20260915-tp/` (main checkout's ignored run directory) freezes
public code with only experimental APC/dummy admission relaxed. Dummy wo_a layout
repair is imported at the known-safe Worker entry, never sitecustomize. Native
hashing, cache managers and scheduler are unchanged. Existing leased eight-card
supervisor owns timeout, collision rejection and reclamation on hw3.

HTTP cases: cold12345 tokens; identical repeat; append256; divergence at token5000;
divergence at token100; original repeat after both branches. Each generates64
native K5 tokens. Request usage explicitly exposes cached_tokens; metrics and
outputs remain in the capsule. Require a real positive repeated hit, no hit past
a divergent token, and all completions. Do not infer reuse from lower TTFT alone.

This is a dummy execution/cache-routing gate, not output-quality or throughput
qualification. Follow with real retained retrieval cold/warm comparisons and DP
routing/ownership before changing public flags. DP caches are engine-local; a
request landing on another engine is not a cache-manager miss bug.

## CPU evidence

Pinned `test_compressed_prefix_cache.py`:3 passed on the installed local donor
runtime, exercising logical hash sensitivity, identical hits and rejection of
partial compressed-page matches. No NPU execution or approximate hash fixture.
Pinned `test_prefix_cache_cp_patches.py`:13 passed, including EAGLE write/read
checkpoint consistency for merged-spec siblings. Tests were copied byte-for-byte
into an isolated directory to avoid unrelated suite-wide NPU fixtures.
The actual device resume gate is recorded separately as it completes.

## TP8 dummy result193

All six cases passed with the existing FULL/K5 program. Cached-token counts were
0,12288,12288,4096,0,12288 respectively. Identical repeat output IDs matched.
The fork at5000 cannot reuse past4096; the fork at100 correctly misses entirely.
No new cache or execution patch was necessary in this gate. See `tp-dummy.json`;
timing remains dummy-only. All eight cards were released by the supervisor.

The next real-weight gate194 uses the same program and checks32 original retained
retrieval inputs, each immediately followed by its identical warm repeat. Score
both arms through the retained OpenCompass evaluator; preserve cached-token counts
and do not equate accepted draft differences with target-quality failures.

## TP8 real result194

Both cold and immediately repeated warm arms score32/32 through the pinned
OpenCompass LongBench retrieval evaluator. All32 warm requests have positive
cache hits and all32 cold/warm output token sequences match. The six synthetic
branch cases also pass with real weights. No additional runtime patch was needed.
See `tp-real.json`. This does not expand preemption or maximum-concurrency claims.

DP gate195 uses the native `X-data-parallel-rank` HTTP header. Simple branch cases
stay on engine0; each retrieval cold/warm pair is routed to request_id modulo8.
A separate16-request cohort (two per engine) repeats the same prompts concurrently.
This avoids confusing engine-local cache affinity with prefix-cache correctness.

## DP8 real result195 and release scope

Cold32/32 and warm32/32 retrieval pass, all32 warm hits positive and all output
sequences identical. The16-request two-seat-per-engine repeated cohort also passes
with4096 cached tokens each for8193-token prompts. See `dp-real.json`. The19 pinned
source files match both the local CPU runtime and the actual hw3 runtime.

Release0.4.1 removes only APC-off admission, retains native cache selection, and
adds no runtime hook.61 package CPU tests and16 native cache CPU tests pass.
The same native numerical program is retained; no NPU reload for release metadata.
