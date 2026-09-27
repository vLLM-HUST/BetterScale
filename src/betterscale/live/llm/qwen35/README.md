> Historical standalone execution research. The current35B `serve-qwen --runtime live`
> command uses the native BetterScale model/MTP/async route with resident State;
> see [`models/qwen35`](../../../models/qwen35/README.md). This module remains
> available for explicit research and the small-model vertical, not the qualified
>35B performance entry. The observations below retain their original scope.

# Experimental Qwen35 live root

Implementation is owned here, not imported from LiveInference or `prototypes`:

- `root.py`: `QwenStateRoot`, composing model State domains and leaves.
- `state.py`: geometry/capacity, target GDN and FA, draft FA, continuation State.
- `execution.py` / `numerics.py`: real-weight synchronous target/draft execution.
- `graphs.py`: `QwenLiveLLMRoot`, four model-call families in real batch buckets.
- `generation.py` / `residents.py`: greedy MTP commit, hot residents and shared pages.
- `scheduler.py` / `ingress.py`: completed-wave batching, preemption and async ingress.
- `gdn_graph.py`: `GDNGraphRoot`, the bounded two-bank numerical/lifecycle probe.
- `gdn_candidates.py`: isolated candidate kernel for that probe, requiring the
  pinned vLLM Triton environment. It is loaded only on execution, not root import.

Explicit Python entry (construction declares; activation allocates):

```python
from betterscale.live import LiveRuntime, TorchStateBackend, live_runtime
from betterscale.live.llm.qwen35 import Capacity, Geometry, QwenStateRoot

runtime = LiveRuntime(
    device="cpu",
    state_backend=TorchStateBackend("cpu", memory_budget_bytes=512 << 20),
)
with live_runtime(runtime):
    root = QwenStateRoot(
        Geometry.from_config(text_config),
        Capacity(execution_seats=4, resident_seats=5, token_pages=32),
    )
root.activate()
try:
    # Resolved model State is available; this is not a model-forward API.
    recurrent = root.target["0"].recurrent.tensor
finally:
    root.close()
```

The base root does **not** seed numerical State with probe data, capture graphs,
load weights, schedule requests or generate tokens. GDNGraphRoot deliberately
uses test seeds and fixed 0.8B geometry; do not use it as a full LLM root.
Inherited LiveModule lifecycle owns allocation, initialization, capture and
retirement rather than a second runner owning the same tensors.

## Optional live serving entry

From this development source build, in the pinned donor/CANN environment:

```bash
python -m betterscale serve-qwen /models/Qwen3.5-35B-A3B \
  --runtime live --devices 0,1
```

The root is `QwenLiveLLMRoot` in `graphs.py`, with real target/draft execution
in `execution.py` and numerical leaves in `numerics.py`. BetterScale owns State
allocation, initialization, graph capture, invocation and retirement.
No external `livemodule`, native Worker, native KV planner or native runner
fallback is used. Default `--runtime native` remains unchanged.

Loopback HTTP on127.0.0.1:8000 exposes `/health`, `/v1/models`,
`/v1/completions` and `/v1/chat/completions`. Only greedy text with
`n=1` is supported; completions additionally supports exact-token SSE streaming; unsupported features fail before distributed execution.
Generation honors model EOS unless `ignore_eos` is explicitly requested.

Defaults are262144 context tokens (prompt plus committed output),16 execution
seats and20 resident seats. MTP tail steps do not reserve unusable lookahead
positions. `--live-execution-seats`, `--live-context-tokens`,
`--live-resident-seats`, `--live-token-pages` and `--live-distributed-port`
configure these bounds. A positive page count is a fixed-capacity override.

With `--live-token-pages 0`, graph calibration precedes fixed resident-State
allocation. Real remaining free memory, less a1GiB floor and explicit allocator
rounding allowance, becomes the shared page budget. Inactive allocator cache
is not credited. Ranks agree on the smaller page count before allocating KV
once; there is no repeated allocation search or seat-times-context cap.
Rebinding and final capture remain in the existing LiveModule transaction.

GDN/continuation lanes are resident-owned; target and draft FA use one shared
128-token page domain. Pages grow on demand, **not** by reserving an entire
request's maximum output. Finish retains hot State. Matching continuation
resumes its seat; shorter prefixes cannot use later recurrent State. Unrelated
work prefers empty seats; page reclamation first uses idle residents.

Under active page pressure, the scheduler waits for the current wave to drain,
preempts the newest request, invalidates its entire seat and returns all its
pages. CPU prompt plus committed output IDs are requeued for recomputation;
GDN candidates/conv, target/draft KV and continuation never survive separately.
The surviving cohort drains before re-admitting victims to avoid immediate
thrashing. Cancellation also invalidates the whole seat. No CPU offload or
intermediate recurrent checkpoint is provided.

One execution owner groups compatible target/draft calls. Decode graph buckets
are1/2/4/8/16 for C16; odd counts decompose into real smaller batches, without
dummy State rows. Prefill has independent single-request4/16/64/256/1024-token
buckets, interleaved at completed-wave boundaries rather than multiplied by C.
The first token after a hot boundary normalizes accepted GDN/conv history before
chunked prefill; draft chunks consume shifted target hidden states.

Paged FIA and GDN chunk kernels are packaged numerical leaves; no native runner
patch is installed. `/health` exposes capacity arithmetic and actual scheduler
activity. SSE reports only committed output, including after recomputation;
`cache_salt` isolates hot resident identity across independent session plays.

## Qualified scope

The real0.8B TP1 four-graph/MTP vertical passed, followed by35B-A3B BF16 TP2.
The latter's installed public entry passed at512 context /20 residents /64
pages: arithmetic and exact-copy chats including EOS,12 raw outputs and8 warm
continuation outputs match an independent pinned native baseline token-for-token.
A189-token chat crosses token-page boundaries and repeats identically.
Separate TP2 checks prove untouched hot GDN/FA State,19-token prefix reuse,
cold-target equality, graph retirement and fresh-generation replay.

The MoE loader explicitly preserves native Ascend's FP32 router-weight contract
for both target and draft. Omitting it silently selects BF16 routing and caused
the now-resolved token discrepancy; it is not a cache-layout workaround.

The preceding receipts describe the historical E1 entry. The subsequent
35B TP2 C16/R20 model-root check captured20 graphs, reached actual batch16,
and matched all16 eight-token outputs against single-request target-only
controls. Both ranks fitted40 shared pages at256-token context; an unrelated
request used seat16, and the retained15-token prefix resumed seat0 correctly.
The0.8B TP1 constrained four-page check forced two whole-seat preemptions,
including one with committed output; recomputation matched all controls and
cancellation cleared GDN/continuation. The installed HTTP TP2 entry also reached
actual batch16 at512-token context with only16 shared pages:16×12 raw outputs
match native;16×9 long-chat outputs match serial controls, despite eight
whole-seat preemptions. A real client disconnect cancelled and reclaimed its
seat; service exit0 and both cards released. Receipts and the final CPU-only
disconnect-error presentation fix are in `docs/evidence/qwen35-live-scheduler.json`.

This is not a maximum-context claim, C32 execution qualification, or a universal
BF16 batch-invariance guarantee. These historical receipts used bounded plain attention; they do not qualify the
new paged/chunked long-context implementation or its speed. The existing published PyPI0.5.1 predates this
source addition; no new PyPI release is implied.

CPU contracts and admitted probes are under `prototypes/qwen35-state-lanes`;
they consume this packaged API. See `docs/evidence/qwen35-live-e2e.json` in the
source repository for the bounded receipts and failed-attempt history.
