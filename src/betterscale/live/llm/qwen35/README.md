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
`/v1/completions` and `/v1/chat/completions`. Only greedy, non-streaming text,
`n=1`, is supported; unsupported features fail before distributed execution.
Generation honors model EOS unless `ignore_eos` is explicitly requested.

Defaults are512 context tokens (including prompt, output and two-token MTP
lookahead),16 execution seats and20 resident seats. `--live-execution-seats`,
`--live-context-tokens`, `--live-resident-seats`, `--live-token-pages` and
`--live-distributed-port` configure these bounds. `--live-token-pages 0` (default)
uses observed-memory fitting with a1GiB free-memory floor. Calibration, State
rebinding and final capture use the existing LiveModule transaction, with
failure-aware CPU-group capacity agreement across TP ranks. Useful pages are
capped at R × ceil(context/128), not every remaining byte of HBM. A positive
page count is an explicit fixed-capacity override.

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

One execution owner groups compatible target/draft calls. Graph buckets are
1/2/4/8/16 for C16 (20 graphs); odd counts decompose into real smaller batches,
without dummy State rows. Prefill remains one token per request per protocol
step and interleaves with decode at completed-wave boundaries. The portfolio
shares serial scratch; its retained MetaTensor output banks and copied replies
are not disposable scratch. `/health` includes actual maximum active/batch
counts and current shared-page/queue/preemption counters.

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
BF16 batch-invariance guarantee. Attention deliberately uses a bounded plain implementation; no speed
claim follows from these probes. The existing published PyPI0.5.1 predates this
source addition; no new PyPI release is implied.

CPU contracts and admitted probes are under `prototypes/qwen35-state-lanes`;
they consume this packaged API. See `docs/evidence/qwen35-live-e2e.json` in the
source repository for the bounded receipts and failed-attempt history.
