# Experimental Qwen35 live root

Implementation is owned here, not imported from LiveInference or `prototypes`:

- `root.py`: `QwenStateRoot`, composing model State domains and leaves.
- `state.py`: geometry/capacity, target GDN and FA, draft FA, continuation State.
- `execution.py` / `numerics.py`: real-weight synchronous target/draft execution.
- `graphs.py`: `QwenLiveLLMRoot`, four owned full-model graphs.
- `generation.py` / `residents.py`: greedy MTP commit, hot residents and shared pages.
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
allocation, initialization, four graph captures, invocation and retirement.
No external `livemodule`, native Worker, native KV planner or native runner
fallback is used. Default `--runtime native` remains unchanged.

Loopback HTTP on127.0.0.1:8000 exposes `/health`, `/v1/models`,
`/v1/completions` and `/v1/chat/completions`. Only greedy, non-streaming text,
`n=1`, is supported; unsupported features fail before distributed execution.
Generation honors model EOS unless `ignore_eos` is explicitly requested.

Defaults are512 context tokens (including prompt, output and two-token MTP
lookahead),20 resident seats,64 shared128-token pages, and one execution seat.
`--live-context-tokens`, `--live-resident-seats`, `--live-token-pages` and
`--live-distributed-port` configure those explicit bounds. GDN/continuation
lanes are resident-owned; target and draft FA use the shared token-page domain.
Finish retains hot State. Matching continuation resumes its seat; shorter
prefixes cannot use later recurrent State. Unrelated work prefers empty seats;
seat/page pressure evicts only idle residents. No CPU offload is provided.

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

This remains a serialized correctness entry, not a high-throughput scheduler,
C16/C32 qualification, maximum-context claim or universal BF16 batch-invariance
guarantee. Attention deliberately uses a bounded plain implementation; no speed
claim follows from these probes. The existing published PyPI0.5.1 predates this
source addition; no new PyPI release is implied.

CPU contracts and admitted probes are under `prototypes/qwen35-state-lanes`;
they consume this packaged API. See `docs/evidence/qwen35-live-e2e.json` in the
source repository for the bounded receipts and failed-attempt history.
