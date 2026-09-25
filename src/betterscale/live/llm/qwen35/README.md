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

The optional `--runtime live` route is under qualification on this work branch.
Its loopback HTTP ingress supports greedy, non-streaming text completions/chat;
request execution is serialized, with independent resident and token-page budgets.
It never constructs a native Worker or KV manager. Unsupported features fail
rather than falling back. Default native execution is unchanged.

The small0.8B TP1 four-graph/MTP/hot-seat vertical passed.35B-A3B TP2 currently
has an unresolved warm/cold token discrepancy; it is **not yet qualified**.
The experiment scripts and repo knowledge retain that failure and the active
independent-native/teacher-forced investigation. No speed or concurrent-serving
claim follows from these bounded correctness probes.

CPU contracts and the admitted GDN numerical probe live under
`prototypes/qwen35-state-lanes`; they now consume this installed package API.
