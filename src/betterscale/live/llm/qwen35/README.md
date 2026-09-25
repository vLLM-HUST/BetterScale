# Experimental Qwen35 live root

Implementation is owned here, not imported from LiveInference or `prototypes`:

- `root.py`: `QwenStateRoot`, composing model State domains and leaves.
- `state.py`: geometry/capacity, target GDN and FA, draft FA, continuation State.
- `gdn_graph.py`: `GDNGraphRoot`, the bounded two-bank numerical/lifecycle probe.
- `gdn_candidates.py`: isolated candidate kernel for that probe, requiring the
  pinned vLLM Triton environment. It is loaded only on execution, not root import.

Explicit Python entry (construction declares; activation allocates):

```python
from betterscale.live import LiveRuntime, TorchStateBackend, construct_live
from betterscale.live.llm.qwen35 import Capacity, Geometry, QwenStateRoot

runtime = LiveRuntime(
    device="cpu",
    state_backend=TorchStateBackend("cpu", memory_budget_bytes=512 << 20),
)
root = construct_live(runtime, lambda: QwenStateRoot(
    Geometry.from_config(text_config),
    Capacity(execution_seats=4, resident_seats=5, token_pages=32),
))
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

`python -m betterscale serve-qwen ... --runtime native` remains the default.
`--runtime live` is reserved and rejects before native resource preparation:
full target/FA/MTP execution, verification and resident scheduling are not yet
connected. There is no fallback and no claim of full-model acceptance. The
original LiveInference serving root is not copied wholesale: its old arena and
history ownership do not implement the accepted separate seat/page domains.

CPU contracts and the admitted GDN numerical probe live under
`prototypes/qwen35-state-lanes`; they now consume this installed package API.
