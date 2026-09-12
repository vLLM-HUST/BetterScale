# Extend DSV4 prefill/mixed FULL graph

Enter here when interpreting donor host gaps, deciding graph coverage, or
implementing a fixed-budget DSV4 prefill/mixed graph. Read
[the investigation and gap map](investigation.md) before changing graph modes.
That file records the initial source audit. For the subsequent bounded TP2
implementation and exact shadow checks, enter
[`prototypes/full-mixed/README.md`](../../../../../prototypes/full-mixed/README.md).
Do not confuse this target-only result with K5, TP8 or production acceptance.

Use the pinned submodules. The installed runtime that produced the September 12
trace is separate: three key Ascend files (DSACP attention, ACLGraph wrapper,
fused MoE) were byte-compared and matched the pins. No blanket whole-environment
identity is claimed.

Do not mistake a dense device task track for graph replay. Use native API links,
not only the collective-end-aligned export. The reproducible CPU-only gap tool
is at `evidence/inspect_launch_gaps.py` in the repository root; it consumes the
native provider DBs referenced in `evidence/donor-launch-gaps.json`.

The long wait_event is a one-off, not a general root cause. Empty compute/comm
coverage is not proof all engines are idle or a removable speedup estimate.
The native profile lacks event handles/producer links and CPU scheduling data;
do not infer a particular MoE event from temporal proximity.

Preserve existing decode FULL and the upstream shared-expert overlap. Start
with a fixed-shape model-forward probe before integrating scheduler dispatch.
Use probe-npu and lease/admission rules before any accelerator execution.

## Fixed-capacity lesson from the first passing probe

Native Compressor faults were traced to request-capacity mismatch: generic FIA
padding changed replay metadata dimensions despite capture retaining four rows.
Keep descriptor request capacity and repeat the final actual query offset through
inactive rows. Stable addresses alone are insufficient; tensor dimensions and
captured scalar bounds are part of the protocol. The CPU regression exercises
the exact padding function used by the probe.

Use shadow checks on valid output rows and the entire KV state. FlashComm1
outputs are TP-local; padded output rows have no semantic contract. The accepted
run012 has zero output/KV difference at rtol=.01, atol=1e-6. Coarse absolute .01
is unsuitable for this dummy model's small outputs. Shadow timings and memory
include copies/comparisons and must never be used as serving performance.

DSpark dummy fixtures need the draft's auxiliary target-layer IDs to follow the
shrunk target. Upstream dictionary hf_overrides deliberately do not propagate to
draft config; applying a callable to the target instead conflicts with Ascend's
quantization config requirement. Keep fixture repair separate from engine patches.
