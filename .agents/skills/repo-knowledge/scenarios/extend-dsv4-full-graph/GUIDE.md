# Extend DSV4 prefill/mixed FULL graph

Enter here when interpreting donor host gaps, deciding graph coverage, or
implementing a fixed-budget DSV4 prefill/mixed graph. Read
[the investigation and gap map](investigation.md) before changing graph modes.
This records a source audit, not a passing FULL-prefill implementation.

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
