# Qwen3.8-27B TP2 end-to-end serving investigation

This is a bounded native-service experiment, not the Qwen3 MoE owned-mode Worker.
The local27B checkpoint declares Qwen3_5ForConditionalGeneration,64 dense hybrid
layers (48 GDN /16 full attention), head256, and one MTP layer. Do not substitute
the512-expert Qwen3.8-Flash-Next checkpoint or the head128 Qwen3-30B-A3B prototype.

Existing same-host C1/2048 synchronous step evidence already establishes that
fixed FULL prefill alone did not reliably improve service: decode graph bodies
~34.6ms were unchanged; profiled gaps12–14ms remain a hypothesis, not pure CPU
idle time. The current task measures actual HTTP/SSE request latency and useful
throughput with bounded concurrency1/4/8, rather than rerunning that step baseline.

`run.sh` seals sources and model config, preserves donor/runtime, uses only an
admitted idle pair, and runs the normal vLLM frontend. `service_probe.py` warms
C1/C8, then measures two cohorts at each concurrency using fixed English prompt
IDs and64 output tokens. Report SSE chunks separately from token count. No
SLO is invented; compare latency/throughput frontiers, not a throughput ceiling.
APC is explicitly off for the first scheduling comparison. It needs a separate
reuse/pressure qualification later. MTP candidates must preserve output behavior
and justify verifier/draft overhead by accepted tokens, not draft throughput.

Optional profile: after all unprofiled cohorts, a separate request warms8steps
and records4steps using native torch-npu Db export. The observation Worker changes
no arithmetic or scheduler behavior. Initial/final profile drains are excluded
from steady-gap interpretation. Parse after worker exit and use TraceLoom native
AugDB and Perfetto exports; do not treat raw task durations as an additive wall
critical path. Profiles are never mixed into performance statistics.
