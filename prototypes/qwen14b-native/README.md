# Qwen2.5-14B single-card native baseline

Local BF16 checkpoint: `/data/shared_models/Qwen2.5-14B-Instruct`,8 shards,
27.51GiB,48 layers,40Q/8KV heads. Native pinned vLLM0.25.1/Ascend0.25.1rc1.
No BetterScale Worker, no model/attention patch, no quantization or speculation.
`observe.Observer` only starts/stops the native profiler through extension RPCs;
it does not wrap forward, scheduling, sampling or graph dispatch.

Native graph policy is not overridden. Record resolved mode and buckets instead
of calling the run FULL in advance. Explicit workload envelope: one card,8 seats,
4096 wave tokens,8192 context,APC off,async scheduling on,seed123. Native default
memory sizing remains in force. These are offline fixed-cohort engine calls,
not an HTTP serving/TTFT benchmark or a model quality gate.

Two windows: one4096-token prompt and eight512-token prompts. Each warms once,
then has two unprofiled32-output-token repeats. A separate8-output-token run
captures prefill and decode; profiler overhead does not enter the unprofiled
cohort measurements. Inputs are synthetic token IDs, real BF16 weights.

`runs/qwen14b-native-20260915/` in the main checkout retains the frozen source,
known local runtime launch and home-lease admission/descendant supervisor. The
initial request waits only for physical device2; other cards' tasks are preserved.
No NPU lease is needed after execution while parsing/exporting.

Run `analyze.py FIRST/measurements` with the same runtime/CANN environment after
successful completion. It uses official torch-npu offline DB parsing and frozen
TraceLoom37323af. Single-rank data needs no collective clock fitting. Preserve
the actual rank/device mapping; publish only completed `.json.gz` exports.
