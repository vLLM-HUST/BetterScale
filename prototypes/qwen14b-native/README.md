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
initial device2 watcher was cancelled before launching. A subsequent local
device4 load collided with a foreign process; only our process group was stopped,
and no measurements from that attempt were accepted. The completed run uses
hw3 physical device0, with the complete checkpoint copied to the task-local
`/workspace/my-ascend-workspace/runs/qwen14b-native-20260915/model` directory.
The frozen executed driver differs only in its model path; the reusable driver
now accepts `--model`. Other cards' tasks are preserved.
No NPU lease is needed after execution while parsing/exporting.

Run `analyze.py FIRST/measurements` with the same runtime/CANN environment after
successful completion. It uses official torch-npu offline DB parsing and frozen
TraceLoom37323af. Single-rank data needs no collective clock fitting. Preserve
the actual rank/device mapping; publish only completed `.json.gz` exports.
When retrieving raw profiles from another host, use `rsync -a --no-owner
--no-group`: CANN rejects foreign-owned input even when the parser runs as root.
Its Python parser can exit zero on this error, so require the resulting DB and
check it rather than trusting the subprocess return code alone.

## Completed native baseline (2026-09-15)

hw3 / physical device0 / 910B2, real BF16 weights. Execution source pin783b8a7
with only the model path substituted in the frozen driver. Native resolved graph
policy: `FULL_AND_PIECEWISE`, capture sizes `[1,2,4,8]`. Both cohorts PASS;
owned worker reclaimed and card returned to idle. See `baseline-summary.json`.

| Cohort | Unprofiled whole-cohort seconds (two repeats) | Late replay compute span, median | Inter-compute gap, median |
| --- | --- | --- | --- |
| 1 ×4096 input,32 output each | 1.3600 /1.3656 | 26.929ms | 1.552ms |
| 8 ×512 input,32 output each | 1.4645 /1.4815 | 27.574ms | 1.595ms |

Last two columns come from the separate8-output-token profiles, not TPOT or
unprofiled measurements. Span is first-to-last compute task per late graph burst;
gap excludes those spans and can include non-compute work, not necessarily idle
hardware. The first replay is excluded from this late-window summary.

Each profile has seven `aclmdlRIExecuteAsync` calls. Actual task/model IDs show
prefill outside the captured model, decode inside it. The eight-request cohort
arrives through native offline enqueue/scheduling: its large forwards have512,
1025 and2563 rows (including mixed decode rows), not a single4096-row forward.
Do not claim fixed wave identity between cohorts or treat every issued request
as admitted simultaneously. Matrix ops dominate the seven decode replays:
158.04ms total atC1 and159.18ms atC8; attention totals17.77/16.32ms respectively.
This suggests looking at native prefill graph coverage before revisiting a
complex continuous-decode protocol. It does not establish an optimization gain.

Local full evidence root:
`/workspace/strengthen-dsv4/runs/qwen14b-native-20260915/hw3-single0/`.
The two `measurements/{c1-4k,c8-512}/analysis/*.json.gz` files are completed
TraceLoom37323af exports (single rank, no inter-rank clock alignment required).
The native profiler databases remain alongside them. These are synthetic token
performance probes, not quality or online QoS acceptance.

For the subsequent BF16 streaming-fusion source investigation and configuration-only
FULL probe, read [FFN-AND-GRAPH.md](FFN-AND-GRAPH.md). Single-card follow-ups now
run locally, not on hw3.
