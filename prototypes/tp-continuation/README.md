# TP continuation and service acceptance (in progress)

This round extends main `7f7bda3`'s DP input producer / target packet / metadata
capture to TP8 while retaining split draft and ordered large-prefill replay.
The production entry remains the user-supplied native `vllm serve` command with
`--worker-cls strengthen_dsv4.worker.Worker`.

## Critical composition

Install `ordered_replay.install_capture()` **before** the target-bank wrapper.
The latter binds its fallback at installation, not import. Warmup then admits
the worker's ordered stream without replacing the bank wrapper. CPU QLI is
installed before the first real device metadata capture. No sibling patches
import each other; Worker chooses the order.

## Reuse the retained acceptance harness

The small files here overlay `prototypes/full-mixed/*.py` from experiment commit
`8f4eab3` (branch `lumi/decode-continuation`). Materialize that committed directory
in an artifact directory, then copy these Python files over it. The existing
`launch.py` holds the home-directory lease, checks selected devices and cleans
only its owned process tree. Keep its `probe_host_npus`, `supervise`, `idle_gate`
helpers available via `PROBE_HELPERS`, as in the retained capsules.

`donor_dp.py` adds the **already qualified four-seat TP envelope**, not sixteen
TP seats, to the packaged observer/oracle driver. `packaged_oracle.py` uses global
rank identities for TP receipts. It remains outside the delivered package.

`service_bench.py` accepts a complete native HTTP server command. Run it under
the same launcher. It measures three repeated synthetic cohorts: occupied
decode, balanced 4K prefill, and queued skew/turnover. SSE first output, chunk
arrivals, total output and native Prometheus counters are retained. Counts are
fixed but generated tokens and draft acceptance remain native. It is neither
an original agent-trace replay nor a proof that latency wins follow from a
matched-step win. Profiled/oracle runs are not timed service evidence.

`control_worker.ControlWorker` is an **experiment-only incremental ablation**:
TP equals kept main's FULL + split-draft + cross-step + ordered + CPU QLI; DP
keeps native-DSA FULL + cross-step without the new producer or target banks.
It is not stock donor. The delivered worker acquires no experiment selector.

Local artifacts: `/workspace/strengthen-dsv4/runs/tp-continuation-20260914/`.
Remote hw3 capsules: `/workspace/my-ascend-workspace/runs/tp-continuation-20260914/`.
No result is claimed until hardware acceptance is complete.
