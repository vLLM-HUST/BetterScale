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

## Current qualification boundary

- `125-tp8-packaged-shadow`: first four-request small-target comparison failed
  substantially (roughly90% of final hidden elements outside the strict gate).
  It is not accepted numerical drift; no performance claim or main promotion.
- `128-tp8-real-source-audit`: all54 GPU publication sources on rank0 matched
  their capture-time storage identity **and current bytes** before replay.
  Therefore changing GPU source addresses does **not** explain that first failure.
  Next isolate in-target recorded copies versus explicit pre-replay live
  publication (including CPU packet fields), using the existing exact oracle.
- Dummy audit did not reach the comparison: the pinned dummy loader leaves
  o-projection weight layout incompatible with native transpose-batchmatmul.
  Do not confuse that fixture failure with the real-weight continuation failure.
  `126` also exposed a test Worker constructor keyword mismatch, corrected in
  `127`. These are diagnostic attempts, not accepted target checks.

- `129` explicit live copies and `131` native metadata still failed the same
  strict TP oracle. Neither observation supports blaming recorded copies or
  metadata capture. The temporary live-copy production bridge is withdrawn.
- `132` adds the missing negative control: **native graph versus itself**, with
  the same KV/side-state restoration, also failed every checked TP step. This
  means the earlier oracle failures do not by themselves establish a candidate
  regression. Its diagnostic driver commits only the first native output/State.
- Recovered prior TP receipt:
  `/workspace/strengthen-dsv4-pingpong/runs/104-tp8-composed-shadow/engine/dp0-protocol.json`.
  That qualification explicitly used `HCCL_DETERMINISTIC=strict` (48 exact target
  comparisons/rank). The current runs used normal HCCL. Rechecking DP121/122 confirms those exact
  oracles ALSO used strict HCCL; DP124's normal-mode result was quality/performance,
  not exact replay equivalence. Do not conflate their envelopes. Next repeat
  the causal matrix in strict mode before deciding whether any state is missing.
  **Do not relax the comparison tolerance to paper over a failed native control.**

- `134` repeats the matrix with strict HCCL and the ORIGINAL recorded-copy
  package: packet96, producer88, metadata88 comparisons all have **zero output
  difference and byte-exact KV**; every native self-control also passes. The
  temporary copy/metadata suspicions are not supported. No runtime tolerance
  or production communication setting was changed to make the test pass.


## Startup preparation correction (September 14, in progress)

Fletcher rejected charging first-seen metadata/producer graph capture to live
requests. Runs133/135 are the retained cold-start HTTP control/candidate pair:
the candidate is not a stable E2E win (its early cohorts are substantially
slower). Runtime shape admission is a hypothesis for those stalls, not a proven
causal attribution. Do not erase them by adding benchmark warmup requests.

The next candidate constructs all finite producer banks and all metadata shapes
from actually captured small target descriptors, for both host carriers, before
Worker READY. Online cache misses now fall back without capture. Startup inputs
are restored, and no model forward/KV write is part of this auxiliary warmup.
Run137 checks the new TP startup route with strict HCCL/native-self controls.
The old split-draft first-use capture remains a separately identified startup
gap; do not claim the whole worker is prewarmed until that gap is handled too.


- `137`: startup exposed a missing inference-mode scope (native buffers include
  inference tensors). Fixed the startup scope, not their storage ownership.
- `138`: TP8 startup producer8/metadata8, strict-HCCL matrix packet96/producer88/
  metadata88 all exact against native, including every native-self control and
  complete KV backing. No serving-time auxiliary capture remains.
- `139`: DP8 starts with producer4/metadata6 on every rank; the online observer
  forbids entry growth. All nine HTTP cohorts finish and original32 retrieval
  questions score32/32. The first measured decode cohort17.03s (135) becomes8.50s, but the three-repeat
  service comparison still does not establish a stable all-cohort throughput win.
  Preserve the control133 and candidate135/139 data, rather than dropping the
  first repeat or attributing every difference to captured metadata.
- `140`: next TP candidate also prepares split-draft4+8 before READY and forbids
  any subsequent NPUGraph construction in the external service observer. That
  capsule was cancelled before warmup to correct an import. The corrected
  shared-draft-pool candidate141 then hit foreign, unlisted HBM occupancy during
  native weight loading. Neither capsule qualifies the new split-draft startup.


HTTP133–141 retain the SAME short128-input/32-output request prelude before the
nine measured cohorts. Do not call their first measured cohort the first HTTP
request, and do not infer complete graph warmup from that prelude. Entry-growth
or NPUGraph-construction guards apply immediately after Worker startup, including
the prelude. The harness now also records `first-http.json` for subsequent runs
without adding any requests or lengthening the warmup.
