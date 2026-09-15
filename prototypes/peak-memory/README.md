# BetterScale allocation staircase

Observation only against release0.4.0/ec01754. No public defaults or native operator
changed. Reuses LiveInfer's counter-only tracer from workspace
`prototypes/dsv4-replicated-prefill/capture_memory_trace.py` and its rising-watermark
method (`CAPTURE_WATERMARKS.md`). The adapter selects BetterScale/vLLM/Ascend Python
frames and adds reserved/peak-reserved and native peak-reset observations.

Never inspect `f_locals`, retain tensor objects or insert per-line synchronization.
Trace timestamps are not performance evidence. Compare the same-host untraced
control before interpreting attribution. Python boundaries expose native transient
high water, not necessarily the precise internal allocator call.

The worker keeps the published automatic memory policy and one target/draft pool.
It records profile/trial, final capture/draft and the first eight model executions;
automatically filled KV is itemized separately. Existing native phase resets stay
unchanged. `summarize.py` takes a run directory, retains every increasing watermark
and gives later sampled-live shoulders. Reserved admission footprint matters even
when allocated peak falls; do not sum all operator peaks or multiply one-layer
scratch by layer count. This is not a tensor-by-tensor live allocation reconstruction.

The task-local launch capsule under `runs/peak-memory-20260915` freezes published
source with dummy-only admission and the already-qualified dummy WO_A layout fix.
Both are fixture-only. It reuses run192's home-lease/admission/descendant supervisor,
native TP8+EP K5, four seats, buckets24/4128, context524288, then a short HTTP prelude,
single8K prefill and four4K requests. No real checkpoint load or new quality claim.
CPU test covers counter changes, native peak resets and restoration of tracing.

## September15 result and a narrowed next move

`evidence.json` owns the compact control, all8 peak-equality checks and every rank0
high-water event. Full raw traces stay in the local capsule. The control passes
all HTTP requests and exits0. Detailed trace4 completes both target buckets, then
exceeds the HTTP harness's900s readiness deadline in draft preparation; exit1 and
all owned workers released. Do NOT relabel that partial trace as a full serving pass.
Trace3 was deliberately stopped: tracing eager-profile/compiler internals burned
CPU time. The narrowed trace omits eager-profile tracing but still incurs hundreds
of seconds per full-model capture; reuse these records, do not repeat that tax.
For further work, isolate the suspected operator/lifetime in a small fixture.

Per-rank observations (rank0; all8 target peaks are byte-identical to control):

- Model41.0325GiB; native eager activation337.45MiB; non-Torch2.4722GiB.
- Budgeted target786.883MiB + draft45.656MiB =832.539MiB. These are measured
  physical graph costs, not sums of individual tensors.
- Trial capture baseline45724.498MiB → allocated peak46074.079MiB:
  **349.581MiB additional peak**, atop trial State and weights.
- Early watermarks appear at rotary construction and before entering DSA's
  custom boundary. A new peak observed on a function's first line does NOT prove
  that function allocated it; compiled producer code may precede that sample.
- MoE fused gate/up/quant and GMM2 are intermediate shoulders. Highest watermarks
  follow `DeepseekV4DecoderLayer.hc_pre`'s native `npu_hc_pre_v2` call.
  Python-only records cannot split that native operator's internal temporaries.
- Target reserved peak46604MiB; allocated peak46074.079MiB. Lowering a transient
  alone does not guarantee a smaller reusable graph pool.
- Final READY reserved58256MiB versus allocated57203.646MiB leaves1052.354MiB
  allocator-held slack. Not all slack is safely reclaimable. After the last draft
  graph, clearing129 final-State views increases the running allocated high water
  from57524.352 to57746.523MiB; the latter is542.877MiB above READY live bytes.
  **222.171MiB is the high-water increase, not543MiB.** This is a phase-local
  observation, not proof of `zero_`'s exact temporary or a removable allocation.

Priority for another experiment: isolate the post-preparation strided-State clear
(avoid redoing an entire model), then check HC-pre/residual-clone alias contracts.
The source does clone the MHC residual twice per layer, but graph reuse prevents
multiplying that clone by43 layers. Never remove a clone or State clear without
validating its producer/consumer mutation and graph reuse contracts. MoE is already
using fused upstream kernels: no evidence justifies replacing it wholesale.
