# Owned AscendC GDN probe

Experimental H/O fork for Ascend910B2, BF16 inputs / FP32 gates and states,
qk8/v24 heads, K/V128. The qualified K-V variant is wired into the opt-in
MixedWorker service; see the final section. Historical probes below retain their
original scope. No MTP/PCP or internal empty-row support claimed.

`h/`, `o/`, `common/` copied from vllm-ascend commit
9bf964cb4b87c8cd0d6852c41a55b3c29711fa95, respectively
`csrc/moe/chunk_gated_delta_rule_fwd_h/op_kernel/arch22`,
`csrc/moe/chunk_fwd_o/op_kernel/arch22`, `csrc/moe/common`.
Original per-file copyright/license notices remain. Catlass is an external,
unmodified dependency pinned to41bf90da655bba3c66d0acd7e00abe33960ecfd6
(the donor gitlink); do not substitute a nearby checkout's current revision.

The two small kernel entries select only the tested dtype specialization.
Owned POD tiling replaces GE-generated structs. The historical non-pool raw
probe owns private scratch/output tensors. Current MixedWorker keeps only the
immutable device PODs and uses the framework host adapter for scoped scratch;
see `src/betterscale/patches/qwen_gdn/README.md`.
The raw generated ACL launch functions consume stable device metadata directly.
No native op-schema override, GE execution, or host-list conversion is needed.

Build on hw3, source CANN env, then in a NEW build directory:

```
cmake -S <source> -B <fresh-build> -DCATLASS_ROOT=<pinned-catlass>
cmake --build <fresh-build> -j2
```

CANN legacy build requires explicit Release and toolkit include. Changing build
configuration in-place left duplicate host objects; use a fresh directory.
Compilation is CPU-only; execution must use the normal selected-card admission.
Set TASK_QUEUE_ENABLE=0 for this ctypes prototype: it does not integrate with
Torch-NPU's asynchronous host submission queue. Pass ASCENDC_GDN_LIB as the
absolute built lib/libbs_gdn.so path. Graph captures must not outlive Kernels.
Raw calls do not provide dispatcher/autograd integration; this is intentional.

## Hardware acceptance (hw3, September17)

Capsules under `/workspace/my-ascend-workspace/runs/qwen27-partition-serving`:

- `ascendc-gdn-build1/release-build`: unmodified donated arithmetic/scheduler,
  raw owned ABI. `ascendc-gdn-fixed1` passes output/state/H/Vnew comparisons,
  all max_abs0 after capture and replay. Initial build attempts exposed a missing
  toolkit include and mixed empty/Release build objects, not a kernel error.
- `ascendc-gdn-build2`, source2d6cb4b: immutable token/H-chunk strides.
  `ascendc-gdn-dynamic1` reuses one512token/4request(+sentinel) capture across seven
  partitions, shorter totals, changed slots and continuing states. All28 output,
  whole-bank and second-pass checks are exactly0, versus native active-shape GDN.
- `ascendc-gdn-build3`, source9e49edc: BS_GDN_OWNED_INIT=ON.
  `ascendc-gdn-owned-init1` passes the same28 checks exactly. Initialization is
  assigned to the consuming AIC/AIV pair; both AIV subblocks retain duplicate
  within-pair copies and their original readiness signals. The CPU ownership
  check covers the donor head-task mapping for1–64requests; hardware capacity
  qualification remains4requests, not64.

Full-pipeline dynamic replay with owned initialization: .871–.934ms; same-run
fixed native control .587–.773ms. Dynamic still includes capacity work, state
management and different graph glue. This is NOT an end-to-end speedup and must
not be compared causally with earlier TASK_QUEUE_ENABLE=1 Triton measurements.

`stage_probe.py` removes transposes/state glue and compares matched head-major
H and O graph stages using native/owned/owned/native twice,30replays/measurement.
`ascendc-gdn-stages1` PASS, all outputs/intermediates exact. Means in milliseconds:

| lengths | native H | owned H | native O | owned O |
| --- | ---: | ---: | ---: | ---: |
|512|.10869|.07768|.07359|.06158|
|1,511|.13442|.08614|.07699|.06107|
|1,1,256,254|.16039|.06546|.08289|.06038|

These are graph-stage costs (native adapter tasks versus raw owned launch), not
an isolated attribution of all H savings to initialization. O's matrix algorithm
is unchanged. No model/HTTP, msprof/TraceLoom or service integration in this probe.

Follow-up `ascendc-gdn-stages-control1` uses build2 (owned initialization OFF)
with the same stage harness and also passes exact checks. Owned H means
.09597/.12244/.13155ms; its native H .11269/.13897/.14808ms. Thus the raw-boundary
control alone saves roughly .0165ms here; the ON result supports additional
initialization savings, especially four requests. OFF and ON are separate admitted
runs (their native controls vary), not a single interleaved ON/OFF confidence study.
Do not present the cross-run difference as a precise guaranteed gain.

## Direct K-V state ownership — current prototype

Fletcher chose a compute-friendly shared pool layout over preserving the donor's
V-K storage convention. H now reads/writes FP32 `[slot,24,K,V]` directly, using a
device `[request,2]` INT64 table `(slot,has_initial_state)` in its otherwise unused
chunk-index argument. Initial and final pointers name the SAME bank. Cold slots
start from zero in UB; no gather/where/scatter or separate final-state allocation.
Positive-length rows must form a packed prefix, slots must be unique/in range;
empty suffix rows perform no state access. State-pool mode REQUIRES the
BS_GDN_OWNED_INIT=ON build: unrelated pairs must not reread/write another pair's
mutable initial state. Use the qualified build4 artifact, not older H/O libraries.
The ctypes prototype has no automatic binary ABI/version negotiation yet.

`pool_forward` is the owned entry; the old `__call__` is the non-pool control.
`pool_probe.py` publishes all dynamic metadata in one pinned536-byte H2D slab;
that publication is once per wave, not per layer. Source asserts capacity and
exclusive-slot ownership before publication. Its single pinned buffer is reused
only after synchronization; asynchronous multi-wave ownership is not implemented.

`decode_kv.py` is a limited nonSpec one-token decode adaptation of pinned vLLM
752a3a50 `model_executor/layers/fla/ops/fused_recurrent.py` (original notices kept).
It uses K-V address/tensor orientation, FP32 state, one program per request/head,
and two64-wide V tiles within that program. No global state transpose. Explicit
FMA creates tiny FP32 differences versus native. Every nonempty cu segment MUST
have one token; noMTP/KDA/vector-beta/general-head-shape support claimed. Native
AscendC recurrent remains the comparison, not this prototype's implementation.

Impact audit: donor gdn.py owns prefill and recurrent consumers; its current
prefill path gathers and transposes V-K state, then reverses that on writeback.
Qwen's state-shape calculator specifies V,K (both128 in this checkpoint).
`get_temporal_copy_spec` copies entire state rows without interpreting K/V;
convolution cache is separate. Thus a new owned serving path can allocate K-V
without changing opaque temporal copies or convolution, but must route ALL its
readers to K-V-aware kernels. Never pass this bank to native recurrent or silently
reinterpret an existing live V-K pool. MTP/PCP/APC and external transfer protocols
are not qualified by this audit. No production allocation/Worker was changed.

### Acceptance and complete core-GDN timing

hw3 `ascendc-gdn-pool4`, source7a01f9c, build4/f6f1bed:
- Eight partitions, including mixed cold/continuing requests. Cold slots seeded
  with NaN, proving cold startup ignores stale pool contents.32 output/bank/second
  pass comparisons against native chunk are max_abs0.
- Actual native role control uses recurrent decode prefix plus chunk prefill tail,
  native gather/transpose/clear/writeback and output concatenation.16 additional
  checks pass; mixed chunk-vs-recurrent differences peak .000244 output/.003468
  state, within atol=.01/rtol=.01. Pure-decode policy adds2 passing checks.
- Prefill/mixed reuse ONE dynamic graph; pure decode has its own graph, NOT the
  entire chunk pipeline. Both consume the same K-V pool contract. No partition
  enumeration. NaN cold slots, inactive rows and changed slot ownership included.
- Same-process ABBA twice,20replays/measurement, complete core-GDN graph means(ms):

| scheduled lengths | roles | native | owned |
| --- | --- | ---: | ---: |
|512|one prefill|.76077|.69236|
|1,511|two cold prefills|.85150|.68772|
|1,1,256,254|two decode + two prefill|.85286|.67432|
|129,63,1|three prefills|.87989|.65967|
|64,64,64,64|four cold prefills|.94384|.65229|
|1,1,1,1|four decode|.03046|.02996|
|512|one prefill, changed slot|.76384|.69164|
|1,127,63,321|one decode + three prefill|.99700|.69214|

These timings include capacity work, layout transformations within the GDN
pipeline, state management and role merge. Common q/k normalization is prepared
outside both arms; convolution/projections, host metadata preparation/H2D,
HTTP/model service and profiling overhead are NOT included. This is a bounded
core-GDN non-regression result, not an end-to-end service claim.

`ascendc-gdn-kv-decode-c1` separately passes four slot-changing continuation
rounds, max output4.77e-7/state2.98e-8. Candidate samples14.92–15.58us; native
steady samples16.81–16.96us plus a first58.999us outlier (retained, not used to
advertise a large gain). Four-request decode9 samples26.67–26.83us versus native
27.55–30.42us. pool4's capacity512 decode graph is slower than that minimal decode
probe, hence use pool4 numbers for the full policy table.

### Rejected/diagnostic capsules

- pool1: direct K-V state correct, but old baseline used generic where/copy glue;
  superseded by the source-faithful actual-role control in pool4.
- build5/pool2: attempted in-UB V-K conversion; numerical mismatch/NaNs. Fletcher
  redirected to shared K-V ownership; that conversion branch was removed. Never
  use its timings or binary as a valid candidate.
- decode1: K-V address-only adaptation correct but~55us versus native~30us.
  decode3 increases V tile32→64 (~38us), decode4 also changes tensor orientation
  (~34us). decode2/5/6/7 attempted128-wide blocks but exceeded192KiB UB, including
  variants with single-token specialization/FMA/reduced buffering. Stop retrying
  that tile without a changed memory design. decode8 hit Triton's prohibition
  on return inside a loop; decode9 hoists the invalid-slot guard and processes
  two64-wide tiles per head, giving the accepted performance above.

Local evidence mirrors capsules with `hw3-` prefixes under the established
qwen38-tp2-serving evidence root. No installation, service edit or remote Git push.


## Eight-request service integration and cold-fill DMA race

Service entry is now `betterscale.qwen_worker.MixedWorker`; its colocated README
owns deployment and whole-model scope. The earlier build4/pool4 evidence remains
four-request evidence, NOT qualification for eight active requests.

Five/eight-request cold/warm mixtures exposed a ping/pong H-UB race. The original
MTE3_MTE2 wait only holds the load engine; a cold-state vector Duplicate can still
zero a buffer while the previous warm state's MTE3 store is reading it. In
`elastic-core4`, five requests corrupt first-request head0 and second-request odd
heads; eight requests corrupt all heads of the first two warm requests. This
matches ownership-loop buffer reuse. A separate initial-pool copy did not fix it
(`elastic-core3`); that diagnostic copy is not part of the service.

Kernel source4e21bb1 adds MTE3_V SetFlag/WaitFlag before cold Duplicate. No global
barrier or state gather/scatter is added. Fresh build6 with OWNED_INIT=ON passes
`elastic-core5`: all12 full-core graph/NONE output, convolution and full-bank checks
exact, plus each initial H tile exactly equals the BF16 warm seed or cold zero.
FULL/NONE parity alone was insufficient because both could share bad initial H.
Standalone convolution already passed, isolating the failure to owned GDN.

`elastic-service9` then passes 44 rank-step shadows (5,676 comparisons) with real
HTTP concurrency, changing mixed partitions and state slots. All valid hidden and
128 cache tensors are exact against owned uncaptured execution. This is bounded
service/capture correctness, not semantic equivalence to every native arithmetic
path. The production manifest rejects unfenced build4. Full artifacts remain in
hw3 capsules and local `runs/qwen38-tp2-serving/hw3-elastic-*` mirrors.


## Graph-pool temporary ownership

The service keeps build6's device kernels unchanged. Its native framework host
adapter (`src/betterscale/patches/qwen_gdn/host.cpp`) computes the workspace
extent and allocates temporary H/V/workspace/output tensors at invocation time.
Torch-NPU's allocator places capture-time allocations in the existing shared
graph pool. `GetWorkspaceSize`-style sizing is not itself an allocator; the
framework adapter owns the allocation/lifetime boundary. No manually shared
Python arena or cross-bank pointer registry is needed. Scope is serial compute
replay with independent fenced metadata banks, not concurrent graph execution.
`BETTERSCALE_GDN_HOST_LIBRARY` is mandatory for the state-pool runtime and pinned
alongside the unchanged device library. Build commands and qualifications live
in the mod README rather than a second prototype deployment path.
