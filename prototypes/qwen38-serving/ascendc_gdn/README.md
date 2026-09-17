# Owned AscendC GDN probe

Experimental H/O fork for Ascend910B2, BF16 inputs / FP32 gates and states,
qk8/v24 heads, K/V128. Not installed or wired into a service. Existing donor
operators remain the oracle. No MTP/PCP or internal empty-row support claimed.

`h/`, `o/`, `common/` copied from vllm-ascend commit
9bf964cb4b87c8cd0d6852c41a55b3c29711fa95, respectively
`csrc/moe/chunk_gated_delta_rule_fwd_h/op_kernel/arch22`,
`csrc/moe/chunk_fwd_o/op_kernel/arch22`, `csrc/moe/common`.
Original per-file copyright/license notices remain. Catlass is an external,
unmodified dependency pinned to41bf90da655bba3c66d0acd7e00abe33960ecfd6
(the donor gitlink); do not substitute a nearby checkout's current revision.

The two small kernel entries select only the tested dtype specialization.
Owned POD tiling replaces GE-generated structs; runtime.py creates matching
ctypes layouts, owns private scratch, outputs, and device tiling tensors.
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
