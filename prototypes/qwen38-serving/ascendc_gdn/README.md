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
