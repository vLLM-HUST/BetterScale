# AscendC GDN fork boundary

Source audit: vllm-ascend9bf964cb, September17. This is a source-derived plan,
not a measured AscendC optimization or a built fork. Dynamic Triton feasibility
receipt863dc20 remains the hardware evidence. No installed source changed.

## Retain the compute, change its dynamic contract

Kernel entry selects arch20 only for __CCE_AICORE__==200 (310P compatibility);
otherwise arch22, relevant to the910B experiment. Do not optimize the wrong branch.
Sources under upstream/vllm-ascend/csrc/moe:

- chunk_gated_delta_rule_fwd_h: H recurrence; arch22/gemm/block scheduler reads
  device cu_seqlens, compacts positive-length requests, constructs chunk offsets.
  Task ownership is request/head/Vtile; chunks of a request recur serially.
- chunk_fwd_o: O calculation; arch22 scheduler distributes chunk/head/Vtile tasks.
- Both op_host/op_api adapters turn aclIntArray into tensors using ConvertToTensor;
  the hardware boundary already has tensor arguments. Expose a separate owned
  tensor ABI rather than changing the native operator schema in place.

H uses logical totalTokens as physical head stride, including Catlass layouts,
and logical totalChunks as H stride. O uses physical seqlen for token strides but
logical numChunks for H stride. Changing only H, or merely swapping metadata
pointers, is unsafe. First fork must give both kernels immutable token/chunk
capacity strides distinct from device runtime work counts. Preserve shape/stride
validation and workspace capacity. Empty rows must have an explicit request/state
mapping, not silently compact original state IDs. An active packed prefix plus
empty suffix is the smallest initial contract; internal empty rows can wait.

## Efficiency candidates, not measured attribution

1. Native chunk.py transposes q/k/w/u/g into head-major buffers, then o/v_new/h
   back. Under SUPPRESS_LEVEL<3, the latter h/v_new results are discarded. Eager
   capture can still record these operations. First confirm actual profile tasks;
   remove unused return conversions without touching arithmetic. Avoid redesigning
   all Catlass matrix layouts merely to remove necessary transposes initially.
2. H arch22/gemm/kernel/gdn_fwd_h_kernel.hpp:315-367 initializes H from initial
   state by looping all heads/requests in each AIV; the loop is not partitioned
   by coreIdx/subBlockIdx. This suggests redundant read/cast/write work and
   overlapping writes, not a quantified timing saving. DO NOT simply stripe the
   loop: current readiness signals are paired cross-core events, not a global
   barrier. Tie initialization to the consuming task owner or establish an explicit
   initialization phase/barrier; preserve ping-pong dependencies.
3. H device scheduling repeats metadata prefix work in participating schedulers.
   A shared precomputed device plan could avoid repeated scans, but four request
   rows alone are unlikely the main cost. Measure before adding a planner kernel.
4. Static head/request assignment makes long/short requests uneven work units.
   O chunks are independently schedulable; H chunks depend on previous H.
   A work-balanced whole request/head assignment is possible, but arbitrary chunk
   parallelization changes the algorithm. This is secondary to ABI correctness.

The previous .186-.268ms dynamic-vs-native gap is whole-pipeline timing with
unequal state glue. It does NOT identify H, O, or any candidate above as its cause.

## Smallest responsible fork

Own only H/O plus a separate tensor dispatcher, pinned provenance and original
per-file license notices (source contains both BSD and CANN-license headers).
Use existing Catlass/common headers and CANN build conventions; no whole donor
fork or installed replacement. First keep arithmetic/layout unchanged except
explicit capacity strides and metadata ABI; state gather/scatter can stay outside
the kernels. Do not simultaneously fuse state access or rebalance tasks.

Validation order: compile/load unchanged owned kernels; exact-shape native parity;
single-capture changing counts/lengths/slots incl shorter total and tail; output,
full state and continuation checks; then independently time H, O and transposes
with equal glue. Only then test initialization/unused-conversion optimizations.
Service integration, MTP/PCP and publication are outside this audit.
