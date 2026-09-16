# Native FIA with an owned ACLGraph boundary

For the current **native host plan per wave** (FD included), start at
[the host metadata protocol](HOST-METADATA.zh-CN.md). The static non-FD
path below remains the explicit `OWNED_HOST_FIA=0` control, not the current
default planner. The native adapter now admits the exact FD variant too.

This is a **pinned experimental adapter**, not a new attention kernel or a
portable public ACLNN API. `static_plan.cpp` bootstraps a native launch in an
isolated preloaded process. Its original static mode selects the installed BF16
paged causal non-FD variant; the host-metadata mode also admits FD. It retains its function handle/configuration/opaque tiling bytes,
and substitutes owned device length buffers. No installed runtime is modified.

The installed mixed kernel prepends an FFTS address. The query is therefore at
byte8, not byte0. Public `aclrtPlaceHolderInfo` entries describe inline payload
relocation: K/V ListTensorDesc, Q lengths, KV lengths, and tiling. Removing only
the two length relocations lets those parameters name persistent GM tensors;
keeping their relocations would silently restore bootstrap lengths on replay.
The K/V descriptors and tiling remain immutable capture payloads. Their exact
geometry, address offsets and binary identity are checked; these are not claims
about other CANN releases or variants. Runtime-owned FFTS/overflow handles stay
in the same process/device context. Workspace is separately owned (native plan
workSpaceSize + the installed 910B 16MiB system reserve), never an allocator
pointer borrowed from the bootstrap executor. Output/LSE mode is fixed to no-LSE.

`probe.py` checks original eager FIA against the raw launch, then TWO captured
banks with changing GM length contents and page permutations. A graph capture
is followed by an explicit first replay. `inspect_launch.*` are read-only
boundary diagnostics; they are not the execution wrapper.

## Observed gates, September16

- `runs/owned-wave/fia-launch1`: no intercepted payload because query offset was
  assumed0. Native execution itself passed. `fia-launch2` records the actual
  byte8 query and five relocation entries.
- `fia-static1`: Q16/KV2 heads, four decode rows, lengths up to1024, two banks,
  ten replays; all max error0. No GE execution or attention task updates.
- `fia-static2/run/q1`: Q16/KV8 (Qwen3-0.6B), four decode rows, lengths through
 8192, twelve changing-page replays; all max error0. The later prefill admission
  failed because the recorder incorrectly assumed all argument blobs have3008
  bytes. It now uses relocation offsets: row count changes inline payload size.
- `fia-static3/run/q128` and `q256`: real prefill query widths128/256, one row,
  changing prefixes through4095, two banks/twelve replays each; max error0.

These establish a bounded native static-plan leaf, not general variable query
population, split-KV, quantized attention, sliding windows, arbitrary head sizes,
TP/DP/EP qualification or a performance benefit.

## Serving integration

`../serving/static_attention.py` is enabled by default (`OWNED_STATIC_FIA=1`) via
`../serving/run.sh`. Native vLLM startup and baseline remain unmodified. During
owned model warmup/capture a scoped `forward_impl` override retains native
projection/RoPE/cache writes but dispatches the native plan instead of the FIA
host executor. Bootstrap templates are per geometry; layer/bank plans bind
capture-stable operands. One128MiB workspace is shared by strictly serialized
layers/banks on the owned compute stream, not by concurrent independent streams.

Frame-local INT64 length tensors are written from graph-resident request State:
prefill cursor+chunk; decode cursor+1 for active rows and1 for inactive rows.
Inactive output is discarded and native slot mapping is-1 (no KV write).
Fixed query offsets remain per bucket. `ServingRoot.publish` becomes a no-op:
there are no attention ExternalEvents or task-update publication dependencies.
Native runner fencing and actual LiveModule capture/shadow/N+2 ownership remain.

Changing query geometry requires another admitted plan; changing device length
contents or block-table contents does not. The scoped override now touches only donated attention instances and restores
their prior bindings on exceptions. Nested scopes are rejected; this is still a
serialized owned execution context, not a general plugin registration API.

## Whole-model gate and performance

`fia-small1` (native-first), `fia-small2` (owned-first), `fia-small-legacy1`
(old attention update control), and independent `fia-small-profile1` all exit0
with lifetime receipts. Real Qwen3-0.6B TP1 passes20 graphs/zero runner calls;
static capture uses20 bootstrap templates and560 layer/bank bindings. All1120
Python attention launches occur during warmup/capture, none during replay.
See [the bounded performance result](PERFORMANCE.zh-CN.md): retained means
2.9772s native,2.6789s legacy owned,1.7593s static owned. Profile/TraceLoom prove
1792 FIA device tasks inside64 replays with zero FIA host/update calls. These
claims do not extend the new wrapper to30B or distributed execution.


## Candidate integration safeguards

The shipped candidate defaults to this path; `OWNED_STATIC_FIA=0` remains an
explicit startup rollback. Exact kernel identity (not substring matching), five
unique in-bounds relocation records, ListTensorDesc rank/count and bounded
workspace sums must match before admitting a plan. Rejected bootstraps consume
pending state cleanly. Retired graph plans are released only after root.close;
IDs are not recycled. Native begin/bind/launch calls are explicit checked calls,
never side effects hidden in Python assertions. CPU fixtures cover admission,
clone isolation, release/use-after-release rejection and instance-scope restore.

The full-model freeze preceding the integration tests differs from final
static_attention.py only by moving imports into their consumers. This avoids an
upstream standalone CPU-import cycle; execution, dispatch and bindings are
unchanged. It does not justify another baseline/model load.


## 30B regression: do not generalize the small-model win

The integrated candidate-only full SWE test passes protocol, but retained time
is95.827s against reused native87.561s and historical owned89.809s. A matched
four-step TraceLoom comparison locates attention58.5->70.5us/layer, not dense
matmul or collective regression. Installed native launch diagnostics switch from
non-FD key5000000000010200203 (8 useful tasks) to FD5100000000010200203 (23 blocks)
at~4.5K for Q16/KV2/batch4. This adapter currently admits only the former.
See [the evidence and missing plan contract](REGRESSION-30B.zh-CN.md). The code is
integrated as a research candidate, not a qualified30B performance release.
Preserve the explicit `OWNED_STATIC_FIA=0` rollback and do not silently claim FD.
