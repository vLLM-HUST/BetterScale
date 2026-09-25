# Qwen35 independent State lanes — first implementation slice

Experimental, opt-in, **not wired into the serving Worker**. Native runtimes and
qualified graph capsules are unchanged. This is a declaration/allocation and
numerical-addressing cut, not a completed model migration or speed claim.

## Ownership and realization

`state.py` uses BetterScale-owned `LiveModule`, `StateTensor`, `StateDomain`
and backend from `betterscale.live`, transferred from LiveInference. Constructing the module tree declares requirements without
allocating storage; `activate()` realizes all domains and initializes the
continuation. Leaves declare their own tensors; the root composes them.

- GDN: one BF16 extended convolution history per resident seat, and three FP32
  K-V recurrent candidates per seat for MTP2. Convolution addresses seat `r`;
  recurrence addresses `3*r+c`. These are deliberately **not common block IDs**.
- Target/draft attention: separate BF16 K and V token-page lanes, one shared page
  capacity domain. No maximum-context array per resident seat.
- Continuation: resident identity epoch, target/draft progress, accepted-input
  selection, anchor and two proposals. These fields have allocation and initial
  values, **not yet scheduler transition semantics**.
- Execution width is a constraint, not an allocation multiplier. Resident count
  is exact; token pages may be exact or consume the remaining backend budget.

The small-model fixture is the full text config from `Qwen/Qwen3.5-0.8B` revision
`2fc06364715b967f1860aea9cf38778875588b17`, not synthetic toy geometry. Its 18 GDN,
6 target FA and 1 draft FA leaves declare 50 numerical tensors plus 6 continuation
tensors. R5/P32/page128 needs 275.2734375 MiB GDN and 56 MiB FA payload, plus
260 bytes continuation (backend alignment/temporary tensors excluded).

`attach_consumers` is an explicit generation-scoped borrowing seam: it refuses
preallocated native caches and consumers that have not opted into the exact
addressing ABI. Activation publishes resolved tensors; release revokes those
references. An ABI label is a caller contract, not proof of numerical correctness.
Native pointer tables/captures are **not** automatically registered or retired;
consumer rebinding is rejected until that ownership is implemented.

## Reproduce CPU contracts

Source reference: LiveInference `05ac15419c0e73650e687ceb9daffeb7874865f0`.
Its common runtime closure now lives in `src/betterscale/live`; no external
LiveInference install or checkout is needed. See that package README for the
intake boundary and preserved license.

```bash
PYTHON_BIN=/absolute/torch-environment/bin/python \
  ./check_cpu.sh
```

Checks use the real Torch and grouped State backends, including full0.8B
geometry, capacity independence, candidate-row clearing, budget refusal,
initialization rollback and consumer binding/release. CPU pass does not qualify
Ascend model execution.

## Bounded Ascend numerical gate

`probe_gdn_npu.py` allocates the complete small-model State tree on one NPU and
executes one GDN leaf through the experimental BetterScale recurrence and native
Ascend causal convolution. Two NPUGraph banks replay 24 waves, alternating
request order, zero-length rows and acceptance counts, including currentT1
selecting previous candidate3. A fifth resident is retained untouched.

The independent CPU history/recurrence checks every candidate and convolution
row. Recurrence consumes observed BF16 convolution outputs so convolution ULP
rounding does not masquerade as a persistent-State error; convolution outputs
are checked separately. This is not an independent end-to-end model oracle.
External probe graphs are synchronized and reset before closing the State root.

`gdn_candidates.py` is an isolated copy of the previously qualified experimental
candidate kernel (original Git blob `4c1518d498467947f8f8e2b42e71e299a0cf1db7`),
with the geometry guard changed from Qwen35 TP2 H8/HV16 to0.8B TP1 H16/HV16;
formatter-only changes aside, the numerical implementation is unchanged.
The production `src/betterscale/patches/qwen_gdn/decode_kv.py` at this worktree's
base is non-speculative/H8/HV24: do not import it for this test or silently relax
its contract. The prototype copy protects that published path; consolidate only
when a separately qualified production integration owns both interfaces.

Run only through the workspace's selected-device lease/admission/foreign-owner
guard. It needs the pinned native/Ascend runtime from the repo-knowledge scenario,
this checkout's `src`, and an output directory in
`CAPSULE`; no model weights are needed. Do not import NPU modules to inspect this
script outside an admitted workload.

## Remaining integration boundary

No resident directory, eviction, FA page allocator, prefill-to-decode handoff,
real MTP verifier, draft continuation, model loader or owned graph execution has
been connected here. A request finish must eventually retain its hot seat;
reuse needs compatible complete continuation, and eviction must quiesce writers
and invalidate identity before clearing. The old scheduler/blockpool must not
remain the hidden authority behind a new tensor binding. Full-model TP1 output
correctness precedes TP2/MoE qualification or any performance claim.
