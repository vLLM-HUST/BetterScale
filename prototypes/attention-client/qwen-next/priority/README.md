# Shared-completion-aware decode/prefill expert scheduling

An expert request is initially urgent for decode, but a large prefill often has
useful shared-expert computation left on its attention device. That is slack,
not permission to starve it. When shared computation ends, routed expert results
become the client's remaining dependency and the client promotes that generation.

## Execution contract

The current two independent sources each own one immutable, homogeneous token
frame. All tokens in the frame inherit its explicit class: decode=0, prefill=1.
The class is **not inferred from row count**: short prefill and speculative decode
can have identical lengths. The native one-sequence fixture uses num_prefills
from its scheduling metadata. Arbitrary mixed-class frames must be split before
publication; this patch does not implement that future ingress interface.

Wire layout (int32 words): READY at0, descriptor `[generation,layer,rows,class]`
at8, a separate shared-completion generation at16, routes at64, hidden payload
at the geometry-specific offset. The new Next client config has16 int64 words,
class at15. Role contract is version4; server config remains27 words. Rebuild
client/server together; the launcher rejects older15-word client closures.

1. Client publishes payload/descriptor then READY.
2. Attention runs its native gated shared expert.
3. For prefill only, `neural_promote` publishes the generation at source+16.
4. Client collects all owners, retires that generation, and adds shared output.

The promotion node is stream-ordered after shared and before collect. It needs
no host completion read or synchronization. It can be captured inside the outer
graph; the existing Next fixture still qualifies FULL **decode**, not FULL
prefill. Decode does not enqueue an extra no-op promotion node.

The server checks urgency both before admission and for already-staged work.
Only exact positive generation matches promote. Old/future signals cannot raise
another task's priority; promotion is monotone until the slot retires. Source
reuse still waits for every owner. One promoted contributor raises its whole
coalesced batch because its output is now blocking that client.

## One scheduling plane, not one mixed weight catalog

Different layers compete in the same admission and ready-slot priority order.
Only equal-class, same-layer frames coalesce in the Next backend. Each slot
retains its immutable layer weight address; no cross-layer GEMM is invented.
Cube chooses urgent ready work first, preferring down to up inside a selected
slot. Vector likewise prefers urgent result return/activation. Active commands
are not preempted; FETCH/REPACK and an already-issued GEMM remain finite blocking
intervals. Priority is not a zero-latency guarantee or an SLO proof.

A maximum burst of three decode admissions supplies a backstop even without a
promotion: a waiting prefill gets a protected turn. This accounting includes
already-staged prefill, not just the unclaimed mailbox, otherwise the other slot
could keep recycling decode forever. Protected work retains rank-1 until drain;
ordinary urgent work is0 and background prefill1. Equal ranks use admission
order, and source ties rotate. Bounds count admissions, not milliseconds. The
liveness argument assumes kernels/transport finish and sources obey retirement;
crash recovery, rollover and arbitrary source counts remain out of scope.

## Evidence

- CPU test compiles the actual policy header, exercises100 repeating3D/1P
  cycles, promoted ties, empty/single-source work, staged-prefill protection and
  stale/future generation rejection. Submit-order tests cover eager/inline,
  route-pull/unpermute and both classes; no decode promotion node is submitted.
- Three-card `bulk-prefill-20260916T165658Z`:36 cases, two physical input
  publishers and one128-expert server,1024-row capacity/source.32-token decode
  is admitted ahead of1024-token prefill, including different layers. Matching
  promotion raises prefill; stale/future generations do not. Same-layer
  prefill1024+1024 still coalesces in one wave. Selected-route max relative L2=0.
  This is partial-owner protocol/math validation, not full serving throughput.
- Native A2/E4 four-layer dummy FULL decode `qwen-next-20260916T165554Z`:
  all six roles pass; every owner agrees on34/58 calls; native MoE oracle max
  relative L2=0.000254821. Published package defaults are untouched.
- Diagnostic profile `qwen-next-20260916T165349Z` observed one **post-admission**
  promotion at owner1. It predates omission of the decode no-op promotion node;
  final native gate165554 covers that omission. Its compressed attention1 trace
  lives under `analysis/attention1-full-decode.json.gz`; not a performance A/B
  or cross-device clock-aligned trace. No guaranteed online batching, latency
  improvement or full48-layer quality claim follows from these fixtures.

## Reproduce

Run existing Next `build.sh`/`run.sh` with a new matched build closure and
`NEXT_FULL_GRAPH=1 EXPERT_ROLE_AUDIT=1`; enable the continuous/fine-pack/early-down/
early-return/route-pull options documented in the parent README. Use six idle
cards. The large-frame policy gate instead uses three idle cards:

```bash
g++ -std=c++17 -Wall -Wextra -Werror \
  prototypes/attention-client/qwen-next/priority/test_policy.cpp -o /tmp/expert-policy-test
/tmp/expert-policy-test
python3 prototypes/attention-client/qwen-next/bulk-prefill/build.py \
  runs/bulk-priority-build --rows 1024
BULK_SCENARIO=priority BULK_BUILD="$PWD/runs/bulk-priority-build" \
  bash prototypes/attention-client/qwen-next/bulk-prefill/run.sh 2,3,4
```

Rolling trace fields11/12/13 now retain original class/admission ticket/effective
rank. `admitted_promotions` counts in-slot transitions, not already-promoted
admissions. Do not infer starvation or absence of signals from a zero in a short
run: shared may finish before admission. Long-running adversarial arrival/SLO
measurement remains separate from these finite protocol gates.
