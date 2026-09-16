# Two-source concurrency: capability versus realized batching

The question is whether same-layer arrivals actually share expert computation,
and what happens when layers differ. These are dummy-weight transport/compute
experiments, **not** whole-model serving throughput.

## Independent clients (six devices)

`qwen-next-20260916T155704Z`: A2/E4,32 rows per client, top-k10, identical broad
expert IDs, different hidden inputs. Each episode has eight device-ordered calls
captured in one graph, after two warmup calls. The host rendezvous occurs before
timing; it does not add a server-side wait-to-batch policy. The third episode
requests an initial100us source1 offset; subsequent submissions follow completion
feedback, not a prescribed per-wave delay. Shared MLP and native attention are
absent from this isolated fixture.

| Episode | Client0 total / 8 | Client1 total / 8 | Measured server waves, each owner |
|---|---:|---:|---:|
| Same layer0 | 6.901 ms / 0.863 ms | 6.701 ms / 0.838 ms |16 solo,0 paired|
| Different layers0/1 | 5.933 ms / 0.742 ms | 5.744 ms / 0.718 ms |16 solo,0 paired|
| Same layer2, initial100us offset | 5.812 ms / 0.726 ms | 5.962 ms / 0.745 ms |16 solo,0 paired|

All sources completed30 calls, all four owners agreed on60 total waves including
warmups, and outputs matched the preceding single-call reference for each case.
Do not interpret the first episode's larger time as an intrinsic same-layer
penalty: this is one fixed-order sweep, and it did not batch same-layer work.
Host launch timestamps were within4.4us in the zero-offset cases, but that is
**not proof** that device publications reached each server simultaneously.

Current admission coalesces sources seen together by Accept, or an unclaimed
same-layer source seen after the useful input pull. It does not hold a request
waiting for a partner. A second empty slot may also claim a later source before
the first slot's post-pull admission check. Arrival skew and this early slot
assignment are distinct explanations; this sweep alone does not attribute the
missed opportunity uniquely to either one. No new batching-delay policy was added.

Different-layer tasks correctly remain separate. Coordinator-observed up/down
commands on server0 occupy about86–88% of the measured episode's server envelope;
the two slots can overlap staging with compute. This is not per-core Cube
utilization or a guarantee of hiding all communication.

`concurrency_report.py <capsule>` writes the complete all-owner table and
`analysis/server-phases-relative.json.gz`. That figure has an **independent zero
origin per server**, not cross-device clock alignment. Its phases include
coordinator publication/join cost and are not precise instruction intervals.

## Both sources already ready (single-server control)

`qwen-next-20260916T160001Z`: one local owner, two prepublished32-row sources,
same broad routes and the same server kernel. Alternating same/different-layer
cases,10 runs each; exclude the first two of each case. This deliberately removes
independent-client launch skew and remote transport from the question.

- Same layer: **one wave**, median observed server span **797.67us**.
- Different layers: **two waves**, median **1112.53us**.
- Same-layer batching reduces this controlled span **28.3%**. This includes its
  intended weight reuse; it is not an isolated measurement of scheduler overhead.

Thus batching is useful and implemented, but the independent-client test did
**not** realize it. The next optimization question is admission timing/slot
assignment, not whether unlike-layer tokens can be put into the same GEMM.

## Reproduce

Use a fresh qwen-next build from the confluence README with the complete candidate
flags. For independent clients:

    NEXT_WIRE_ONLY=1 NEXT_WIRE_CONCURRENCY=1 NEXT_PROFILE=1 bash prototypes/attention-client/qwen-next/run.sh <six idle devices>

For the prepublished-source control:

    NEXT_LEAF=1 NEXT_LEAF_CONCURRENCY=1 bash prototypes/attention-client/qwen-next/run.sh <one idle device>

The bounded concurrent fixture retains at most60 waves/420 phase records; it
fails the report rather than treating a wrapped trace as complete evidence.
