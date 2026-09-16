# Qwen3-Next: native attention clients, four independent expert servers

This experimental path keeps Qwen3-Next's native GatedDeltaNet/full attention,
recurrent/KV state, gated shared expert, scheduling and sampling. Only the routed
MoE backend and role-specific weight loading are replaced. It is not enabled by
the released BetterScale Worker and does not change the installed donor.

## Execution and layer ownership

Two TP1 attention processes submit to four expert-only processes. Each server
owns128 of512 experts in every admitted layer. The client carries no routed
parameters; server loading reads only its checkpoint slices, converts one layer
to NZ at a time, and retains a table of per-layer weight addresses. The underlying
CATLASS GEMM math is reused, not retuned here.

A source has one reusable communication frame. Its descriptor contains generation,
layer and valid row count, followed by top-k IDs and hidden rows. READY is published
after the payload. A server freezes that descriptor for its slot's lifetime;
only same-layer requests coalesce because they use the same weight catalog.
Another layer can occupy the other slot. Packed inputs/intermediates/results are
shared across layers and reused only after the slot retires. Source reuse waits
for all four servers' generation-matched results to be copied locally. EOF remains
source-specific; host teardown is session-wide.

The device call order is:

    copy routing/input -> submit graph -> native gated shared MLP
                       -> collect graph -> routed weighted reduce + shared

The collector is not queued ahead of shared computation. The native shared MLP
includes its sigmoid gate, and its result is added exactly once. A fresh final
output protects native residual consumers from later reuse of the graph bank.
The banks share scratch, retain private IO, and carry a device layer field, so
48 layers do not require48 copies of the entire row-bucket graph catalog.
There is no host completion polling or per-forward socket RPC.

## Bounds and caveats

- Hidden2048/intermediate512/top-k10; two sources/four equal owners;1–32 rows.
- Four-layer dummy fixture covers three GDN layers and one full-attention layer.
- Full target is48 layers; MTP is deliberately off.
- Native model remains eager; only submit/collect are captured. Whole-model FULL
  graph integration has not been qualified.
- Probabilities are converted to BF16 at the remote weighted-reduce boundary,
  as in the preceding prototype; independent numerical checks use that explicit
  arithmetic. This is not bitwise parity with every native MoE implementation.
- Shared overlap is an execution dependency opportunity, not a measured speedup.
  One512-wide shared expert need not hide ten routed experts or their communication.
- Persistent kernels require an explicit1200-second device execution budget;
  the supervisor remains bounded at1200s and channel drain at600s. This is a
  bounded experimental session, not an indefinitely available production daemon.
- Generation rollover, cancellation/recovery, dynamic membership, arbitrary owner
  counts and untrusted clients are not supported.

## Reproduction and diagnosis

Use the existing pinned runtime and `build.sh` to prepare a frozen Qwen-Next
binary closure. Set PERSISTENT_BUILD and DEVICE_SERVICE_SOURCE_BUILD to that same
build, then run `run.sh <six comma-separated idle devices>`. Admission uses the
existing per-device locks/foreign occupancy check. `NEXT_LAYERS=4` is the default;
`NEXT_REAL=1 NEXT_LAYERS=48` loads full target weights. `EXPERT_ROLE_AUDIT=1` prepares the independent reference before IPC registration,
then compares only small retained tensors after generation. Earlier capsules
constructed reference weights during live service and their client peaks include
those diagnostic allocations; do not report them as ordinary service residency.

Isolation gates: `NEXT_LEAF=1 run.sh <one device>` exercises engine geometry,
hot/broad/zero/skew routes, different layers, actual GEMM and output canaries.
`NEXT_NATIVE_ONLY=1` runs unchanged four-layer donor on one device.
`NEXT_WIRE_ONLY=1` exercises six-card layer/row changes without native attention.
CPU `test_submit_order.py` protects submit/shared/collect ordering and output
ownership. Build sources, process logs and JSON receipts remain in each run capsule.

Initial six-card native run125736 failed with device-stream507011 while leaves
passed. The synchronized diagnostic130713 got through profiling but failed after
cold prefill. The explicit server execution-timeout arm131028 completed all
requests and numerical checks but its temporary daemon diagnostic thread was not
joined before process teardown; a server exitedSIGSEGV. It is NOT a clean pass.
That observer was removed; clean arm131201 passed all six process exits,84 source
calls (30/54), and max relative L2 error0.000254821. Device timeout is the current
supported explanation for the earlier long-lifetime failures, not a claim that
507011 uniquely diagnoses a timeout. No steady-forward global synchronization
was retained.

Retained local capsules:
- `runs/qwen-next-20260916T130200Z`: five local engine route/layer gates.
- `runs/qwen-next-20260916T130542Z`: clean six-card wire/EOF gate.
- `runs/qwen-next-20260916T130816Z`: unchanged native hybrid generation.
- `runs/qwen-next-20260916T131201Z`: clean native four-layer A2/E4 + oracle.
- `runs/persistent-control-20260916T131136Z`: legacy geometry regression.

The first full48 real run131330 loaded all roles and completed native startup;
its request fixture then failed before generation because chat-template output
was not explicitly requested as an integer-ID list. The fixture now specifies
return_dict=False and validates IDs; a CPU tokenizer check covers that contract.
This is not a model/transport failure and not a generation pass.

Full48-layer real-weight acceptance is recorded in `real-result.json`; neither
the fixture nor this short gate establishes serving quality, throughput or DFC parity.

## Full-model investigation and clean gate

Run131945 completed the first two requests on each client, then encountered native
cache preemption with the inherited128MiB fixture KV budget. Run132344 uses1GiB
and completed all five requests (both sources answered5 andParis), but failed in
post-generation numerical audit during reference-weight loading; thus larger KV
alone did NOT fix the complete gate. The relationship between live reference
allocations, remote DMA and the507011 errors is still an inference, not a proven
allocator defect. The next arm prepares oracle outputs before registration; no
large reference weights are loaded during active service. That revised arm passed as described below. Simple answers are not a quality benchmark.

The first failing run also exposed a transient nonnumeric container PID in NPU
monitoring. Admission now treats that as unresolved0 and applies the EXISTING
host-PID/start-time/grace rule; unknown owners remain foreign. Identity reuse and
expired-grace cases have CPU tests. This does not weaken occupancy admission.


Run132952 passed the complete full48 real-weight gate with references prepared
before IPC registration. All six independent processes exited zero; both clients
and all four servers agree on242/1010 completed calls, including startup and audit.
Each server processed1252 waves: this fixture did not demonstrate source coalescing.
The two sources answered “5” and “Paris”; source1's third response hit its16-token
cap. This is a five-request generation smoke test, not an accuracy evaluation.

Independent MoE samples at layer0/3, rows3/5 passed with maximum relative L2
0.000157574 under the documented BF16 probability boundary. Routed weights occupy
36GiB per expert server; clients carry zero routed parameters and4,729,962,240
parameter bytes each. Reported client Torch allocation peaks after reference
preparation reset are6,011,446,272 and6,011,593,728 bytes; these exclude the earlier
oracle peak and are NOT total device residency. KV fixture budget is1GiB/client.

The lifecycle change is sufficient for this successful run, not proof of the root
cause of prior507011 failures. No latency advantage, sustained load, complete
model FULL capture or production failure recovery has been established. The clean
capsule is `runs/qwen-next-20260916T132952Z`; portable receipts are `real-result.json`.


## Six-role native timeline

Set `NEXT_PROFILE=1` on the same real-weight launch to collect into the capsule's
`profile/`. Attention collection begins after native model startup and ends after
the requests, before audit/drain. Expert collection starts immediately before the
persistent replay and includes client startup through EOF. Offline parsing releases
cards before analysis; use the pinned donor Python with:

```bash
python prototypes/attention-client/device-service/profile_export.py CAPSULE \
  --roles attention0 attention1 expert0 expert1 expert2 expert3 \
  --label attention2-expert4 \
  --scope 'Qwen Next full48 BF16, short generation; eager A and persistent E'
```

Run `qwen-next-20260916T134603Z` passed all six exits and matching240/1008 calls
(without the two oracle calls/source). TraceLoom analyzed six native profiler DBs
and exported `analysis/attention2-expert4-provider-clock.json.gz` (about3.7MiB).
This restores provider timestamps, NOT a fitted collective endpoint alignment:
IPC roles have no cross-role HCCL collectives. Do not infer microsecond cross-device
ordering accuracy. Native first-event-normalized export is also retained, but
normalizing each role independently destroys their relative startup timing.

The persistent expert AIV/AIC kernels appear as two long resident tasks per server;
profiler task duration includes queue waiting, not just GEMM. The native timeline
cannot expose individual expert waves inside those tasks. Client `neural_collect`
medians are15.18us/7.68us over192/960 calls respectively, including whatever result
wait remains when that kernel executes; this is NOT total remote-expert latency.
Instrumentation and eager host supply also affect this short run. Use it to inspect
the actual client graph boundaries, shared MLP and waits, not to claim throughput
or quantify hidden compute from the long server bars.
