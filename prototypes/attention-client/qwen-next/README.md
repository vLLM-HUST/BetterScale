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
- Default native model remains eager. Opt-in `NEXT_FULL_GRAPH=1` qualifies native
  FULL decode only (below); prefill remains eager. No FULL prefill claim.
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


## FULL decode and the useful single-attention view

`NEXT_FULL_GRAPH=1 NEXT_PROFILE=1` now uses native `FULL_DECODE_ONLY`, capture
bucket1, TP1 target-only. GDN advertises UNIFORM_BATCH, not general FULL prefill.
Do not claim that setting this flag graphs prefills. The remote submit/collect
nodes are inlined into the outer model capture rather than replaying child banks;
shared MLP remains between them. Copies, layer publication, remote retirement and
weighted reduction all belong to that same outer graph.

Use compilation mode0 (native ACL capture without Dynamo): ctypes launches cannot
be traced by Dynamo. The prototype sets runner.use_aclgraph for native metadata
registry initialization and replay updates; upstream otherwise couples this flag
to VLLM_COMPILE. Installed donor files remain untouched. Failed140655 retains the
Dynamo `_FuncPtr` trace error;140855 retains the missing graph workspace registry
before this hook. Neither is a passing graph gate.

Four-layer dummy141043 passes generation, MoE samples and clean drain. All generated
IDs match the earlier eager131201 fixture; this is not a whole-state oracle.
Full48 real141742 also passes all six exits, MoE samples (max relL2 0.000157574),
and the same five short generated outputs as132952. Sources retire290/1058 calls,
including startup/audit; all four servers agree. The intervening141147 queued
launch was cancelled to avoid occupied4/5;141533 was rejected after foreign
occupancy appeared on0 during loading. No timings from those arms are accepted.

For a usable view, parse/export only attention1, which has the longer decode:

```bash
PYTHONPATH=prototypes/attention-client/device-service:$PYTHONPATH \
  python prototypes/attention-client/qwen-next/export_attention.py CAPSULE
```

The pinned donor Python and CANN environment are required for offline parsing.
The exporter retains TraceLoom's native execution hierarchy AND raw provider
tracks, not the flattened distributed view; it omits all server residency bars.
`runs/qwen-next-20260916T141742Z/analysis/attention1-full-decode.json.gz` is about
4.4MiB. It includes explicitly eager prefills as well as FULL decode.
Native evidence:17 `aclmdlRIExecuteAsync` calls and816 instances each of submit,
collect and retire inside graph model85 (17 steps ×48 layers). There are144 eager
instances of each from the three prefills. `full-decode-result.json` retains the
bounded receipt. Model IDs are run-local, never an API contract.

FULL decode collect median is214.29us, versus the earlier eager generation trace's
7.68us across mixed query sizes. They are not equivalent stage-latency controls:
collect measures remaining wait PLUS data gathering, and eager host supply can
hide remote progress before collect starts. The comparison motivates inspection,
not a quantified regression or a claim that shared MLP hides remote experts.
CPU coverage protects direct submit→shared→collect→retire ordering and independent
returned output storage. Full-state graph/eager equivalence and load performance
remain outside this profiling gate.

## Efficient-server confluence (opt-in)

The integrated server retains layer-addressed weights, same-layer coalescing,
source-specific EOF and two cross-layer staging slots. It adds the continuous
up/down scheduler, fine-grained pack readiness, early down-prefix return and a
client-owned token pull/reduce path from the bounded expert prototype.

Enable the complete candidate after rebuilding **this tree**:

```bash
export PERSISTENT_BUILD=$PWD/runs/confluence-next-build
export DEVICE_SERVICE_SOURCE_BUILD=$PERSISTENT_BUILD
OUTPUT_DIR=$PERSISTENT_BUILD bash prototypes/attention-client/qwen-next/build.sh
export DEVICE_SERVICE_INTERNAL_PIPELINE=1 DEVICE_SERVICE_FINE_PACK=1
export DEVICE_SERVICE_EARLY_DOWN=1 DEVICE_SERVICE_EARLY_RETURN=1
export DEVICE_SERVICE_ROUTE_PULL=1 DEVICE_SERVICE_PULL_POLL_CYCLES=250
NEXT_FULL_GRAPH=1 EXPERT_ROLE_AUDIT=1 \
  bash prototypes/attention-client/qwen-next/run.sh 0,1,2,3,4,5
```

For full checkpoint validation add `NEXT_REAL=1 NEXT_LAYERS=48`; default is the
four-layer dummy fixture. This is not enabled by the released Worker.

Each server publishes a source generation after copying a routed contribution.
A client AIV owns its token's FP32 accumulator, pulls ready contributions across
all four owners, and writes one BF16 hidden row. It no longer allocates the
client-side `[rows, topk, hidden]` expansion or launches native unpermute in this
mode. Server-local packed/output storage still exists. Duplicate route slots
remain distinct weighted contributions. A final all-four-server drain precedes
retirement even when an owner has no matching experts. Native gated shared work
still runs **after submit and before collect**, and the returned tensor is fresh
storage rather than the reusable bank.

**ABI boundary:** the old serving configuration used slots16–18 for open service
and the layer table, colliding with the efficient server's scheduling controls.
The merged configuration retains scheduling fields0–23 and uses24=open service,
25=layer weight table,26=layer count. The client uses15 words and contract version3.
The launcher requires the matching `abi.json` in a single fresh build closure;
old binaries must not be reused. Legacy K8 move-quantum/resident modes are rejected
for the K10 geometry. Generation-indexed internal timing arrays are disabled in
open service rather than overflowing during full-model execution.

Arrival-order FP32 reduction may differ slightly from native BF16 unpermute.
Correctness and performance must be assessed separately; the earlier bounded
A2/E2 timing gains do not establish a Qwen-Next serving speedup or DFC parity.
EOF, numerical and complete-model gates for this confluence are recorded below
when completed, not inferred from the predecessor's results.

### Long-service completion race

The first confluence full48 run (`qwen-next-20260916T153434Z`) completed two
requests but timed out in a later decode. A cheap unequal-source wire stress
(`qwen-next-20260916T153945Z`, 290/1058 planned calls) reproduced the hang.
Server3's slot1 showed down command2322 completely finished (`DOWN_ALL_READY=2322`)
but its prefix signal remained2318. This was not an exhausted KV budget.

The coordinator checked prefix readiness, then full readiness. A core completing
between those checks allowed the full join to clear the active Cube slot without
publishing the prefix; the returning AIVs could then wait for a signal no longer
being checked. The full-completion branch now publishes **both** readiness signals
before retiring that observation. Full completion is a stronger condition, so
this does not permit early reads. `NEXT_WIRE_ONLY=1 NEXT_WIRE_STRESS=1` exercises
long unequal lifetimes and changing layers/row counts cheaply; optional
`NEXT_DIAGNOSTICS=1` samples control state on the host for diagnosis only and is
not a performance configuration. Its observer is stopped and joined before
resource teardown.

### Confluence acceptance

`confluence-result.json` records the merged candidate's evidence, separately from
the earlier client-only handoff. Fixed wire stress154246 completed290/1058 calls,
all four owners agreed, and1341 server waves show actual same-layer coalescing.
Full48 real154330 then exited cleanly on all six roles with the same290/1058
counts; each server used1348 waves (no observed coalescing in this neural fixture).
Selected MoE checks reached max relative L2 `0.00015757398796267807`, and short
request token outputs matched the predecessor. This is not comprehensive quality
or full-state equivalence validation.

The candidate's source1 profile contains17 native FULL replays and816 layer
calls. `measure_decode.py <capsule>` selects that graph rather than mixing eager
prefill into its statistics. Its compressed timeline is
`runs/qwen-next-20260916T154330Z/analysis/attention1-full-decode.json.gz`.
Full-real listener rendezvous is now600s, within the existing1200s supervisor:
a subsequent old-path comparison154641 hit the generic180s startup listener
limit before all clients registered, not a forward-result failure.

Same-host old-path control155142 also passed all six exits, matching request token
IDs and17 FULL replays. Source1 median graph span was20.07124ms versus18.91032ms
for the candidate (**5.78% shorter**). Old collect median202.09us plus a separate
unpermute kernel (3.76us median) became fused collect/reduce184.54us. These include
residual server wait, not just data transfer or GEMM. This is one profiled short
A/B, not a sustained-throughput claim. See [CONCURRENCY.md](CONCURRENCY.md) for
same-layer/different-layer arrival experiments and the presently missed batching
opportunity.

## Decode priority and shared-completion promotion

The new role contract is version4, client config16 words (class at15); rebuild
the matched closure. See [priority/README.md](priority/README.md) for the
cross-layer scheduling plane, generation-tagged shared completion, starvation
backstop, six-role gate and its limits. See [model readiness](priority/model-readiness.md)
before substituting the Qwen3.8-Flash-Next W8A8 checkpoint for this BF16 model.
