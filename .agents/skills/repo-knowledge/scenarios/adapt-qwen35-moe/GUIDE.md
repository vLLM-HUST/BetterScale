# Adapt Qwen3.5-35B-A3B to BF16 / MTP / native256K

Use for this MoE port; do not reread the1250-line dense27B history as if it were
current MoE support. Read `prototypes/qwen35-moe-serving/README.md` and current
probe sources. This is active unqualified work, not a released capability.

## Evidence and source boundary

- Released source started at `c685a2b`, rejects `qwen3_5_moe_text` on CPU.
- The newer September22 experimental MTP2 stack is in workspace
  `runs/betterscale-leaderboard/20260922-official-tp2-mtp2/qualified-candidate`;
  source-identity.json and candidate-source.tar.gz bind its frozen origin.
  It has owned K-V candidate state, device continuation, lookahead APC,
  banked target/draft FIA and GDN layout fusion. It is not the released wheel.
- `qualified-native` installs ONLY the pinned SD/V1 Mamba postprocess ABI bridge.
  Its semantics map18 vLLM positional args to the older Ascend kernel; reject
  DS layout or V2 request mapping. Copying this bridge does not enable a MOD.
- `/workspace/strengthen-dsv4` is a separate DIRTY checkout. Read its
  `qualify-qwen-mtp/GUIDE.md` and `mixed-fusion.md` for paid state/lifetime traps;
  do not edit it or use its mutable source for runs.

## Geometry and context boundaries

Official model snapshot ModelScope `712cf74392b05026a6db2bf213d343747d1f6d45`:
40layers =30GDN+10FA; hidden2048;256experts/top8; TP2 GDNqk8/v16/KVdim128,
packedconv4096; FAq8/kv1/head256. Old dense geometry is GDNv24/packed5120,
FAq12/kv2,48GDN+16FA. Do not replace every literal24:24 also counts hardware
cores, MTP padded token capacity and unrelated ABI fields.

Pinned `patch_mamba_config.py` derives attention block size from SSM bytes
and single-K page bytes. New TP2 FP32 SSM16*128*128*4 / (1*256*2) =2048,
versus old1536. Startup confirmed this source-derived value. A scheduler
budget below2048 is invalid with APC align; old graph budget2048 may still fit.
The262144 context contract is not a262144 query wave: chunked prefill remains.

Artifact root `/workspace/my-ascend-workspace/runs/qwen35-moe-mtp-256k` owns
model Git/LFS verification, frozen baseline source, selected-device launcher,
and eventual receipts. `native_worker.Worker` keeps native numerics and graph
policy with only the ABI bridge. Actual speculation counters, near-limit real
requests, cold/warm equality and independent state gates remain required.
AgentX needs matched SPEED-Bench forced-acceptance calibration separately;
functional MTP gates must use real accept/reject, not synthetic forced results.

## First new-geometry operator gate (2026-09-23)

`gdn-recurrence1` under the artifact root passed MTP2 with qk8/v16, two FULL
banks and24 successive device-feedback waves. Every candidate state and the
whole convolution history are checked against independent CPU recurrence:
max state error1.1176e-8, output3.8147e-6; exact conv history. The kernel body
was unchanged; only shape admission and fixture geometry changed. Exit0 and
localcard1 reclaimed. This is NOT mixed-prefill, FIA or model qualification.

Fletcher then selected **dummy model profiles first**, before real-weight
serving. The prepared weight-waiting native job was cancelled before admission;
weight download continues. `dummy_profile.py` / `dummy_worker.py` observe native
BF16/MTP2 with262144 configured context and bounded target/draft decode,
prefill and mixed windows. Actual dispatch/graph records must prove coverage;
random-weight MoE routing and synthetic acceptance are not real-model throughput
or numerical evidence. The dummy receipt alone does not validate256K execution.

## Native dummy execution coverage

`dummy-native2` completed BF16/MTP2/262144-configured TP2 with exit0 and
release evidence. Both ranks agree on six observed dispatches per window:
decodeC8 uses24 tokens/FULL; prefill8193 splits6144+2049/NONE; mixedC4
contains a2061-token NONE wave (four3-token verification rows plus2049 new
prompt tokens), returning to FULL afterward. Thus the dense27B prefill/mixed
graph gap also exists here, but this is not proof its MoE implementation ports.
Native startup confirmed the source-derived2048-token cache block size.

Raw per-rank PROF roots and native msprof exports remain in that capsule;
`coverage.json` and `native-graph/analysis/manifest.json` bind schedule evidence,
rank owners, DB integrity/identity and TraceLoom bf6fb491 diagnostics. Raw kernel
counts include boundary/in-flight work: do not divide all recorded operators
by six and call that a measured wave. No cross-rank clock or throughput claim.
The first attempt failed only because the observer renamed keyword parameters;
retain the native names and the focused CPU regression test.

`hw3-dummy-long1` then completed cold and warm262016-input+128-output requests
(total262144), with cached_tokens0/258048 and service exit0. This is dummy
execution, not real-weight quality. Its long profile windows both captured
late decode: an assumed chunk count was wrong. New captures trigger on actual
computed context>=240000 instead; keep the old raw evidence correctly labeled.

## FULL MoE port gates

Fletcher explicitly requires FULL prefill and FULL mixed, including MoE.
`stage_service.py` stages the bounded candidate; retain the donor BF16 MoE
route first (no EP means stable ALLGATHER strategy in the pinned selector),
not the dense27-only MC2 override requiring128 row projections. Prove changing
expert routing under replay and actual FULL coverage before calling this done.
At block2048, a2048-token scheduler budget can leave less than one aligned
prefill block alongside live verification rows. The candidate expands its
query/FIA fixtures and graph bins to4096; the new envelope needs its own gates.

The first mixed GDN oracle rejected the port: a text substitution24->16 had
also corrupted Q/K offset1024->1016. `stage_mixed.py` now replaces standalone
head literals only, protected by `test_qwen_moe_geometry.py`. Preserve the failed
`hw3-gdn-mixed1` attempt. `hw3-gdn-mixed2` with `moe-full3` passed six mixed
cases/two FULL banks: max output7.63e-6, max state1.014e-4, exact convolution
history against independent CPU recurrence. This is not full-model acceptance.

The frozen candidate changes one donor runner file as well as Python patches;
native and candidate source-pin manifests differ there. The task-owned
`candidate-runtime1` copies the admitted runtime and replaces only that runner
from the immutable September22 capsule. Never substitute the native runner or
relax its pin to bypass this distinction; never edit shared donor sources.

The first full-model candidate stopped before readiness on the C++ host
adapter's retained2048 capacity guard, not a MoE numerical failure. Extend
both Python preprocessing and C++ allocation admission when expanding the
query envelope; rebuild the host adapter and bind its new digest. With that
change, `moe-full4` / `hw3-gdn-large1` passed pure4096-prefill and mixed
[3,2049,2044] in two FULL banks against independent CPU state/output/history
(max output7.63e-6, max state5.54e-5, exact convolution history).

`hw3-fia-moe1` passed120 dual-bank/device-feedback waves versus native eager
FIA: Q8/KV1/head256,4K/256K verification,4096-token prefill/mixed and a smaller
mixed case. `hw3-fia-draft1` passed72 additional waves for compacted draft
padding capacities24/2048/4096. Both observed max output error0. These are
operator-envelope gates, not proof of whole-model FULL or real-weight quality.

## Whole-model dummy FULL coverage

`hw3-full-dummy2` with `moe-full4` completed28 startup graphs, three request
phases and shutdown (exit0). Both ranks agree on all18 recorded dispatches:
decode24FULL; prefill8193 split4096/4096/1FULL; mixed contains2060FULL
(four3-token verification rows plus2048 prefill), followed by13FULL
(four3-token verification rows plus the remaining1 prefill). No NONE fallback.

Native msprof plus TraceLoom bf6fb491 exact launch/body membership confirms,
for every phase/rank, six target launches with40 MoE top-k and80 grouped
matmuls, and six draft launches with2 top-k and4 grouped matmuls. This is
physical MoE-in-FULL evidence, not just configuration/dispatch strings.
See `native-graph/analysis/graph-inventory.json` in that capsule. Replay cost
units are not scheduler steps (their grouping differs across these windows);
use exact launch/member/db/device identity, not cost-unit count or whole-DB
operator counts. Native baseline used8192 query budget; this candidate4096,
on different hosts: none of these dummy profiles is a throughput comparison.

Real model objects subsequently passed the fixed ModelScope Git/LFS manifest
(28 files,71,927,086,369 bytes) and complete SSH rsync to hw3 task `model/`;
`model-verification.json` / `hw3-model-transfer.json` bind that boundary.
`hw3-full-long1` then completed cold/warm262016+128 requests (cached0/260096),
exit0. Cold profile captured actual prefill at computed241664 through260096,
including four4096, one2048 and one1920-token FULL waves; warm captured FULL
3-token decode. Both ranks' exact body inventories again show six40-layer
MoE target launches and six2-layer draft launches. This closes dummy256K
execution and physical long-prefill graph coverage, not model correctness.

Real native `hw3-real-native1` completed all eight cold/warm requests through
262080+64, and observed actual MTP acceptance. Its strict equality gate failed
only at256K: first differing output token index16. Crucially both outputs first
emitted `<|endoftext|>` at index8; this raw-completion stress probe forced
`ignore_eos=True`. Preserve the failure and full token logprobs. `hw3-real-eos-native1` changed only EOS handling/logprob depth and allowed
stop-length output, keeping equality strict. It failed earlier at8193: common
prefix `5.\n\n<think>`, then cold selected double-newline with top-two margin
0.125 logit, while warm had single/double newline exactly tied. This is an
observed low-margin native branch change, not evidence establishing an APC
state bug. Proper chat-template retrieval (`real-chat-controller1`) places a
unique code at mid-context and checks its exact output at8K/32K/128K/256K,
cold/warm and four concurrent requests; this remains a functional check, not
an independent numerical oracle or workload quality score. Do not mistake
post-EOS forced continuation for normal chat quality, or dismiss any pre-EOS
cache divergence as numerical noise without investigation.

Actual model quality, changing routing correctness and calibrated AgentX remain
separate gates; do not publish dummy-derived Frontier points.

## Real-weight functional acceptance (2026-09-23)

`hw3-real-chat-native1` and `hw3-real-chat-full1` both PASS/exit0, with selected
cards reclaimed. Twelve exact chat-template retrievals each (four lengths,
cold/warm, plus four concurrent4097..4190 prompts) returned the mid-context
code correctly. All matched output token sequences agree across native and
FULL; largest selected-token logprob delta0.003682. Native drafted60/accepted55,
FULL50/50. These short functional counters are not SPEED-Bench calibration.
Longest real chat request is262080 input+7 output; the earlier native forced
stress and candidate dummy exercise the exact262144 total boundary. Do not
rewrite seven-token chat completion as64 or claim broad numerical equivalence.

Use the native chat template with `enable_thinking=False` and honor EOS for
semantic checks (`probe.py` default). `--raw-stress` retains the original exact
64-token forced continuation, its strict equality and its known native failure.
Candidate child environment must set `CAPSULE` to the output directory: APC
boundary logging consumes it even when profiling is disabled. The first real
candidate (`hw3-real-eos-full1`) failed on missing CAPSULE before any response,
not on a kernel or model numerical assertion; corrected full-chat run passed.

Formal AgentX MTP evidence is still unavailable. The pinned Ascend V1 rejection
sampler accepts `synthetic_mode` / `synthetic_conditional_rates` parameters but
never uses them; its initializer also omits passing speculative configuration
to the core sampler. Core configuration alone therefore does not establish
forced-acceptance support. Audit the executed route before any calibrated run;
never label silent real acceptance on synthetic content a compliant point.
