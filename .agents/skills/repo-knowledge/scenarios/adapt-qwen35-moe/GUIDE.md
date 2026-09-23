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
versus old1536. This is source-derived, pending runtime observation. A scheduler
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
