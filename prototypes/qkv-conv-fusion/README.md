# QKV/Conv serving integration (qualification in progress)

Target: the qualified Qwen3.5-35B-A3B BF16 TP2/MTP2,16-seat FULL4096
`moe-request-sampling1` serving capsule. This is not the released dense27B
Worker, nor an integration of the whole experimental MoE stack into main.

The installed native artifact is the `ascend-op-prof`4577d42 wheel:
`qkv_fused-0.1.0-cp312-cp312-linux_aarch64.whl`, SHA-256
`6d18d1b96d48e56c680858275dabbf8b27c71cf0f9a87056efcdde3aef84b87d`.
(The launch capsule records and verifies the actual transferred artifact.)
No runtime compilation of this native kernel, shared-runtime overwrite or PyPI
publication is part of the integration. Triton bridges use the pinned donor JIT.

## Real boundary, not microbenchmark shorthand

The donor BF16 projection is **joint QKVZ**,6144 channels perTP2 rank, not
QKV4096 alone. This candidate uses the existing contiguous QKV/Z weight views,
fuses the QKV projection with Conv and computes the gate projection separately.
No new weight copies are intended. Include the split projection cost in A/B.

The real Conv pool is `[slot,5,4096]` forMTP2. Snapshot its first3 rows through
the device slot/initial metadata, run the full-backing producer/consumer, and
scatter only live prefill tails to their original slots. Preserve the last2 rows
and all inactive slots. Native verification Conv still owns accepted-token
semantics. This temporary bridge is explicit traffic/cost, not hidden overhead.
Recurrent state never leaves the original pool. Metadata stays device-dynamic
under FULL replay. Negative sentinel slots and repeated cumulative lengths must
not read or write state. Native Conv writes only3 history rows for prefill even
when the allocated slot is5 rows long.

`stage.py` copies the qualified baseline and makes a candidate with an extracted
`MixedCore.after_conv` body and default Worker integration. Only envelopes2048
and4096 are selected; other shapes retain the original joint projection and core. Runtime selection lives inside
the opaque custom operator; never put it in the polymorphic traced forward. The
private native control10 selects the already-qualified staggered producer; it
is not a serving/user switch. Baseline/candidate selection exists only in the
experiment launcher, selecting separate source trees/processes.

## Gates

1. `probe_pool.py`: joint projection + native Conv versus split fused projection,
   including gate GEMM, state bridges and native verification Conv. Check changed
   slots, cold/warm history, short1/2-token requests, padding and accepted counts
   in replay; verify untouched slots and extra history rows.
2. `probe_core.py`: compare complete GDN output, Conv pool and recurrent state
   using the actual pinned BetterScale metadata and numerical runtime.
3. Real-model MTP retrieval/APC/concurrency acceptance, with real accept/reject.
4. Matched original16-seat AgentX smoke,900 measured seconds, same full256K
   corpus, seed, warmup, forcedAL2.63, KV20.25GiB/rank and hw3 pair. Forced
   acceptance is a performance protocol, not a quality oracle. Report effective
   wire concurrency alongsideC16; do not equate tree slots with GPU saturation.

Work capsule: workspace `runs/operator-response/20260927-qkv-serving01`;
remote `/home/jingyuan/ascend-probes/qkv-serving-20260927-01`.
At the first pool gate (one process, nine timing samples),128 and512 regress;
2048 improved278.30→262.45µs and4096 improved490.44→449.15µs. All12 changed
shape/state cases passed. These are boundary-leaf observations, not replicated
E2E gains. The complete GDN leaf subsequently passed six2K/4K graph-replay cases with
bit-exact output/Conv/recurrent state against the previous composition. The real
35B TP2/MTP2 model passed24 retrieval/APC/concurrency checks through262080
input tokens, exit0; both ranks recorded all30 layers at both selected capacities.
The paired C16 smoke completed:116.244→118.382 output tokens/s (+1.84%),
but P95 TTFT worsened17.60%. This was one arrival-limited pair, not a
repeatability or saturated-capacity claim. See [C16.md](C16.md) for the complete
comparison, preserved audit warnings and installed-default boundary.
