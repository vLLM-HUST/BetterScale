# Study native donor DP+EP

Enter here for native DP/TP launch, cache ownership, skew and profile comparison,
not implementation of our graph patches. Read
[`DONOR_DP.md`](../../../../../prototypes/full-mixed/DONOR_DP.md) for the measured
matrix, native mechanism anchors and the bounded runner/report protocol.

The offline DP API needs one native client per DP rank with `VLLM_DP_*` env;
Ascend itself offsets local devices. TP1 rejects FlashComm1. All clients must
remain alive during EP drain. The runner preserves native scheduling; its worker
extension is observation plus the already-qualified K5/TP8 LCM startup repair,
not FULL-mixed or draft optimization. Never accidentally import `extension.py`
into this baseline while repairing a fixture.

Use the actual scheduler rows for global query-count comparisons. Profiles
contain dummy EP service after local completion and native32-step global finish
checks. Those are not generated tokens. Native reported cache tokens are hybrid,
max-length concurrency equivalents and depend on the prefill budget; do not
multiply a per-engine receipt and call it verified maximum active capacity.

Runs052/053 established a modest occupied decode-cycle DP advantage without a
stable cohort-throughput win. A1K per-DP-rank budget hurt the single8K skew case;
that is not proof that larger-budget DP cannot improve. Keep real-weight results,
synthetic workload scope and native profiler overhead distinct.

The longer native DP traces exposed a frozen TraceLoom export anti-join with a
missing child-edge index. Reuse `profile_tools/analyze.py --jobs 4 --resume`:
it adds the lookup index to derived DBs only and reuses completed native rank
analyses. Read its README before repeating a slow export; do not truncate real
waves or recapture the model to work around this CPU query issue.
