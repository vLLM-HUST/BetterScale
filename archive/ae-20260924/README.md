# Retired attention/expert separation research

Fletcher retired this line on 2026-09-24. This personal fork preserves the
research; it is not a supported BetterScale MOD or a new release.

This branch combines the qualified transport MOD (`d7546e4`, code `397efcf`)
with transport prototypes from `f98c2b7`. Original commits and distinct experiment
branches are preserved exactly; see [sources.json](sources.json). Event-client
and native-forward experiments remain on `archive/ae-event-client`: they are
not silently merged into the qualified implementation. Historical branches retain
their original upstream pins, build assumptions, licenses and validation limits.

- MOD: `src/betterscale/patches/expert_service`, model `qwen_experts`.
- Network/admission/readiness probes: `prototypes/expert-transport`.
- Early client/server implementations: `prototypes/attention-client`.
- Qualification: `prototypes/expert-service-qualification`.
- [Final transport evidence and limits](transport-results.md).

The final C64/900s/MTP2 experiments reached A4E4 163.91 and A6E2 173.58
output token/s/chip. These gains over our old AE implementation did not justify
the architecture against the alternative frontier. The 2026-09-24 small-fish
TP2/C16 point reaches 368.93 output token/s/chip, decode p90 55.38 token/s,
TTFT p95 0.823s. Different deployment/context mixes preclude claiming a matched
8-card experiment, but the result does not support further AE investment.
Source: https://vllm-hust.sage.org.ai/leaderboard-runs.html?v=small-fish-20260924-3#frontier

No new accelerator runs are needed to recover this archive. Large raw traces,
weights, binary providers and machine-local environments are not copied into Git.
The report preserves their evidence locators, not a promise of portable runtime
paths. Read each prototype's pinned-source assumptions before attempting replay.
