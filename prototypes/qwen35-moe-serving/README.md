# Qwen3.5-35B-A3B BF16 / MTP / native 256K qualification

Active investigation, **not qualified product support**. Native means unchanged
numerical/model/scheduler implementation plus the pinned SD/V1 Mamba postprocess
ABI bridge required by vLLM752a3a50 + Ascend9bf964c. The bridge is copied unchanged
from the completed September22 MTP2 campaign, not an optimization.

Target: BF16 unquantized weights and KV, native262144 context, real MTP2
accept/reject execution, APC align, TP2 initially. Fixed model revision:
ModelScope Qwen/Qwen3.5-35B-A3B `712cf74392b05026a6db2bf213d343747d1f6d45`.
Verify downloaded files against this Git/LFS manifest before model initialization.
Do not install into the shared donor environment or alter its source.

Artifact root: `/workspace/my-ascend-workspace/runs/qwen35-moe-mtp-256k`.
Server launches require selected-card leases, fresh admission and foreign-owner
supervision. Model downloads and CPU preparation do not reserve accelerator cards.

Before qualifying support, establish native startup, actual speculation counters, short coherent
continuations, cold/warm prefix behavior, and real near262144 context requests.
A configured maximum alone is not a long-context acceptance receipt. Adapt
BetterScale's qualified frozen MTP implementation (September22 candidate archive),
not only the older0.5.1 published route. Preserve independent numerical/state
oracles before accepting new GDN/FIA geometry or MoE FULL graph coverage.
Fletcher selected dummy profiles first; exploratory adaptation may precede
real-weight checks, but it cannot substitute for those acceptance gates.

FULL prefill and FULL mixed are explicit acceptance requirements, not optional
follow-up optimization. Stage GDN geometry with `stage_mixed.py`, then the
unqualified full-model port with `stage_service.py`. These are frozen-capsule
staging helpers, not product installers. Preserve native BF16 MoE routing,
grouped GEMMs, shared experts and TP collectives first; do not apply the dense
27B-only MC2 row-projection replacement to MoE. Prove routing changes survive
replay, not merely that one dummy capture finishes.

Use `dummy_profile.py --candidate-full --worker candidate_worker.Worker` only
with the separately pinned candidate runtime and native libraries. The candidate
uses4096 query capacity so a2048-token APC block can coexist with verification
rows. `--long-context-only` issues exact262016+128 cold/warm requests; long-cold
profiling triggers at actual computed context>=240000 rather than guessing the
number of chunked-prefill steps. Inspect schedule records for actual coverage.

AgentX performance is a separate gate: full256k corpus,900s smoke /3600s formal,
no filtering or shortening. Synthetic content requires official matched
SPEED-Bench acceptance calibration and forced-acceptance settings for comparable
speculative performance; true MTP functional validation must not use forced
acceptance. Do not label uncalibrated traffic a compliant Frontier point.

Current functional gate: use `probe.py` with the model-native nonthinking chat
template and EOS respected. It retrieves a unique code from mid-context at
8K/32K/128K/near256K, cold/warm and four concurrent prompts, and requires real
MTP counters. `--raw-stress` retains the earlier64-token forced continuation;
its native cold/warm equality can fail at low-margin branches or after EOS.
Keep those failures distinct from normal chat correctness, and retain both.
These targeted checks and dummy graph evidence do not qualify benchmark points
or make the experimental capsule a released wheel.
