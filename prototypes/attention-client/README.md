# Retired attention/expert separation prototypes

AE separation was retired on 2026-09-24. Its source and experiment branches are
preserved in [CubeLander/BetterScale](https://github.com/CubeLander/BetterScale/tree/archive/ae-final).
Start with that fork's `archive/ae-20260924/README.md`; it distinguishes qualified
MOD code, transport prototypes and unqualified event-client experiments.

The exact former main tree remains at
[the pre-retirement snapshot](https://github.com/CubeLander/BetterScale/tree/d5ccbd5201cd8878ced155978bff39712d8ff0df/prototypes/attention-client).
No experimental `expert_service` MOD was merged into product main.
The shared host-admission helper remains in `prototypes/host-admission`;
normal TP/EP, MTP, native DFC and unrelated graph experiments are not retired.
