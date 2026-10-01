# Qwen35 quota-FIA leaf qualification

`probe.py` compares the built Q8/KV1/D256 kernel with native FIA, two captured
banks, nonuniform Q1..3, contexts through16×256K and guarded output/workspace.
It also checks immutable inputs and an independent CPU reference on short edges.

Build with the main module's `build.py`, then run this probe only inside a fresh,
supervised selected-device lease. It imports torch-npu and initializes logical
NPU0; the launcher must expose exactly one admitted physical card. The native
FIA host adapter must be preloaded before Python startup.

```sh
LD_PRELOAD=/path/to/qualified/libbs_fia.so python probe.py \
  --output /path/to/new-receipt \
  --source ../../src/betterscale/patches/qwen_fia/context_parallel \
  --kernel /path/to/build/libbs_fia_cp.so \
  --native /path/to/qualified/libbs_fia.so --device-lengths
```

`--device-lengths` fixes the host plan while device lengths change across tile
boundaries and down to tiny contexts, including empty partials and13-query
zero-KV padding. Omitting it exercises exact host/device lengths and full/split
plan transitions. This is a correctness gate, not a serving performance score.
The real serving integration is now in main's `wave.py`; no external capsule
stager or alternate Worker is part of the feature.

For a bounded failure investigation, `--case <name>` selects one existing case;
that subset does not qualify the full gate. A numerical mismatch writes a small
failure.json / failure.pt capsule with live/padding errors and independent CPU
reference errors for rows with KV <=4096. Full KV tensors are not dumped.

CANN9.1 observation: native FIA produced nonzero zero-KV padding in the moderate
case, while the candidate produced zero and live-row max error was0.000244.
The probe therefore compares real rows to native and independently demands
exact zero padding in BOTH candidate banks (including initial execution).
It does not relax live-row tolerance. This separates two distinct contracts.
Subsequent extreme-case replay had a real live-row mismatch; CANN9.1 candidate
is not qualified merely by fixing the padding reference.
