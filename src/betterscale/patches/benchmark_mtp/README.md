# Explicit synthetic MTP for controlled throughput benchmarks

The pinned Ascend sampler accepts but ignores the core synthetic configuration.
This opt-in adapter calls the pinned core synthetic greedy rejection kernel and
broadcasts decisions within each TP group. It never changes real sampling unless
`rejection_sample_method=synthetic` is explicitly selected. Generated text in this
mode is not quality evidence. Both comparison arms must use the same adapter.

The donor implementation was qualified at TP2; TP1/TP8 admission is being extended
for separated attention and native eight-card baselines. Fresh distributed leaf
and serving checks are required before benchmark claims on those configurations.
Qwen35's retained nonthinking SPEED-Bench reference is AL2.63. Real MTP execution
qualification remains a separate gate and must not use that forced acceptance.
