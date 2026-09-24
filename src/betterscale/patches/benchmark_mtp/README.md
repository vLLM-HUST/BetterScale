# Explicit synthetic MTP for controlled throughput benchmarks

The pinned Ascend sampler accepts but ignores the core synthetic configuration.
This opt-in adapter calls the pinned core synthetic greedy rejection kernel and
broadcasts decisions within each TP group. It never changes real sampling unless
`rejection_sample_method=synthetic` is explicitly selected. Generated text in this
mode is not quality evidence. Both comparison arms must use the same adapter.

The donor implementation was qualified at TP2. TP1/TP8 distributed leaf gates
now pass12 independent CPU-oracle cases per rank, including rank RNG disagreement
and native global argmax. Serving integration remains a distinct required gate
before benchmark claims; the leaf result alone is not an HTTP throughput point.
Qwen35's retained nonthinking SPEED-Bench reference is AL2.63. Real MTP execution
qualification remains a separate gate and must not use that forced acceptance.
