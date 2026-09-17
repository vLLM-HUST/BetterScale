# MTP on the persistent expert pool

The full-model prototype now supports `--mtp-tokens K`. Target-only remains the
default. This is not a change to the published BetterScale Worker or its donor
pins. [Integration receipts](mtp-integration-result.json) distinguish passed
transport/graph gates from unqualified model quality and wider-shape numerics.

## Execution and ownership

The existing server is launched once with `--mtp`; its resident catalog contains
48 INT8 target layers and one BF16 MTP layer(index48). Each of four owners keeps
its128 experts per layer,31,551,651,840weight bytes, about1.17GiB extra over the
target-only catalog. The two staging slots and source channels are still shared
across layers. There is no host restart or per-layer GEMM launch loop added to
the server. Different layers select different weights; only matching layers
co-batch. These gates preserve the existing coarse INT8 pipeline ABI, not the
older BF16-only fine-prefix pipeline's performance qualification.

Attention retains the MTP attention/state/shared expert. `RemoteMoE` submits its
routed work as layer48; shared computation precedes collect, then the result is
broadcast within the TP2 source. Both target verification and draft/reconcile
work use decode priority; prompt initialization uses prefill priority. Existing
promotion and bounded decode preference are unchanged.

`mtp_client.py` composes the owned runtime's `Qwen38DraftCabin`,
`Qwen38TargetCabin`, and `Qwen38GreedyWaveCommit` rather than reimplementing
acceptance or GDN/PLE rollback:

1. Target prefill returns the first pending token and target multi stream.
2. Scheme-A initialization shifts prompt/pending tokens against the preceding
   target multi stream to initialize real MTP State.
3. Draft K proposals, verify pending+K rows, then select the accepted endpoint.
4. On all-accepted rows, consume the final proposal into MTP State before moving
   to the bonus token. Execute the fixed reconciliation call on all ranks, with
   inactive rows directed to scratch, to preserve collective order.
5. Advance pending token, position and selected multi stream on device.

Draft, target+PLE, commit and reconciliation all fit inside one FULL graph.
The small shadow gate restores model State and advancing scheduler operands,
while keeping transport/PLE generations monotonic. It compares accepted counts,
committed token cabins and output counts exactly against same-State eager work.
Full-State cloning is correctness-fixture overhead, not a serving operation.

## Passed gates and honest limits

- One TP2+E4, B2/K1: FULL graph token/count shadow exact; actual acceptance,
  rejection and all-accepted reconciliation exercised;598calls/owner drained.
- Two TP2+E4, B2/K1: all four graph shadows exact; first16 generated tokens for
  each request match independent width-one target continuation;1718calls/source
  per owner drained;49-layer catalog observed. All sources/ranks agree.
- Two TP2+E4, B16/K1, short synthetic prompts:161.43 aggregate committed tok/s,
  ~283ms step median,1.4307outputs/request/step. The earlier matched batch/prompt
  target-only fixture gives162.83tok/s/~196ms. This is **no demonstrated net
  throughput gain**. It is not a same-generated-token-work or quality-qualified
  online comparison; K1 B16 sequences differ from the earlier K0 run.

K3/B8 passes same-State eager/graph output comparison, but **fails** strict
width-one target reference. The diagnostic remains failed, not waived:
[shape divergence](mtp-shape-divergence.json). Initial target residuals match
exactly through layers0/1. At layer2, HC normalized inputs still match, but the
BF16 low-rank down projection differs by5.58e-5relative L2 between query widths1
and4. This occurs before that layer's attention and expert calls; later residual
difference grows to~0.187. This proves an early stateless shape-dependent numeric
seed, not that every later discrepancy is explained or that model quality passes.
Do not call this a server protocol failure, nor claim wider-MTP speedups before
independent quality validation. `--reference-tokens` remains fail-closed.

## Reproduction

On hw0, source `/workspace/betterscale-hw0/environment.sh` and enter
`/workspace/betterscale-hw0/repo`. The existing launcher admits idle devices,
waits when occupied, and reclaims only its children.

```bash
bash prototypes/attention-client/qwen38/run_model.sh 0,1,2,3,4,5,6,7 \
  --sources 2 --batch-size 2 --state-gib 4 --prompt-width 3 \
  --mtp-tokens 1 --reference-tokens 16 --decode-graph --decode-steps 16 \
  --align-steady-start --defer-steady-gc
```

The reference runs before timing from a cloned post-prefill State, then restores
that State. Its layer hooks record the first verification/reference divergence;
no diagnostic hook runs in the steady graph window. Failure writes the full
receipt before a compact assertion, avoiding huge token arrays in terminal logs.
This is a short greedy gate, not OpenCompass or a probabilistic rejection sampler.

Current source wire/PLE capacity requires
`batch_size * max(prompt_width, K+1) <= 32`. Thus B16 fits K1 but not K3; B8 fits
K3. Pages are separately sized for worst-case progress throughout the bounded
run. The existing State capacity estimates were K=0; do not reuse their fixed
per-request GDN/PLE byte count for K>0 candidate storage. Increasing MTP depth
is not free in either verification row budget or State memory.

Host receipts preserve every measured step and count actual committed tokens,
not `batch * (K+1)`. Aggregate throughput uses the two sources' common wall-clock
window. As before, bounded GC deferral is an experimental control, not a new
production GC policy. No whole-model quality or serving-SLO claim follows.
