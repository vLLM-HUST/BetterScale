# Return a completed down prefix while Cube computes the tail

Opt-in DEVICE_SERVICE_EARLY_RETURN=1 requires early down/internal pipeline and
excludes resident/quantum movers. Qualified here with fine-pack enabled. No
published Worker/default policy changes; this is one bounded prefix/tail
continuation, not per-expert DFC-style FIX notification.

## Protocol and lifetime

Down retains one tile object across all groups. At the existing whole-expert
half boundary, it drains pending tile/FIX work once, then each of24 AICs publishes
a down-command generation. No barrier is inserted per expert. The coordinator
joins these prefix completions and publishes per-slot prefix readiness.

AIV can start SEND after prefix readiness AND complete activation of BOTH parts.
That second condition prevents a deadlock: a returning AIV team must not wait
for down while withholding activation needed by that down. One SEND command
loads both source route maps once and keeps them locally across two passes.
It copies prefix-owned route outputs, then waits for complete down retirement
before copying the tail. No source output slice is written twice or shared by
multiple movers; unowned outputs remain untouched.

Control lines89/90 are slot prefix readiness;91/92 are complete-down readiness;
96..119 are Cube-owned prefix completions. Control allocation grows96→128 lines
(2KiB more). Config22 enables the feature. Retained route maps use2KiB of local
C++ storage per mover; no new dense HBM output workspace is introduced.

SEND completion can retire the slot only after all24 Cube completions and all16
mover completions. Only then does the coordinator publish the original source
DONE generations. Readiness flags use the unique down-command generation and
cannot satisfy a newer slot use. Both source frames remain owned through final
retirement; early prefix copies are NOT permission for attention to consume a
partial expert sum or overwrite its request frame.

Existing watchdog/STOP checks cover blocked waits. The leaf includes zero/hot/
skew/one-row cases and independent random BF16 arithmetic, output guards and
source immutability. It is not a dedicated missing-prefix-signal fault injection.

## Measurement and interpretation

Same2,3,5,7 device set; expert servers5/7. Leaf130522 passes. All four-card runs
pass48 output checks; paired controls have24 waves/server, changing source
generations, routes and alternating layer addresses.

| Pair / mode | Run | Pack→return median0/1 | Exposed return tail0/1 |
|---|---|---|---|
|Uninstrumented control|130606|537.59/492.04us|42.06/39.37us|
|Uninstrumented early return|130655|513.32/471.33us|19.62/19.92us|
|Observed early return|130722|518.41/487.54us|23.11/25.63us|
|Observed control, reverse order|130811|537.18/487.26us|see compact receipt|

First pair improves4.5%/4.2%. Reverse-order observed pair improves3.5% on server0
and is essentially flat on server1 (487.26→487.54us). Do not hide that variability
or claim every server/run gets4%. The observed server1 also has higher pack,
up and down durations with the candidate; exposed-return reduction alone does
not determine whole-stage gain. No single-source cause is established.

In130722 the first prefix return read begins65.05/57.42us before complete down
retirement. persistent_analyze.py checks it follows every Cube's completed prefix,
tail reads follow full down completion, both activation parts precede SEND
admission, and SEND retires after down. These are real internal timestamps,
not inferred overlap between two coarse operation names.

The SEND envelope includes waiting for tail readiness; its approximately93–98us
duration is NOT all copy work and must not be compared directly with the old
39–42us SEND as if return became that much slower. The timeline labels this
explicitly and marks the first prefix/tail read plus completed Cube prefixes.
No MMAD instruction utilization is measured.

Heterogeneous32/1 rows with source1 delayed2ms:
- early130942:44/44 waves,4/4 paired, episode15.587/15.544ms,
  pack→return499.14/447.54us.
- control131013:41/42 waves,7/6 paired, episode15.694/15.658ms,
  pack→return515.20/467.88us.

No observed episode regression, but pairing differs and episode gain is small.
Holding AIV while waiting for a tail can delay other-slot preparation; this
remains a real policy tradeoff. Keep opt-in. No new donor or serving claim.

## Reproduce

Build build_persistent.sh into runs/attention-early-return-build and pass its
absolute PERSISTENT_BUILD. Set INTERNAL_PIPELINE=1, EARLY_DOWN=1, FINE_PACK=1,
EARLY_RETURN=1 with DEVICE_SERVICE_ prefixes. Use run_persistent.sh for leaf;
for run_remote_dfc.sh also set PERSISTENT=1, PARALLEL=1, BURST=1. Toggle only
EARLY_RETURN for controls. INTERNAL_TIMING=1 supplies the causal markers.

Compressed internal trace:
runs/remote-dfc-control-20260916T130722Z/analysis/persistent-work-relative.json.gz

Use persistent_analyze.py to regenerate. Device origins are independent, not a
cross-device clock fit. Results: early-return-result.json. Cards were released.

## Weight-cache comparison correction

remote_dfc_control.py repeats the same dummy expert values for its two layers,
but device_joint.py allocates/copies each layer separately and concatenates the
two weight ranges. burst_control.py alternates layer0/1. We therefore repeatedly
visit two physical weight-address sets; we do not generate fresh weights per wave.
dfc_probe.py repeats one weight-address set. Equal numerical values do not imply
equal cache identity. This is an unmatched condition, NOT evidence that caching
explains the entire DFC gap or that scheduling is already equal. These internal
A/B results keep the same address-alternation pattern in both arms.

For the subsequent matched weight-address study including fetch, see
[FAIR-DFC.md](FAIR-DFC.md). Internal scheduling gains do not establish DFC parity.
