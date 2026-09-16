# Resident mover continuation with urgent activation

Opt-in `DEVICE_SERVICE_RESIDENT_MOVES=1` requires the internal-prefix pipeline
and excludes the previous MOVE_QUANTUM chunking experiment. No published worker
or default prototype policy changes.

## Execution protocol

The16 AIV movers retain their descriptor, route map and copy-loop cursor during
one ordinary FETCH/REPACK command. After completed row DMA, a mover checks a
separate generation-tagged urgent mailbox. If another slot has a ready activation,
it runs its assigned activation rows, publishes its urgent completion, and resumes
the original copy loop. Already-finished movers check that mailbox in their idle
loop too. No global join is imposed at every copy row.

The coordinator may publish one urgent activation while an ordinary move is in
flight. It joins all16 urgent completions before advancing activation state. It
does not issue another ordinary vector command while urgent work remains active.
Cube still independently consumes ready work from the two slots. The urgent slot
cannot be the moved slot, and slot/source retirement remains dependent on complete
math and return writes. Maps and cursors are local C++ state across the interruption;
this is not a guarantee that the compiler keeps every scalar in registers/UB.

Activation overwrites the mover's scratch UB. This is safe only because the
interruption point follows completed DMA, routing maps have already been copied
into local state, and the resumed copy reloads its own payload. The protocol does
not interrupt an in-flight DMA, pack unfinished inputs into up, or remove the
full-pack/whole-down readiness boundaries.

This is a bounded urgent-work path, not a general priority queue or full DFC
fusion. It avoids a complete command reissue per chunk; it still has coordinator
publication and per-row polling costs.

## Evidence and timing interpretation

Leaf `runs/persistent-control-20260916T074701Z` passes independent random BF16
expert oracles, broad/hot/zero/one-row/extreme skew, immutable inputs and guards,
plus bounded invalid/missing descriptor termination. Four-card runs exercise real
urgent delivery and resumed movement; both source/layer generations change.

Instrumented engine2 denotes urgent activation on the SAME vector cores, not a
third physical engine/team. Parent movement intervals include that interruption.
The analyzer subtracts urgent intervals on each core before attributing time to
pull/pack, draws exclusive work slices on the same physical vector lanes, and
validates separate mailbox generations,16-core joins and producer-prefix ordering.
Do not count activation time as mover interference or pretend two tasks execute
simultaneously on one AIV core. Coordinator envelope overlaps remain explicitly
inclusive diagnostics; exclusive core intervals are the attribution evidence.

First same-card32/1-source comparison (2ms source1 delay):

| Run / policy | Waves0/1 | Pack-to-return median0/1 | Tail activation move overlap total0/1 |
|---|---|---|---|
|074758 /control|41/39|566.70/531.12us|167.52/128.76us|
|074919 /resident|41/40|555.00/515.86us|48.44/36.76us|

Resident delivers9/6 urgent activations;5/3 ordinary move commands are actually
interrupted (the remainder can be consumed after movers reach their idle loop).
All48 output checks pass. These independently arriving sources have different
natural batch compositions, so the small median/episode improvement is not an
identical-wave causal speedup claim. A reversed-order control is retained below.

## Reversed order and decision

| Run / policy | Waves0/1 | Pack-to-return median0/1 | Tail activation move overlap total0/1 |
|---|---|---|---|
|075056 /resident|42/41|552.63/517.62us|49.88/59.42us|
|075200 /control|40/42|563.48/510.22us|110.10/177.82us|

Resident again performs actual in-command interruptions (5/5 move commands,
9/8 urgent activations) and passes48 outputs plus the causal/physical-lane audit.
The first pair's server envelopes17.379/17.335 ->16.998/16.942ms improve modestly;
the reversed pair17.116/17.068 ->17.061/17.003ms are essentially flat. One server's
median gets slightly worse in that reversed control. Natural pairing differs.

**Keep opt-in.** The repeat supports reduced head-of-line blocking, not a stable
throughput win. This measured component is small relative to the approximately
17ms episode. Do not extrapolate it into closing the broad-expert DFC gap, add
server times together, or count nested urgent work twice. All compact receipts
are in `resident-moves-result.json`; no fresh DFC or real-model serving run is
claimed. Selected cards were released. Bounded hw3/hw0 checks also found idle
process tables, but all reported numerical/performance gates ran locally; no
runtime installation or modification occurred on those hosts.

The analyzer exports `persistent-work-relative.json.gz` with urgent work and
resumed movement on the same physical core lanes. It additionally asserts that
exclusive vector intervals never overlap on one core. This is per-device-relative
clock evidence, not a new distributed alignment.
