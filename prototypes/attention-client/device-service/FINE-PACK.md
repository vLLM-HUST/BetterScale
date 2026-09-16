# Row-ready publication removes the whole PACK gate

Opt-in DEVICE_SERVICE_FINE_PACK=1 requires the internal pipeline and excludes
quantum/resident movers. It composes with early down, as tested here. No published
Worker/default changes. This is a dependency cut, not a new expert ordering,
new matrix kernel or a claim of full DFC fusion.

## Ownership and execution

Group freezes all source admission, expert counts and destination offsets before
PACK. Coordinator records the pack command generation in control lines87/88,
then issues REPACK. It may now issue up while that slot is PACK; Cube promotes
the slot to UP. Later vector completion must not demote an already-running UP
back to READY_UP (completion dispatch now recognizes the REPACK kind).

Each destination row has one64-byte ready line, with a single writer. After the
existing Copy waits for MTE3 output completion, that writer publishes the unique
pack command generation. Two slots ×512 rows ×64 bytes = **64KiB** per server.
No dense state payload is added and no layout/route ordering changes.

Before calling the external CATLASS tile operator for an expert, each AIC that
owns a tile waits for all that expert's rows, including all source contributions.
A core with no tile has no input dependency. The wait precedes any GM→L1 issue
for that expert; matrix prefetch does not run ahead of the guard. A single tile
object and its buffers still span all experts. STOP and a bounded poll watchdog
remain active. No per-expert full-pipeline drain is introduced.

Flags are not reset at every wave: the command generation prevents old contents
from satisfying a new slot use. The catalog/slot remain immutable until complete
math and SEND retire; final source DONE still follows all returns. There is no
new source-frame admission during compute, no partial DONE and no new buffer
reuse freedom. Zero-hit groups have no reads and need no row flags.

## Causal measurements

PACK_TIMING remains an opt-in observer. Config19 is row observation storage,
20 is the optional64KiB ready array,21 is optional per-core/expert first tile-call
issue storage. Diagnostics add24MiB to the existing16MiB row observer; both are
absent when disabled. Newly frozen Python configs must match these new binaries.

expert_ready_audit.py checks each observed route/destination, all expected Cube
participants according to the rotating tile assignment, and that **every expert's
first tile-call timestamp on every participating core follows completion of all
its packed rows**. A CPU negative check moved one timestamp inside the valid
core envelope but before required rows were ready; the dependency audit rejected
it. This is not a claim of measured MMAD instruction timestamps.

In observer124816, median first tile call is12.31/11.65us before full PACK
completion. Median5/4.5 experts have begun tile submission by that point. The
full PACK boundary is actually crossed, not merely hidden by naming a waiting
Cube kernel “compute”. Cube routine envelopes nevertheless include input waiting,
so reduced routine gaps alone are not the benefit metric.

## Qualification and cost

Leaf124643 passes hot/broad/zero/skew/one-row independent random BF16 oracles,
immutable inputs, guards and existing invalid/missing-source termination cases.
This is not a dedicated fault injection of an omitted row-ready publication.

Four-card tests use2,3,5,7 with expert servers5/7. All runs pass48 output checks.
Same-work controls have24 paired waves/server and changing inputs/layers:

| Pair / mode | Run | Pack→return median server0/1 |
|---|---|---|
|Uninstrumented control|124722|550.66/517.81us|
|Uninstrumented fine-pack|124749|536.48/494.36us|
|Observed fine-pack|124816|554.08/502.22us|
|Observed control, reverse order|124936|569.05/526.05us|

Both pairs improve about2.6%/4.5%. Do not compare cross-pair rows as if
instrumentation overhead were equal. All configurations use the same newly built
tile/source objects with feature switches; this is incremental to early down,
not to original DFC or an older persistent build.

Heterogeneous32/1 arrivals, source1 delayed2ms:
- fine125003:46/46 waves,2/2 paired, episode15.788/15.777ms,
  pack→return527.58/475.92us.
- control125030:44/46 waves,4/2 paired, episode16.652/16.656ms,
  pack→return551.75/489.39us.

No regression observed, but source pairing changes; do not claim an equal-work
causal throughput improvement from these episode totals. Row polling and memory
traffic are not free. Keep opt-in until broader workloads justify adoption.

Cards were released after all bounded jobs. No fresh donor or real-model
serving experiment is implied. Compact receipts: fine-pack-result.json.

## Reproduce / visual evidence

Build with OUTPUT_DIR=runs/attention-fine-pack-build and build_persistent.sh.
Pass its absolute PERSISTENT_BUILD through run_persistent.sh or run_remote_dfc.sh.
Four-card settings: DEVICE_SERVICE_PERSISTENT=1, PARALLEL=1, BURST=1,
INTERNAL_PIPELINE=1, EARLY_DOWN=1, FINE_PACK=1 (all DEVICE_SERVICE_ prefixes).
Toggle only FINE_PACK for the control. For causal observation additionally set
DEVICE_SERVICE_INTERNAL_TIMING=1 and DEVICE_SERVICE_PACK_TIMING=1.

Run persistent_analyze.py and expert_ready_audit.py on the observer capsule:
runs/remote-dfc-control-20260916T124816Z/analysis/
- persistent-work-relative.json.gz: internal core routine envelopes;
- expert-ready-relative.json.gz: ready-to-first-tile-call waiting intervals.

Clocks are independently zeroed per device, not a cross-device clock fit.
cube_supply_audit.py's old whole-command prerequisite model is intentionally
rejected for fine-pack receipts; use the per-expert audit instead.
