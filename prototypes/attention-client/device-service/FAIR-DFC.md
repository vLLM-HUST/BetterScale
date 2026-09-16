# Compare complete expert service, not a favorable internal interval

The address-controlled fixture uses `DFC_WEIGHT_SETS=1|2` and
`DEVICE_SERVICE_WEIGHT_SETS=1|2`. DFC captures TWO consecutive invocations in
both modes, and normalizes external-event timing by128 calls (64 replays).
One mode uses the same physical catalog twice; two alternates two distinct NZ
allocations with equal numerical values. Remote keeps its existing allocation
and changes only the layer address selected by successive source jobs. All other
persistent pipeline flags stay fixed. This tests address reuse, not fresh random
weights, a flushed cache or a modeled cold-memory bound.

Even this is not exact DFC/service equivalence: native DFC performs local+remote
EP2 dispatch and combined owner outputs on two cards; the service uses two sources
and two separate expert servers, returns route outputs and reduces on each client.
Both exclude router/gate computation. Keep the expert servers on the same physical
cards as DFC and report source topology, actual pairing, full source episode and
expert-local spans separately. Remote changing route offsets preserve broad-hit
counts but do not make its descriptor traffic identical to the fixed DFC probe.

## Existing receipt, expanded scope

Reanalysis of `remote-dfc-control-20260916T130722Z` (not a new NPU run):

| Median, us | Server0 | Server1 |
|---|---:|---:|
| First fetch start → final return end |580.48|580.49|
| First fetch start → pack start |56.27|80.90|
| Pack → return |518.41|487.54|
| Up→down math envelope |489.99|453.91|
| Last fetch end → pack start |19.03|18.98|
| Up end → down start |0.26|0.27|
| Down end → return end |23.11|25.63|

Per-wave joins precede aggregation. Medians of components need not sum to the
median of their total. Fetch→return excludes source publication, final client
route reduction/retirement and queueing before first fetch. It is NOT complete
client latency. The new `persistent_analyze.py` fields retain this distinction.

## Source leads, not attributed speedups

- Both our `streaming_gmm.hpp` and pinned native DFC GMM1/GMM2 set the weight
  L2 cache-disable hint for groups with M no larger than the tile M. This weakens
  a blanket cache explanation; a hint does not establish actual transaction
  behavior or identity of a separately built lab binary.
- The small up→down launch gap is already sub-microsecond. Do not repeatedly
  target it as a ten-microsecond scheduler stall.
- After FETCH, the coordinator observes another admission opportunity, then
  `persistent_vector.cpp::Group` builds counts/prefixes and writes five metadata
  arrays serially before REPACK. The approximately19us interval includes more
  than Group itself; it is a measurable preparation lead, not19us guaranteed
  recoverable time.
- FETCH movers currently read the same map buffer Group later rewrites. Moving
  Group across FETCH without separating descriptor ownership introduces a race.
  Late-source coalescing also changes destination offsets; do not freeze grouping
  early by silently disabling work-conserving batching.
- Broad64-expert NZ weights contain384MiB up and192MiB down per server. At least
  one traversal is substantial traffic. The observed math envelope is not all
  idle time; Cube instruction utilization and actual bytes remain unmeasured.

## Fresh same-host address control

All five admitted runs pass on hw0, expert devices0/1; remote source devices2/3.
Capsule: `runs/expert-fair-address-20260916/hw0/`. Exact binaries are copied from
local attention-early-return-build and the lab A2 DFC provider; installed runtime
is hw0 Python3.12/torch-npu, not the local donor environment. Controls are compared
within hw0 only. No serving defaults changed. Compact receipts: fair-dfc-result.json.

| Mode, broad32 tokens/source | DFC call us0/1 | Remote pack→return us0/1 | Remote fetch→return us0/1 |
|---|---:|---:|---:|
|Single physical catalog|414.21/413.31|523.61/474.42|583.72/532.78|
|Alternating two catalogs|486.19/486.04|527.28/480.18|610.24/562.52|
|Single catalog, DFC repeat after remote runs|422.52/423.03|—|—|

DFC address alternation costs about15–18% relative to bracketing single-catalog
runs. Hot8 is nearly unchanged in the first pair. This is direct evidence that
address reuse matters despite the cache hint; actual cache transactions remain
unmeasured. Remote pack→return changes only about1%, whereas its fetch interval
and independent-arrival timing vary. Do not subtract the DFC percentage from
remote latency or call the remaining discrepancy a proven cache miss penalty.

With two catalogs, our pack→return is about8.5% slower on server0 and1.2% faster
on server1 than DFC's COMPLETE invocation. Including fetch makes it25.5%/15.7%
slower, still before client reduction. Therefore we have NOT demonstrated parity.
All remote runs have24 paired waves and48 numerical outputs checked; both source
episodes are about19.6ms. That is about816us/job including startup/drain and
client work, not steady-step latency. Median intervals between server retirements
are668/665us with two catalogs and652/649us with one; these are a useful separate
continuation measurement, not a DFC-equivalent API boundary.

The two-catalog math envelopes are499/452us. There is no large exposed up→down
launch bubble left to remove; the plausible remaining seams are input preparation,
return/client continuation, and math traffic/notification behavior under persistent
AIV activity. No new Cube-instruction utilization claim is made.

Numerical checks use the existing independent BF16 oracle, relative L2<1%, and
output tolerances. Remote causal analyzer verifies producer-before-consumer and
retirement ordering. Native DFC's operator totals do not expose its internal
GEMM timeline. Compressed remote traces are in each mode's analysis directory.

## Hardware admission and portability

Local attempts were cancelled while our own launchers were still waiting; no
foreign process was terminated. hw3's five-minute admission timed out as memory
became occupied despite no process rows. hw0 passed its per-device admission and
released cards after all five runs. Early hw0 setup failures were a missing sibling
libvllm_ascend_kernels.so and overwriting CANN's PYTHONPATH (tbe). Copy the matching
sibling library and PREPEND the capsule source to the sourced CANN PYTHONPATH;
do not install a different compiler/runtime to repair that path mistake. Admission
changes cwd, so all executable/output paths passed to it must be absolute.


For the address-level return-path audit and redundant internal-mode catalogs,
continue with [PREPARATION-RETURN.md](PREPARATION-RETURN.md).

The subsequent opt-in implementation and acceptance are in
[ROUTE-PULL.md](ROUTE-PULL.md), including client-owned fused reduction.
