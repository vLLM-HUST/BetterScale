# Qualify rank-private pinned host capacity and DMA

Enter when an apparently roomy rank cache fails host allocation, when changing
the host allocator/registration API, or before claiming80% DRAM is physically
reserved. The driver-backed fast path, registered private mappings, logical
object bytes, allocator reservation and NUMA placement are separate contracts.
Use the existing installed CANN/driver APIs before inventing an allocator.
No global sysctls, cache drops, driver replacements or new pageable/shared tier
are authorized by this note.

The v26 failure below is historical. Startup-reserved private VMM arenas and
automatic idle LRU subsequently passed v28/v29 native gates; see the latest
sections before choosing a launch budget. Capacity-only success must NOT
overrule a duplex-bandwidth regression.
Before resuming serving, qualify allocation/placement, exact native DMA in both
directions, lifetime cleanup, private TP replication, then the numerical gate.

The repository's earlier shared-cache research already ruled out using mapped
device aliases as D2D copy endpoints. Do not repeat that experiment. A regular
registered mapping passing single-direction DMA does not prove duplex speed.

## October3: physical pinned-capacity boundary, v26 invalid

v26 used the v24 native nodes (0c1304d) and v25 bounded-admission frontend
(7b77d1b), C48/R56/44GiB D, four C16/R20 P instances. The verified1024-session
Open-SWE pool preserves the previous512-session prefix;4.0 arrivals/s for240s
plans960 sessions. At231.68s D physical6/NUMA5 failed128MiB pinned allocation:
aclrtMallocHostWithCfg207001, drvRetCode6. A later32MiB receiver allocation
also failed. Both nodes were shut down, retaining quarantine semantics.
The run is INVALID:927 sessions launched,3084 requests,2450 completed,
634 streaming failures; never report its throughput as a qualification result.
512 client connections also queued (P95 5.15s); this is a separate pressure
effect, not the original allocation failure.

Published D manifests account for only44–60GiB/rank after Torch power-of-two
rounding (resident95,604,788B→128MiB; FA23,068,672B→32MiB), below128GiB logical
budget. This census excludes in-flight objects and cached allocator blocks.
Logical byte limits are not physical locked-memory reservations.

Isolated hw86 probes, CANN9.1.0 / driver26.0.rc1 / torch_npu2.10.0.post4:
- rank-pinned-capacity-v1: one rank physical6,96GiB pinned passed.
- v2: same rank,52GiB HBM+200GiB pinned passed; NUMA5 free plateaued near13.6GiB.
  Allocation success does NOT prove200GiB local placement.
- v3: eight ranks,52GiB HBM each,128MiB pinned allocations, cap128GiB/rank.
  First failure at483.25GiB aggregate (physical1 at65.75GiB); other ranks stopped.
  MemAvailable still1.45TiB, MemFree182GiB, no cgroup OOM/max events, maps~1350.
  All eight allocators released to0. This is not a proved512GiB driver quota.
- Crucial v3 peak buddyinfo: all nodes nearly exhausted order9+ blocks
  (4KiB base pages =>2MiB), while many order0–8 blocks remained.
  Installed driver host adaptation uses GFP_NORETRY and its huge-page path
  allocates contiguous pages. This supports fragmentation/high-order supply as
  the current hypothesis; do not relabel it an established allocation quota.
- Bounded32MiB private mmap+MADV_NOHUGEPAGE+first-touch+ACL_HOST_REG_PINNED
  registration succeeded; every page landed onNUMA5,4KiB pages; synchronous
  H2D/D2H byte roundtrip and unregister passed. Capacity, async bandwidth and
  lifetime integration still need qualification before replacing the allocator.

Exact Torch host allocator source at git5dd8ef3f9b375b5ae4a83538d5785754148c3302
uses aclrtMallocHostWithCfg with vaFlag1, and falls back only on unsupported
feature, NOT OOM. pinned_mem_register still allocates through that same path
first; toggling it alone does not bypass the failure.
CANN9.1 official HostRegisterV2 docs support private malloc/mmap converted to
locked memory (4KiB aligned, Linux>5.10); this is not a pageable cache tier:
https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/runtimeapi/aclcppdevg_03_2128.html

Artifacts live in /workspace/betterscale-pd-runtime/: v26 memory/process
snapshots and manifest census, rank-pinned-capacity-v1/v2/v3.{py,json,log},
pinned-register-probe.{py,log}. The v26 eight-second native profile completed
before failure under online-rank-v24-D-profile; use it only for diagnostic
cadence attribution, not headline workload qualification. Both hosts require
fresh native process incarnations after this failure (TE endpoint reuse risk).
No global cache drop/sysctl/runtime upgrade/pageable fallback was used.

Follow-on capacity/DMA observations (not yet production adoption):
- v4 private mmap+NOHUGEPAGE+PINNED reached128GiB on EACH of8 ranks with52GiB
  HBM/rank. NUMA maps show the full allocation on each intended node (physical2
  shows128GiB minus one base page in its contiguous mapping). No allocation
  failures. Its first cleanup helper retained exported ctypes buffer references:
  explicit mmap.close raised BufferError; joined process exit reclaimed memory.
  Do not call that an explicit unregister/close lifetime qualification.
- registered-dma-probe: physical6/NUMA5, eight128MiB host buffers, four distinct
  buffers per direction,32 copies/direction, three timed repeats after warmup.
  Every byte roundtripped. Standard torch pinned H2D25.89 /D2H28.32 /duplex25.46
  GB/s per direction. NOHUGEPAGE+PINNED10.60/10.89/5.93; adding MAPPED
  11.24/6.27/5.25. All API registrations/unregistrations returned success.
- MADV_HUGEPAGE first-touch (no sysctl changes; host defrag=madvise) restored
  one-direction25.49/27.82 or25.73/28.15, but duplex still7.91 or5.43GB/s.
  smaps saw~1GiB AnonHugePages. Thus capacity success and nominal pinning alone
  are insufficient; cannot substitute this path into production on those facts.
- v26 DP0 exported profile:80 target slot-map anchors,160 target/draft replays.
  Fast full MTP cycle median78.48ms with compute31.22/communication43.35ms;
  slow median127.78 with compute33.08/communication93.76ms. Uncovered union
  ~1.05ms. Communication includes waits and profiler perturbs cadence; this
  does NOT establish slow kernels or a return-path bottleneck. Replays are NOT
  full MTP decode steps.

Further discriminators, still no allocator replacement:
- HAL HOST_UVA + explicit NORMAL (module33, no continuity flag) exactly matches
  torch pinned DMA25.84/28.24/25.38GB/s, but v5 eight-rank capacity fails at
  517.875GiB aggregate. Peak again exhausts order9+ buddy blocks while584.6GiB
  MemFree remains. All explicit halMemFree calls return0. Thus merely passing
  the NORMAL flag does not remove this physical allocation boundary.
- HAL HOST (without UVA) + NORMAL also passes exact DMA25.85/28.24/25.37.
  Its aggregate capacity is not yet qualified.
- Do NOT pass registered mmap device aliases as D2D memcpy endpoints. The
  earlier shared-cache research already observed507001/SDMA; the repeated
  private-mapping discriminator failed identically. Device aliases are for
  kernel access, not generally memory-copy APIs (official HostRegister docs).
  The failed isolated process exited; fresh physical6 context and subsequent
  torch/driver byte roundtrips succeeded without reset. This failed direction
  must not be repeated or integrated.

CANN host VMM boundary:
- HOST_NUMA id5 with ACL_MEM_NORMAL and independently ACL_DDR_MEM_NORMAL,
  queried minimum granularity2MiB; allocated/mapped128MiB handles return HOST
  pointer attributes pageSize4096. Both pass exact DMA~25.8/28.3/25.4GB/s.
  Pointer pageSize4096 is therefore NOT proof of no physical high-order demand.
- v6 eight-rank HOST_NUMA/ACL_MEM_NORMAL,52GiB HBM/rank,1GiB handles fails at
 258GiB aggregate: physical4/NUMA4 at31GiB. Other ranks31–33GiB. plog861094:
  MallocPhysical→halMemCreate drvRetCode6,1GiB requested. Node4 order9+ nearly
  depleted while other nodes retain large blocks. Explicit NUMA stops remote
  fallback; all successfully allocated handles unmap/free/address-release0.
- Installed svm_master_phy_allocator.c devmm_master_alloc_huge_pages uses
  order9 with GFP_NORETRY and THISNODE for explicit NUMA. Kernel source family
  contains a separate normal-page path; do not infer which userspace wrapper
  selects it from an enum name or pointer-query page size.
- A fresh CANN granularity query remains2MiB before/after
  halSupportFeature(6,FEATURE_SVM_VMM_NORMAL_GRANULARITY)==true. Direct HAL VMM
  remains the narrow discriminator; no system runtime/driver edits were made.

Startup reservation discriminator and opt-in arena:
- Direct HAL VMM HOST_NUMA/NORMAL also returns2MiB granularity and exact
  ~25.9/28.35/25.46GB/s. Do not repeat wrapper changes as a capacity remedy.
- pinned-thp-preflight-probe: physical4/NUMA4, private128GiB MADV_HUGEPAGE
  first-touch then unmap took226.49s, produced~100.81GiB AnonHugePages.
  The same local VMM allocation route subsequently reached101GiB versus31GiB
  in v6; allocation6.83s, explicit frees all0. This strongly supports physical
  fragmentation/high-order availability rather than a fixed allocation quota.
  No global compaction/sysctl/cache drop was performed; compact_memory is
  read-only in this container. Do NOT place this preparation in request paths.
- rank_pinned_arena.py adds an opt-in startup reservation over native HOST_NUMA
  VMM regions, then aligned variable-size suballocation with adjacent coalescing.
  Existing checkpoint/writer/reader ownership keeps buffers alive. Close refuses
  live buffers; uncertain native setup/teardown remains quarantined until exit.
  No shared memory, pageable cache tier, runtime driver allocation or copy added.
- BETTERSCALE_PD_PINNED_ARENA=1 selects it; default remains0. Explicit
  BETTERSCALE_PD_STATE_HOST_GIB sets1..128GiB/rank (default128). Reservation must
  succeed before listener publication. This does NOT solve high-order scarcity
  or establish128GiB local capacity. No automatic THP preparation is implemented.
- rank-arena-state-gate-v1.json: physical0/NUMA6,512MiB reservation,83 lanes,
  three known-value patterns plus untouched rows and post-H2D audit all pass.
  D2H14.76–15.51GB/s; H2D19.63–20.48GB/s; final used/live0 and explicit close
  succeeds. This is a native State gate, not duplex/network/model qualification.
- Arena + pool + runtime + online wiring CPU tests:30 passed. Pressure eviction
  and admission are still outstanding: a bounded arena miss is not permission
  to evict an active generation or silently recompute during an uncertain DMA.

- rank-arena-transfer-v1-{D,P}/result.json: two-host512MiB arenas, three
  116,572,172-byte payloads pass registered Mooncake TCP write, checksum,
  receiver commit and exact83-lane H2D/untouched-row audit. Both pools finish
  used/live0 and explicitly close. Single-stream network0.97–1.05GB/s is a
  correctness-gate observation, NOT aggregate network or model performance.
- evict_idle now provides a fenced two-replica retirement primitive. New turns
  wait through eviction both before and after ingress-permit waiting; active
  sessions cannot be victims. Both TP drop quorums must match before metadata
  and Directory generation are forgotten; partial failure fails closed.
  Sticky P/D placement survives.43 coordinator/wiring/eviction CPU tests pass.
  This is not yet an automatic LRU/capacity admission policy.

- The first full-model arena launch (online-rank-v27,source3cebf35) was NOT
  admitted: hw86 D8 and7/8 hw81 P ranks reserved24GiB each, but hw81 physical5 /
  NUMA2 failed aclrtMallocPhysical207001 before readiness. Both owned node trees
  stopped; no online workload result. NUMA2 had~1.7GiB MemFree and~224.7GiB
  active+inactive file cache, with no order9+ blocks. Host MemAvailable~1.72TiB
  does not imply readily allocatable local pinned capacity. Do not call this
  a24GiB capacity qualification or silently fall back to a remote NUMA node.
- BETTERSCALE_PD_HOST_PRESSURE=1 is a new opt-in coordinator path. It requires
  empty physically reserved arenas and matching live P/D frame-size receipts.
  Host admission reserves the old version plus P/D incremental resident/tail/
  new-page bytes until old versions retire; P/D device permits stay independent.
  Only idle or not-yet-admitted host waiters are LRU candidates; both TP copies
  retire before the Directory forgets a cache generation. Sticky owners survive,
  so later full prompts can recompute a miss. No disk or pageable tier is added.
  The initial operating limit is80% of reserved arena minus audit scratch
  (max_transfers times largest frame), not80% of system DRAM. The20% margin is
  operational fragmentation headroom, NOT a proof against arbitrary arena
  fragmentation. Existing variable-size allocator exhaustion still fails closed.
 95 CPU ledger/coordinator/wiring/eviction tests pass; model/LRU pressure
  qualification is pending. A startup-reserved arena and this ledger are not
  yet permission to present a high-pressure benchmark as valid.

- hw81 pinned-thp-preflight-node2:80GiB strict-node2 anonymous THP first-touch
  took365.39s, yielded only7.33GiB AnonHugePages; after unmap MemFree~80.7GiB
  but order9+ supply only7.35GiB. A48GiB local VMM reservation still failed
  207001. Unlike hw86 NUMA4, this bounded preparation does NOT repair the node.
  Stop repeating allocations/preparation without a new hypothesis. Host-level
  controlled compaction assistance was requested; no sysctl or global cache
  drop performed. Native model qualification may use explicitly smaller pools,
  but must not report those as production-memory or peak-throughput results.
- Arena reservation now occurs on its NUMA-bound helper thread before model
  weight loading; the later State runtime adopts that same reservation rather
  than competing with newly faulted model/file cache after graph capture.
  Startup/ownership CPU test and64 targeted tests pass; model gate pending.
  online_probe --require-host-evictions checks automatic LRU activity, while
  --retire-all retires only its known fixture sessions and verifies all16 rank
  pool/arena usage returns to0. Neither option qualifies peak performance.

- A follow-on audit-lifetime check found Python retained the prior scratch buffer
  while allocating the next page frame. Successful audits now release it
  immediately after the completion/hash fence; failed/uncertain copies still
  quarantine their buffers. The controller conservatively budgets the old
  resident-to-page/page-to-page overlap too, so older qualified worker capsules
  remain covered.72 targeted tests pass.
- An oversized cold turn is now a request-local HostCapacityError before any
  numerical writer or device permit, not a reason to kill unrelated sessions.
  It is checked before evicting cached State. Such rejected workload requests
  still invalidate an end-to-end benchmark; never hide them in a throughput rate.
- online-rank-v28 native capsule2e4e3cd deliberately reserves D24GiB/rank and
  P4GiB/rank for functional LRU qualification only. All16 early reservations
  succeeded. This is NOT the requested production DRAM budget or a peak
  throughput baseline; service readiness/model/pressure gates are still pending.

## Native LRU qualification — October3

online-rank-v28-gate-v1 completed successfully in65.174s with native capsule
2e4e3cd, controller3e89a6b plus startup-cleanup-only probe edit. Both hosts use
MTP2, D C48/R56/44GiB HBM, D24GiB private arena/rank; P4GiB/rank deliberately
forces pressure. All16 native State audits were enabled.128 sessions at4K/128
outputs followed by128 continuations of8 outputs: exact output counts, D cache
handoff checks, post-H2D byte audits and sticky placements all pass. This is
not a model-output identity comparison against unsegmented prefill.

Automatic LRU performed240 evictions before cleanup;16 sessions remained cached.
All128 continuations missed the host cache under the deliberate cyclic pressure
and still completed. Peak State transactions P groups3/4/4/3, D groups3/3/3/3.
Rank arena peak use0.725–0.835GiB. Explicit retirement of the gate's own sessions
then returned all16 pool bytes/checkpoint counts and arena used bytes to0.

Live frame sizes: resident95,608,332B, FA page23,071,424B, max_transfers20.
Controller limits P1,062,377,676B/group and D18,242,246,860B/group conservatively
include old-worker audit-scratch overlap and operating fragmentation headroom.
Do not call the65s gate a throughput result or this P budget production-qualified.

Artifacts: /workspace/betterscale-pd-runtime/online-rank-v28-gate-v1/
(summary.json, control-events.jsonl, retired-memory.json), v28 health/launch/logs.
The first v28-gate attempt omitted client BETTERSCALE_PD_CONTEXT=262144 and was
rejected before model requests; use the v1 result. Probe startup now closes its
client/controller even when that qualification check fails.

After the gate, hw81 NUMA2 order9+ free supply measured13.23GiB while its4GiB
arena was still reserved (online-rank-v28-P-buddy-after-gate.json). Capacity is
time-dependent; the earlier7.35GiB is not a permanent quota. A smaller explicitly
bounded full-context qualification can proceed, but target80%-DRAM reservation
still needs physical-capacity repair/qualification rather than logical promises.

- v29 native candidate026d452 is prepared with P12GiB/D24GiB startup arenas and
  D profiling enabled, same C48/R56/44GiB/MTP2 execution. The P increase is a
  bounded full-context gate: current node2 high-order supply was13.23GiB while
  the previous4GiB arena was still owned, rather than another blind retry of
  the failed24/48GiB demand. It remains far below80% system DRAM. Both v28
  process trees were stopped before starting new TE incarnations. Launch
  receipts now record the explicit nonsecret settings; do not guess flags.

- Independent P/D replica drops now run concurrently, but the controller joins
  both outcomes and validates both TP quorums before metadata removal. This
  removes an unnecessary serial control roundtrip, not a device synchronization
  fence. Cancellation now preserves a nonempty failed-closed reason
  (CancelledError has an empty str), so it cannot accidentally reopen admission.
 98 targeted CPU tests pass; the following native gate must cover this change.


## Full-context arena gate and real SWE supply boundary — v29

Native026d452/controller39774e0, P12GiB/D24GiB per-rank physical arenas,
C48/R56/44GiB D and MTP2: v29-gate passed256 cold4K/128 turns,256 continuations
and an exact262144 total-context turn.113.219s,400 automatic evictions;
post-retirement all16 pool/checkpoint/arena-used counters returned to0.
All16 byte audits were enabled for the gate, then disabled before performance.

The same frozen512-session SWE2/s240s plan completed VALID:2106 requests,
1848 in-window and258 drained,0 failures/misses, all2106 State commits.
Aggregate accepted throughput1878.44 tokens/s; TTFT P95 15.801s. This is an
explicit reduced-host-capacity point, not production peak. Host admission P95
22.12s and all four P ledgers reached7.39GiB limits; D ledger peaks9.11–10.83GiB
versus17GiB limit.1597 cache evictions,426 P loads. Last60s D host dispatch
averaged10.58–12.28 rows with median44.89–45.23ms and P95 77.8–81.9ms.
These are host dispatch timestamps, NOT native device step measurements.
P host-byte supply, not D concurrency, is the demonstrated constraint here.

Artifacts: /workspace/betterscale-pd-runtime/online-rank-v29-gate/,
online-rank-v29-front/control.jsonl, online-rank-v29-D-timing/;
/workspace/swe-workloads/rank-v29-rate2.0/. analyze_online_supply.py now scopes
worker windows explicitly while labeling controller whole-file cohorts/drain.

Per-P-instance physical budgets can now be selected with
BETTERSCALE_PD_P_STATE_HOST_GIB='[32,32,12,48]'; each entry applies to both
private TP ranks of that instance, not a shared pool. D retains the scalar
BETTERSCALE_PD_STATE_HOST_GIB. Existing capacity-normalized routing selects new
P sessions; old sessions remain sticky. This option does not qualify those
budgets until native startup and gates pass. Before v30, hw81 free order9+
GiB by node0..7 was27.46,44.76,4.66,107.70,13.66,74.28,75.02,60.93 while
12GiB/node remained reserved. Thus the bounded next candidate32/32/12/48 uses
available NUMA capacity without retrying the weak node2 at24/48GiB. Physical
availability is time-dependent; stop on startup failure rather than fallback.


## Unequal P pools, native gate and SWE — v30

Native/controller378c85d: P budgets32/32/12/48GiB per rank reserved successfully,
D24GiB unchanged.256 cold4K turns +256 continuations +exact256K gate passed
in121.277s,165 LRU evictions,128 continuation host hits; all16 pools retired to0.
SWE2/s240s then completed VALID:2208 requests (1951 window/257 drained),
0 failures/misses and2208 State commits,2172.05 aggregate accepted tokens/s,
TTFT P95 14.174s. Host admission P95 fell22.12→4.563s;890 evictions and1109 P
loads. All four D host ledgers now reach their17GiB operating ceiling. P peaks
20.66/21.11/6.73/28.28GiB, below group limits. Last60s D mean rows13.47–18.46;
host dispatch median50.37–50.95ms/P95 91.28–95.82ms, NOT device timings.
owner-admitted duration includes earlier host admission and identifies both
D owner and p_instance; do not mislabel it pure P permit wait or group by D
owner to diagnose P skew. Full-session completion is not required by this
bounded arrival plan: unfinished session chains after drain are expected;
all requests actually submitted must succeed/commit.

Artifacts: online-rank-v30-gate/, online-rank-v30-analysis.json,
online-rank-v30-D-buddy.json under /workspace/betterscale-pd-runtime;
/workspace/swe-workloads/rank-v30-rate2.0/. A next scalar D32GiB startup is
bounded by measured minimum free order9+9.99GiB (node2) plus its existing24GiB
reservation to be released. Other nodes have21.55–101.05GiB free order9+.
This is a candidate, not yet a32GiB native capacity qualification. Keep P
budgets and execution fixed for the comparison; do not infer80%-DRAM success.
