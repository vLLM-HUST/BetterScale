# Experimental expert-sharded persistent service

Default remains whole-layer placement. `placement=expert` is contiguous EP2/EP4:
every owner holds all 40 target layers plus the one physical MTP layer, but only
128/64 of each layer's 256 experts. No expert matrix TP and no host forward RPC.

Each TP1 attention source publishes every layer frame to all owners before any
join. Each owner has an independent source READY/DONE mailbox, and executes only
its expert range. Joint collect selects each route's owning output, reduces in
original top-k order in FP32, then casts once to BF16. Retirement follows collect
on the same client stream for every owner; graph banks retain all pointers.
Shared expert computation runs once between publication and collection.

This deliberately restores route-valued return (8x the full-layer reduced return
capacity), not partial BF16 sums across owners. Changed transfer/launch costs are
part of placement performance, not isolated GEMM speed. Maximum frame4096,
top-k8, hidden2048, intermediate512, seven source slots, two server slots. Scratch
continues to cover the worst case of every route on one owner, not average EP
balance. Input/top-k buffers are read-only; source/frame/output windows cannot be
reused until all peer results are consumed. Local route-map -1 means unowned and
must never form a destination address. Owners with zero routes still publish
completion and participate in generation drain.

Build ABI binds placement, owner count, local expert count, target/draft catalog,
and return mode. Mixed binaries/placements fail before device load. Native FULL
decode and real MTP2 are unchanged; changing shape/routing and empty-owner paths
need fresh hardware gates. Reference is native routed FFN plus independent
fixed-order route reduction. Required gates: CPU geometry/ABI/launch ordering,
clean EP2/EP4 builds, bounded changing-route/skew/empty-owner numerical replay,
41-layer native shadows, concurrent MTP HTTP and exact EOF generations. No
throughput or released capability claim follows from source/build alone.
