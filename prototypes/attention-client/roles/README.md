# Independent roles with a transparent MoE backend

Unlike the earlier joint model-forward transplant, `role_worker.Worker` retains
native Qwen model.forward, native attention and native token scheduling/sampling.
Only Qwen's FusedMoE construction is replaced during native model loading.
`RemoteExperts.forward` performs native routing and replays the remote-call bank;
there is no host layer coroutine, completion polling or forward-path socket RPC.

The client model never allocates routed expert parameters. The server constructs
its own two-layer BF16 expert shard and starts independently. This initial fixture
uses dummy weights; real-checkpoint role loading, arbitrary layer counts, quantized
weights and distributed attention TP are not qualified by it.

## Startup and capture

`server.py --owner 0|1 --directory <private-dir>` owns expert weights, result
windows and a persistent AIV/AIC engine. The native worker creates its Session
before native warmup: it connects to both explicit local UNIX sockets, checks a
versioned shape/dtype contract, registers one small input window for both peers,
and prebuilds the finite layer/row MoE graph catalog with publication disabled.
Only after capture is complete does the graph config enable publication.

The IPC capabilities travel only through mode0600 UNIX sockets in a mode0700
session directory. No pickle, network discovery, logged keys or inherited pipes.
Forward only uses stable device addresses, routing tensors and generation flags.
The experiment's supervisor starts executables for convenience and bounds failure;
it supplies neither weight payloads nor communication handles.

## Lifetime

The opt-in open_service engine no longer exits after a prescribed number of
layer jobs. EOF is negative(next_generation), published only after a client has
retired its final request. Each owner accepts it only while that source has no
claimed frame; the other source may continue independently. After both EOFs,
engine queues drain and the server checks device-retired counts against the
host's close receipt before releasing any peer mapping.

Diagnostics use bounded rolling storage instead of growing with requests. Idle
open engines do not use the short experiment polling watchdog; the supervising
process deadline/control-channel timeout remains mandatory. A crashed peer is
fail-stop, not transparent recovery. Reconnect, cancellation reclamation,
membership changes and 32-bit generation rollover remain explicit unsupported
boundaries. Never bypass drain to free an in-flight IPC allocation.

`run.sh` captures the exact sources/binaries and uses existing fail-closed NPU
subset admission. `test_control.py` covers framing, bounded messages, disconnect,
wrong transition and private non-clobbering endpoints without NPU access.

The current native gate deliberately uses eager model execution plus captured
MoE calls. It does not claim full-model graph replay; native full graph capture
with remote-call composition needs its own gate. Published Worker defaults are
not modified.

## Qualified checkpoint (2026-09-16)

Four independent processes on four 910B devices completed native two-layer Qwen
3-30B-A3B dummy generation. Sources issued 7 and 8 requests (53 generated tokens),
retiring **44 and 66 MoE calls** respectively, including native profiling calls.
Both servers observed those counts and drained. The numerical-audit variant
retired 46/68 calls and compared remote results with an independent local BF16
expert implementation: maximum relative L2 error **0.000181**. Single-device
empty-session EOF and invalid-generation EOF gates also passed.

Normal client PyTorch NPU peak was **1,663,289,856 bytes**; routed expert parameter
count was zero. Each server owned 1,207,959,552 bytes of dummy expert weights.
These are fixture measurements, not full-model capacity or throughput claims.
The audit intentionally reconstructs reference weights and has a higher peak;
never substitute that peak for the ordinary client measurement.

Compact receipts are in `result.json`. Full local capsules:
- `runs/independent-roles-20260916T080309Z`: normal generation.
- `runs/independent-roles-20260916T080615Z`: independent numerical oracle.
- `runs/independent-roles-20260916T080938Z`: EOF boundaries.

Client closure is currently session-wide: device EOF allows another source to
continue, but host teardown waits for both sources. Native Worker shutdown drains
before releasing native storage; its ordering is covered by CPU tests, while the
hardware gates above explicitly invoked drain. Dynamic server membership and
production failure recovery are not implemented.

### Initialization lessons and reproduction

Create the Session at the end of Worker.load_model, not compile_or_warm_up_model:
native memory profiling already executes forward before that hook. Preserve CANN's
PYTHONPATH when adding prototype paths; dropping it breaks format conversion with
missing `tbe`. Prebuild the 64 MoE graphs using one shared scratch pool, not 64
private expandable pools (the latter exhausted virtual-address reservations).
Inputs and returned outputs remain persistent **outside** that shared scratch
pool: a later native residual consumer may still need the previous MoE output.

`run.sh` requires PERSISTENT_BUILD and DEVICE_SERVICE_SOURCE_BUILD pointing to
qualified binaries. The persistent kernel and Python configuration must be built
from matching sources: open_service adds configuration word 16. Do not combine
this launcher with an older persistent binary. Set EXPERT_ROLE_AUDIT=1 for the
numerical oracle, or ROLE_EOF_ONLY=1 for the single-device EOF gate. Read run.sh
for explicit device admission and capsule paths; no installed donor is modified.

CPU gates: `python3 -m unittest discover -s prototypes/attention-client/roles
-p 'test_*.py'` (join this command onto one shell line).
