"""Opt-in hw81/hw86 private State runtime, built off the compute thread.

Only State threads inherit the explicit NUMA placement. The owning worker must
drain its completion executor and retire checkpoint references before close().
No shared Store, shared-memory mapping, or implicit peer selection is involved.
"""
from concurrent.futures import ThreadPoolExecutor
import json
import os
import sys


def allocation_failure_snapshot(pool, size, physical, node, host_stats):
    """Bounded failure evidence before native fail-closed tears workers down."""
    from pathlib import Path
    status = {}
    for line in Path("/proc/self/status").read_text().splitlines():
        key, _, value = line.partition(":")
        if key in ("VmSize", "VmRSS", "VmLck", "VmPin", "Threads"):
            status[key] = value.strip()
    return dict(stage="rank-pinned-allocation-failed", pid=os.getpid(),
                owner=pool.owner, physical_device=physical, numa_node=node,
                requested_bytes=size, pool_bytes=pool.bytes, budget=pool.budget,
                objects=len(pool.objects), checkpoints=len(pool.groups),
                readers=sum(x.readers for x in pool.objects.values()),
                host_stats=host_stats, process=status,
                maps=len(Path("/proc/self/maps").read_text().splitlines()),
                meminfo=Path("/proc/meminfo").read_text(),
                buddyinfo=Path("/proc/buddyinfo").read_text(),
                node_meminfo=Path(f"/sys/devices/system/node/node{node}/meminfo").read_text(),
                cgroup_current=Path("/sys/fs/cgroup/memory.current").read_text().strip())


def transfer_engine():
    # Load the qualified CPU engine before optional connector imports can bind
    # the system Ascend wheel. Never unload a different native engine in-place.
    from pathlib import Path
    site = "/workspace/pd-kv-layout-results/store-cpu-venv/lib/python3.12/site-packages"
    if site in sys.path:
        sys.path.remove(site)
    sys.path.insert(0, site)
    # This isolated site contains Mooncake/zstandard, not another Torch. Keep
    # the path in spawned model workers as well as this actor process.
    paths=[p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p and p != site]
    os.environ["PYTHONPATH"]=os.pathsep.join([site, *paths])
    import mooncake.engine as module
    if not Path(module.__file__).resolve().is_relative_to(Path(site).resolve()):
        raise RuntimeError("private State loaded an unqualified Mooncake engine")
    return module.TransferEngine


def owner_for(config, tp_rank, logical_device, mapping, visible):
    from state_numa import resolve_node
    side = config["pd_rank_role"]
    instance = config["pd_rank_instance"]
    dp_rank = config["pd_rank_dp"]
    if (side not in ("P", "D") or type(instance) is not int
            or type(dp_rank) is not int or type(tp_rank) is not int
            or tp_rank not in (0, 1)):
        raise ValueError("invalid private rank topology")
    group = instance if side == "P" else dp_rank
    if not 0 <= group < 4 or (side == "P" and dp_rank != 0) or (side == "D" and instance != 0):
        raise ValueError("private backend requires P4 TP2 / D DP4TP2EP8")
    physical, node = resolve_node(mapping, visible, logical_device)
    if physical != 2 * group + tp_rank:
        raise ValueError("private rank owner differs from physical placement")
    return (side, group, tp_rank), physical, node


class RankRuntime:
    def __init__(self, pool, receiver, replica, control, transport, host_arena=None):
        self.pool, self.receiver, self.replica = pool, receiver, replica
        self.control, self.transport = control, transport
        self.host_arena = host_arena

    def release(self, key):
        # Receiver owns incoming operation receipts as well as checkpoint refs.
        # Drop first: a pending transfer must preserve its routing on failure.
        self.receiver.drop(key)
        with self.replica.lock:
            self.replica.routes.pop(key, None)

    def close(self):
        # Call only after CacheWorker.waiters.shutdown(wait=True) and Core
        # retirement. Refuse uncertain lifetimes; do not free on shutdown errors.
        with self.replica.lock:
            if self.replica.registered or self.replica.quarantined:
                raise RuntimeError("outgoing private State not drained")
        if self.transport.quarantined:
            raise RuntimeError("private State DMA is quarantined")
        with self.pool.lock:
            if self.pool.groups:
                raise RuntimeError("private State checkpoints not retired")
        self.control.close()
        self.pool.close()
        if self.host_arena is not None:
            self.host_arena.close()


def arena_configuration(runner, tp_rank, config):
    raw = os.environ.get("BETTERSCALE_PD_STATE_NUMA")
    if raw is None:
        raise ValueError("private State requires explicit physical-device NUMA mapping")
    owner, physical, node = owner_for(
        config, tp_rank, runner.device.index, json.loads(raw),
        os.environ.get("ASCEND_RT_VISIBLE_DEVICES"))
    if os.environ.get("BETTERSCALE_PD_COMPRESS_RESIDENT", "0") != "0":
        raise ValueError("private State qualification requires uncompressed frames")
    budget = config["state_cache_host_bytes"]
    if type(budget) is not int or not 0 < budget <= 128 << 30:
        raise ValueError("invalid private rank host budget")
    arena_mode = os.environ.get("BETTERSCALE_PD_PINNED_ARENA", "0")
    if arena_mode not in ("0", "1"):
        raise ValueError("invalid private pinned arena flag")
    return owner,physical,node,budget,arena_mode


def prepare_host_arena(runner,tp_rank,config):
    """Reserve on an isolated NUMA-bound thread before loading model weights."""
    owner,physical,node,budget,mode=arena_configuration(runner,tp_rank,config)
    if mode=="0":return
    if hasattr(runner,"_pd_reserved_arena"):
        raise RuntimeError("rank host arena already prepared")
    def reserve():
        import torch
        from state_numa import bind_thread
        from rank_pinned_arena import RankPinnedArena
        bind_thread(physical,node);torch.npu.set_device(runner.device)
        arena=RankPinnedArena(budget,node)
        print(json.dumps(dict(stage="rank-pinned-arena-reserved",owner=owner,
                              **arena.stats())),flush=True)
        return arena
    with ThreadPoolExecutor(1,thread_name_prefix="rank-host-reserve") as builder:
        runner._pd_reserved_arena=builder.submit(reserve).result()


def build(runner, worker, config):
    import torch
    from state_numa import bind_thread
    owner,physical,node,budget,arena_mode=arena_configuration(runner,worker.rank,config)
    def construct():
        bind_thread(physical, node)
        torch.npu.set_device(runner.device)
        import vllm_ascend.vllm_ascend_C
        from rank_state_pool import RankStatePool
        from rank_replica_receiver import RankReplicaReceiver
        from rank_replicator import RankReplicator, HOSTS, CONTROL_BASE
        from rank_peer_control import RankControl
        from rank_state_transport import RankStateTransport
        TransferEngine = transfer_engine()
        arena = None
        if arena_mode == "1":
            arena = runner._pd_reserved_arena
            if arena.budget!=budget or arena.node!=node or arena.closed or arena.closing:
                raise RuntimeError("preloaded rank arena differs from runtime placement")
            print(json.dumps(dict(stage="rank-pinned-arena-ready", owner=owner,
                                  **arena.stats())), flush=True)
        def allocate(size):
            torch.npu.set_device(runner.device)
            try:
                return arena(size) if arena is not None else torch.empty(
                    size, dtype=torch.uint8, pin_memory=True)
            except MemoryError:
                # A bounded arena miss has no uncertain driver allocation.
                raise
            except Exception:
                # Never retry, empty a process-global allocator, or replace
                # pinned memory with a slower pageable/shared tier here.
                try:
                    print(json.dumps(allocation_failure_snapshot(
                        pool, size, physical, node, torch.npu.host_memory_stats())),
                        file=sys.stderr, flush=True)
                except Exception:
                    pass  # diagnostics must not hide the original allocation error
                raise
        pool = RankStatePool(owner, budget, allocate)
        engine = TransferEngine()
        host = HOSTS[owner[0]]
        if engine.initialize(f"{host}:{56320 + physical}", "P2PHANDSHAKE", "tcp", "") != 0:
            raise RuntimeError("private State TransferEngine initialization failed")
        receiver = RankReplicaReceiver(pool, engine, max_pending=config["state_cache_max_pending"])
        replica = RankReplicator(pool, engine)
        control = RankControl(receiver, host, CONTROL_BASE + physical, set(HOSTS.values())).start()
        def submit(descriptors, to_host, stream):
            torch.npu.set_device(runner.device)
            with torch.npu.stream(stream):
                begin = torch.npu.Event(enable_timing=True)
                end = torch.npu.Event(enable_timing=True)
                begin.record(stream)
                torch.ops._C_ascend.swap_blocks_batch(*descriptors, 1 if to_host else 0)
                end.record(stream)
            def wait():
                end.synchronize()
                wait.device_seconds = begin.elapsed_time(end) / 1000
            return wait
        wire = "mtp-prefix" if config.get("pd_mtp") else "target"
        transport = RankStateTransport(
            pool, f"qwen35-{wire}-private-v1/tp2/head{worker.rank}", submit,
            replica.replicate, verify=os.environ.get("BETTERSCALE_PD_VERIFY_OBJECTS") == "1")
        runtime = RankRuntime(pool, receiver, replica, control, transport, arena)
        transport.release_checkpoint = runtime.release
        return runtime
    with ThreadPoolExecutor(1, thread_name_prefix="rank-state-init") as builder:
        return builder.submit(construct).result()
