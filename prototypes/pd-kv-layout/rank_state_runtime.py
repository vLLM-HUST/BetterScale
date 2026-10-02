"""Opt-in hw81/hw86 private State runtime, built off the compute thread.

Only State threads inherit the explicit NUMA placement. The owning worker must
drain its completion executor and retire checkpoint references before close().
No shared Store, shared-memory mapping, or implicit peer selection is involved.
"""
from concurrent.futures import ThreadPoolExecutor
import json
import os
import sys


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
    def __init__(self, pool, receiver, replica, control, transport):
        self.pool, self.receiver, self.replica = pool, receiver, replica
        self.control, self.transport = control, transport

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


def build(runner, worker, config):
    import torch
    from state_numa import bind_thread
    raw = os.environ.get("BETTERSCALE_PD_STATE_NUMA")
    if raw is None:
        raise ValueError("private State requires explicit physical-device NUMA mapping")
    owner, physical, node = owner_for(
        config, worker.rank, runner.device.index, json.loads(raw),
        os.environ.get("ASCEND_RT_VISIBLE_DEVICES"))
    if config.get("pd_mtp") or os.environ.get("BETTERSCALE_PD_COMPRESS_RESIDENT", "0") != "0":
        raise ValueError("private State qualification is target-only and uncompressed")
    budget = config["state_cache_host_bytes"]
    if type(budget) is not int or not 0 < budget <= 128 << 30:
        raise ValueError("invalid private rank host budget")
    def construct():
        bind_thread(physical, node)
        torch.npu.set_device(runner.device)
        import vllm_ascend.vllm_ascend_C
        from rank_state_pool import RankStatePool
        from rank_replica_receiver import RankReplicaReceiver
        from rank_replicator import RankReplicator, HOSTS
        from rank_peer_control import RankControl
        from rank_state_transport import RankStateTransport
        # Keep the CPU-only Mooncake wheel out of model-package dependency
        # resolution. Torch/Ascend are already imported from the pinned runtime.
        site = "/workspace/pd-kv-layout-results/store-cpu-venv/lib/python3.12/site-packages"
        if site not in sys.path:
            sys.path.append(site)
        from mooncake.engine import TransferEngine
        def allocate(size):
            torch.npu.set_device(runner.device)
            return torch.empty(size, dtype=torch.uint8, pin_memory=True)
        pool = RankStatePool(owner, budget, allocate)
        engine = TransferEngine()
        host = HOSTS[owner[0]]
        if engine.initialize(f"{host}:{56320 + physical}", "P2PHANDSHAKE", "tcp", "") != 0:
            raise RuntimeError("private State TransferEngine initialization failed")
        receiver = RankReplicaReceiver(pool, engine, max_pending=config["state_cache_max_pending"])
        replica = RankReplicator(pool, engine)
        control = RankControl(receiver, host, 56400 + physical, set(HOSTS.values())).start()
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
        transport = RankStateTransport(
            pool, f"qwen35-target-private-v1/tp2/head{worker.rank}", submit,
            replica.replicate, verify=os.environ.get("BETTERSCALE_PD_VERIFY_OBJECTS") == "1")
        runtime = RankRuntime(pool, receiver, replica, control, transport)
        transport.release_checkpoint = runtime.release
        return runtime
    with ThreadPoolExecutor(1, thread_name_prefix="rank-state-init") as builder:
        return builder.submit(construct).result()
