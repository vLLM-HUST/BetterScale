"""Explicit per-physical-device placement for State completion threads only.

Does not rebind model compute threads, migrate existing pages, or place the
shared Store segment. Set the measured mapping explicitly; never guess topology.
"""
import ctypes
import json
import os
from functools import partial
from pathlib import Path


def cpus_from_list(text):
    cpus = set()
    for part in text.strip().split(","):
        edges = [int(v) for v in part.split("-")]
        if len(edges) == 1:
            cpus.add(edges[0])
        elif len(edges) == 2 and edges[0] <= edges[1]:
            cpus.update(range(edges[0], edges[1] + 1))
        else:
            raise ValueError("Invalid NUMA CPU list")
    return cpus


def resolve_node(mapping, visible, logical_device):
    if not isinstance(mapping, list) or not mapping or any(type(n) is not int or n < 0 for n in mapping):
        raise ValueError("NUMA mapping must list physical-device node ids")
    devices = list(range(len(mapping))) if not visible else [int(v) for v in visible.split(",")]
    if len(set(devices)) != len(devices) or any(d < 0 or d >= len(mapping) for d in devices):
        raise ValueError("Visible devices do not match explicit NUMA mapping")
    if not 0 <= logical_device < len(devices):
        raise ValueError("Logical device is absent from visibility map")
    physical = devices[logical_device]
    return physical, mapping[physical]


def bind_thread(physical, node):
    root = Path(f"/sys/devices/system/node/node{node}")
    cpus = cpus_from_list((root / "cpulist").read_text()) & os.sched_getaffinity(0)
    if not cpus:
        raise RuntimeError("No allowed CPU in requested State NUMA node")
    # Linux NUMA policy and affinity are per-thread; children inherit them.
    lib = ctypes.CDLL("libnuma.so.1", use_errno=True)
    word_bits = ctypes.sizeof(ctypes.c_ulong) * 8
    mask = (ctypes.c_ulong * (node // word_bits + 1))()
    mask[node // word_bits] = 1 << (node % word_bits)
    lib.set_mempolicy.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_ulong), ctypes.c_ulong]
    lib.set_mempolicy.restype = ctypes.c_int
    if lib.set_mempolicy(2, mask, len(mask) * word_bits) != 0:  # MPOL_BIND
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    os.sched_setaffinity(0, cpus)
    print(json.dumps(dict(stage="state-numa-thread",physical_device=physical,
                          node=node,cpus=sorted(cpus))),flush=True)


def initializer(logical_device):
    raw = os.environ.get("BETTERSCALE_PD_STATE_NUMA")
    if raw is None:
        return None
    physical, node = resolve_node(json.loads(raw), os.environ.get("ASCEND_RT_VISIBLE_DEVICES"), logical_device)
    return partial(bind_thread, physical, node)
