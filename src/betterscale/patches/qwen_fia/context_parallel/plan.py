"""CPU oracle for original-order quota partitioning, Qwen TP2 verification.

Host lengths select the static ownership plan. Endpoints carry denominators so
the kernel scales coverage using device-authoritative KV lengths; scaled_groups
is the corresponding CPU oracle. Host lengths never determine consumed KV.
"""

from betterscale.models.qwen35.execution_capacity import EXECUTION
from collections import Counter
import struct

CORES = 24
HEADS = 8
DIM = 256
MAX_QUERY = 16
MAX_PARTIALS = 48
LSE_BYTES = ((MAX_PARTIALS * HEADS * MAX_QUERY * 4 + 4095) // 4096) * 4096
PARTIAL_O_BYTES = MAX_PARTIALS * HEADS * MAX_QUERY * DIM * 4
SYSTEM_BYTES = 16 << 20


def schedule(lengths, queries, divisor=18, startup=4):
    if not 1 <= len(lengths) == len(queries) <= EXECUTION:
        raise ValueError(f'Expected1..{EXECUTION} real requests; padding is not live work')
    if any(type(q) is not int or not 1 <= q <= MAX_QUERY for q in queries):
        raise ValueError(f'Only Q1..{MAX_QUERY} owned attention is admitted; queries={queries}')
    if any(type(n) is not int or not q <= n <= 262144 for n, q in zip(lengths, queries)):
        raise ValueError('Actual KV must include all query tokens and fit native256K')
    if type(divisor) is not int or not 1 <= divisor <= CORES or type(startup) is not int or startup < 0:
        raise ValueError('Invalid quota parameters')
    tiles = [(n + 511) // 512 for n in lengths]
    total = sum(n + startup for n in tiles)
    eligible = [(n + startup) * divisor > total for n in tiles]

    def pack(cap, cutting):
        groups, group, load = [], [], 0
        for task, n in enumerate(tiles):
            start = 0
            while start < n:
                room = cap - load - startup
                if n - start <= room:
                    group.append((task, start, n)); load += n - start + startup; start = n
                elif cutting and eligible[task] and room > 0:
                    group.append((task, start, start + room)); start += room; load = cap
                elif not group:
                    return None
                else:
                    groups.append(group); group, load = [], 0
                    if len(groups) >= CORES:
                        return None
        if group:
            groups.append(group)
        return groups if len(groups) <= CORES else None

    def bounded(cutting):
        lo, hi = max(startup + 1, (total + CORES - 1) // CORES), total
        while lo < hi:
            mid = (lo + hi) // 2
            if pack(mid, cutting) is None:
                lo = mid + 1
            else:
                hi = mid
        return lo, pack(lo, cutting)

    uncut_cap, uncut = bounded(False)
    split_cap, split = bounded(True)
    cap, groups = (split_cap, split) if split_cap + startup < uncut_cap else (uncut_cap, uncut)
    counts = Counter(t for group in groups for t, _, _ in group)
    return dict(lengths=list(lengths), queries=list(queries), divisor=divisor, startup=startup,
                total_cost=total, eligible=eligible, cap=cap, uncut_cap=uncut_cap,
                split_cap=split_cap, groups=groups,
                loads=[sum(hi-lo+startup for _, lo, hi in group) for group in groups],
                split_nodes=sum(n > 1 for n in counts.values()),
                partials=sum(n for n in counts.values() if n > 1))


def encode(native, plan):
    """Preserve native geometry/scale/scratch; replace only scheduler/partial fields."""
    if len(native) != 2528:
        raise ValueError('Pinned FIA tiling payload must be2528 bytes')
    u = lambda off: struct.unpack_from('<I', native, off)[0]
    if tuple(u(off) for off in (0, 4, 8, 16, 28)) != (HEADS, DIM, DIM, 128, 1):
        raise ValueError('Expected Q8/KV1/D256 paged128 tiling')
    lengths, queries, groups = plan['lengths'], plan['queries'], plan['groups']
    if u(32) not in (len(lengths), len(lengths) + 1):
        raise ValueError('Native batch must match live rows plus optional padding')
    assert len(groups) <= CORES and plan['split_nodes'] <= 23 and plan['partials'] <= 46
    assert plan['split_nodes'] <= len(groups)
    b = bytearray(native) + bytearray(4096 - len(native))
    b[200:2488] = bytes(2288)
    def i(off, index, value): struct.pack_into('<i', b, off + 4 * index, value)
    def q(off, value): struct.pack_into('<Q', b, off, value)
    q(152, LSE_BYTES); q(160, PARTIAL_O_BYTES)
    scratch = sum(struct.unpack_from('<Q', native, off)[0] for off in (56, 64, 72, 80))
    if not 0 < SYSTEM_BYTES + scratch + LSE_BYTES + PARTIAL_O_BYTES <= 128 << 20:
        raise ValueError('Unqualified scratch requirement')
    q(88, scratch + LSE_BYTES + PARTIAL_O_BYTES)
    i(168, 0, plan['split_nodes']); i(172, 0, len(groups))
    counts = Counter(t for group in groups for t, _, _ in group)
    offsets, offset = {}, 0
    for task in sorted(counts):
        if counts[task] > 1:
            offsets[task] = offset
            offset += counts[task] * HEADS * queries[task]
    seen = Counter()
    for core, group in enumerate(groups):
        first_task, lo, _ = group[0]
        last_task, _, hi = group[-1]
        for j, value in enumerate((first_task, 0, 0, lo, last_task, 0, 0, hi)):
            i(200 + 104*j, core, value)
        first, used = None, 0
        for task, _, _ in group:
            if counts[task] > 1:
                here = offsets[task] + seen[task] * HEADS * queries[task]
                if first is None: first = here
                assert here == first + used
                used += HEADS * queries[task]; seen[task] += 1
        q(1032 + 8*core, first or 0); q(1240 + 8*core, (first or 0)*DIM)
        struct.pack_into('<8I', b, 2560 + core*32, first_task, last_task, 0, lo, hi, first or 0, (lengths[first_task]+511)//512, (lengths[last_task]+511)//512)
    for core in range(len(groups), CORES):
        i(200, core, 1); i(616, core, 0)
    for node, task in enumerate(offsets):
        for j, value in enumerate((task, 0, HEADS, 0, queries[task], counts[task])):
            i(1448 + 104*j, node, value)
        q(2072 + 8*node, offsets[task]); q(2280 + 8*node, offsets[task]*DIM)
    assert offset*4 <= LSE_BYTES and offset*DIM*4 <= PARTIAL_O_BYTES
    struct.pack_into('<I', b, 3328, len(lengths))
    return bytes(b)


def scaled_groups(plan, actual_lengths):
    """CPU oracle for device-consumed endpoint scaling, including empty pieces."""
    if len(actual_lengths) != len(plan['lengths']):
        raise ValueError('Actual-length count differs from the planned live requests')
    if any(type(n) is not int or not q <= n <= 262144 for n,q in zip(actual_lengths,plan['queries'])):
        raise ValueError('Actual lengths outside the admitted verification domain')
    result=[]
    for group in plan['groups']:
        pieces=[]
        for task,lo,hi in group:
            denominator=(plan['lengths'][task]+511)//512
            tiles=(actual_lengths[task]+511)//512
            pieces.append((task,lo*tiles//denominator,hi*tiles//denominator))
        result.append(pieces)
    return result
