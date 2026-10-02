"""Opt-in quota hooks for Qwen35 target decode/verification attention.

Worker admission and capture-key selection remain outside this adapter. Banked
metadata publication remains owned by wave.Frame, including device feedback.
"""

from betterscale.models.qwen35.execution_capacity import EXECUTION
from betterscale.models.qwen35.count_policy import SPEC_CAPACITIES
import ctypes
import copy
from itertools import accumulate
import os
import logging
import struct

from .plan import encode, schedule

CAPACITIES = tuple(sorted(set(SPEC_CAPACITIES) | {16}))
_LIBRARY = None


def enabled(tokens):
    return os.environ.get('BETTERSCALE_CONTEXT_PARALLEL') == '1' and tokens in CAPACITIES


def capture_metadata(metadata, tokens):
    """Disposable capture only: match GDN's clamped queries, then pad with KV0.

    Caller must prove startup owns an empty real-request pool. Never modify the
    shared layer metadata or the actual device KV-length source.
    """
    ends = list(metadata.actual_seq_lengths_q)
    lengths = list(metadata.seq_lens_list)
    queries = [hi-lo for lo,hi in zip([0]+ends,ends)]
    live = len(lengths) - int(lengths[-1] == 0)
    queries = [min(q,3) for q in queries[:live]]
    if not 1 <= live <= EXECUTION or any(q <= 0 for q in queries):
        raise ValueError('Unqualified disposable capture metadata')
    result = copy.copy(metadata)
    result.actual_seq_lengths_q = list(accumulate(queries))
    result.seq_lens_list = lengths[:live]
    if result.actual_seq_lengths_q[-1] < tokens:
        result.actual_seq_lengths_q.append(tokens)
        result.seq_lens_list.append(0)
    if result.actual_seq_lengths_q[-1] != tokens:
        raise ValueError('Capture queries exceed their fixed token capacity')
    return result


def target_metadata(metadata, live_requests, tokens):
    """Translate native target padding into the owned zero-KV padding ABI.

    At full native request capacity, Ascend appends a positive-KV dummy row.
    KV positivity therefore cannot identify real requests. The runner's live
    count and unpadded token frontier identify the boundary; never mutate the
    shared metadata or its device-authoritative length tensor.
    """
    ends = list(metadata.actual_seq_lengths_q)
    lengths = list(metadata.seq_lens_list)
    if (type(live_requests) is not int or not 0 < live_requests <= len(ends)
            or len(ends) != len(lengths)
            or ends[-1] != tokens
            or ends[live_requests - 1] != metadata.num_actual_tokens
            or any(b <= a for a, b in zip([0] + ends, ends))
            or any(n <= 0 for n in lengths[:live_requests])):
        raise ValueError("Target FIA live rows/frontier do not match runner metadata")
    if len(ends) == live_requests:
        return metadata
    # Native padding may be KV0 or KV1; it is not another execution seat.
    if any(n not in (0, 1) for n in lengths[live_requests:]):
        raise ValueError("Target FIA padding contains a non-padding KV length")
    result = copy.copy(metadata)
    result.actual_seq_lengths_q = ends[:live_requests] + [ends[-1]]
    result.seq_lens_list = lengths[:live_requests] + [0]
    return result


def prepare(frame, metadata):
    if not getattr(frame, 'context_parallel', False):
        return
    ends = list(metadata.actual_seq_lengths_q)
    lengths = list(metadata.seq_lens_list)
    queries = [hi-lo for lo,hi in zip([0]+ends,ends)]
    live = len(lengths) - int(lengths[-1] == 0)
    if any(n == 0 for n in lengths[:live]):
        raise ValueError('Only one trailing zero-KV padding row is qualified')
    plan = schedule(lengths[:live], queries[:live])
    raw = ctypes.string_at(frame.h_tiling.data_ptr(), 2528)
    encoded = encode(raw, plan)
    ctypes.memmove(frame.h_tiling.data_ptr(), encoded, len(encoded))
    # The base native workspace remains unchanged for Planner's capture check.
    frame.cp_workspace = (16 << 20) + struct.unpack_from('<Q',encoded,88)[0]
    if not 0 < frame.cp_workspace <= 128 << 20:
        raise ValueError('Context-parallel scratch exceeds the qualified bound')
    # Aggregate only, no per-wave file writes in the serving hot path.
    count = getattr(frame, 'cp_statistics', [0, 0, 0])
    count[0] += 1; count[1] += int(plan['split_nodes'] > 0)
    count[2] += plan['split_nodes']
    if plan['split_nodes'] and not getattr(frame, 'cp_reported_split', False):
        logging.getLogger(__name__).warning('Context-parallel first split: capacity=%s live=%s split_nodes=%s groups=%s', frame.tokens, live, plan['split_nodes'], len(plan['groups']))
        frame.cp_reported_split = True
    frame.cp_statistics = count


def launch(frame, query, key, value, mask, output):
    import torch
    global _LIBRARY
    if _LIBRARY is None:
        _LIBRARY = ctypes.CDLL(os.environ['BETTERSCALE_CP_LIBRARY'])
        _LIBRARY.lane_launch.argtypes = [ctypes.c_void_p,ctypes.POINTER(ctypes.c_uint64)]
        _LIBRARY.lane_launch.restype = None
    # Fixed capacity: length-dependent partial count never changes graph storage.
    # Native geometry planning may change scratch offsets across FD variants.
    # encode admits at most128MiB; keep graph allocation fixed at that bound.
    scratch = torch.empty(128 << 20, dtype=torch.uint8, device=query.device)
    if frame.tokens > 48:
        # MTP's large graph capacities may have thousands of zero-KV padding
        # rows; the Q1..3 kernel initializes only its small query tile.
        output.zero_()
    ptrs = (ctypes.c_uint64*10)(*[t.data_ptr() for t in
        (query,key,value,mask,frame.table,output,frame.q,frame.kv,scratch,frame.tiling)])
    _LIBRARY.lane_launch(torch.npu.current_stream().npu_stream,ptrs)
    return output
