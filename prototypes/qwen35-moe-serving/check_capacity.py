"""CPU-only checks of a staged request-row publication/padding envelope."""
import argparse
import ast
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np


def function(path, name, namespace):
    node = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


def host_fixture(module, requests, decode):
    slot_rows, offsets = requests+1, requests+2
    h = object.__new__(module.HostMetadata)
    h.capacity, h.width, h.decode = 4096, 3, decode
    z = lambda *shape: np.zeros(shape, dtype=np.int64)
    h.h = NS(cu=z(offsets), prefill_conv=z(slot_rows,1), verify_conv=z(slot_rows,1),
             initial=z(slot_rows), accepted=z(slot_rows), prefill_map=z(4096),
             verify_map=z(requests*3), restore=z(4096), verify_ids=z(slot_rows))
    h.p = NS(cu=z(offsets), state=z(slot_rows,2))
    h.v = NS(cu=z(offsets), slots=z(slot_rows,3), accepted=z(slot_rows))
    h.indices = {size:z((4096+size-1)//size+requests-1, 2) for size in (64,256,1216)}
    return h


def check(root, dp_skew=False):
    import copy
    import dataclasses
    requests = json.loads((root / "capacity-contract.json").read_text())["max_requests"]
    assert requests in (16, 32)
    slot_rows, offsets = requests + 1, requests + 2
    spec = importlib.util.spec_from_file_location('host_metadata_capacity', root/'host_metadata.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    chunks = function(root/'package/betterscale/patches/qwen_gdn/metadata.py', 'chunk_rows', {})
    compact = function(root/'draft_fia.py', 'compact_padding', {'copy': copy})
    descriptor = function(root/'package/betterscale/patches/qwen_gdn/graphs.py',
                          'descriptor', {'dataclasses': dataclasses,
                                         'PREFILLS': (16, 32, 64, 4096)})
    batch = dataclasses.make_dataclass('Batch', ['num_tokens', 'num_reqs'])
    for tokens in (16, 32, 64, 4096):
        padded = descriptor(batch(tokens, 1))
        assert padded.num_reqs == min(requests, tokens)
    for count in sorted({1, 8, 9, 16, requests-1, requests}):
        for decode in (False, True):
            h = host_fixture(module, requests, decode)
            # Shrinking and reversing membership catches stale row/sentinel reuse.
            for n in (count, 1, count):
                roles = [decode or i % 2 == 1 for i in range(n)]
                lengths = [1+i%3 if role else 65+i for i,role in enumerate(roles)]
                slots = np.arange(n*3).reshape(n,3)[:,::-1].copy()
                h.prepare(lengths, roles, slots, [True]*n)
                assert list(h.h.cu[:n+1]) == [0]+list(np.cumsum(lengths))
                assert all(h.h.cu[n+1:] == sum(lengths))
                assert all(h.h.verify_conv[n:,0] == -1)
                ids = np.flatnonzero(roles)
                assert np.array_equal(h.v.slots[:len(ids)],slots[ids])
                assert np.all(h.v.slots[len(ids):] == -1)
                if not decode:
                    pre = [x for x,role in zip(lengths,roles) if not role]
                    for size, rows in h.indices.items():
                        assert rows.tolist() == [list(row) for row in chunks(pre,size,4096)]
                    assert h.p.cu[requests] == h.p.cu[requests+1]
            m = NS(actual_seq_lengths_q=list(range(1,86)), seq_lens_list=[7]*count+[0]*(85-count),
                   seq_lens=list(range(85)), _mtp_device_seq_lens=list(range(85)))
            result = compact(m)
            assert len(result.actual_seq_lengths_q) == slot_rows
            assert result.actual_seq_lengths_q[-1] == 85
            assert result.seq_lens_list[:count] == [7]*count
            assert len(m.actual_seq_lengths_q) == 85
    m.seq_lens_list[requests] = 7
    try:
        compact(m)
    except ValueError:
        pass
    else:
        raise AssertionError('compacted a live request beyond capacity')
    if dp_skew:
        h = host_fixture(module, requests, False)
        lengths = [1+i%3 for i in range(requests)]
        slots = np.arange(requests*3).reshape(requests,3)
        h.prepare(lengths, [True]*requests, slots, [True]*requests)
        assert np.all(h.h.prefill_conv == -1) and np.all(h.p.cu == 0)
        assert np.all(h.p.state == 0)
        assert np.all(h.h.restore[:sum(lengths)] == 4096+np.arange(sum(lengths)))
        assert np.all(h.h.restore[sum(lengths):] == 4096)
        for rows in h.indices.values():
            assert np.all(rows[:,0] == requests) and np.all(rows[:,1] == 0)
    print(f'PASS: up to C{requests} publication, shrinking reuse, sentinel and draft compaction')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capsule', type=Path)
    parser.add_argument('--dp-skew', action='store_true')
    args = parser.parse_args()
    check(args.capsule, args.dp_skew)
