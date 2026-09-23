"""CPU-only checks of a staged sixteen-row publication/padding envelope."""
import argparse
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np


def function(path, name, namespace):
    node = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


def check(root):
    import copy
    spec = importlib.util.spec_from_file_location('host_metadata_capacity', root/'host_metadata.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    chunks = function(root/'package/betterscale/patches/qwen_gdn/metadata.py', 'chunk_rows', {})
    compact = function(root/'draft_fia.py', 'compact_padding', {'copy': copy})
    for count in (1, 8, 9, 16):
        for decode in (False, True):
            h = object.__new__(module.HostMetadata)
            h.capacity, h.width, h.decode = 4096, 3, decode
            z = lambda *shape: np.zeros(shape, dtype=np.int64)
            h.h = NS(cu=z(18), prefill_conv=z(17,1), verify_conv=z(17,1),
                     initial=z(17), accepted=z(17), prefill_map=z(4096),
                     verify_map=z(48), restore=z(4096), verify_ids=z(17))
            h.p = NS(cu=z(18), state=z(17,2))
            h.v = NS(cu=z(18), slots=z(17,3), accepted=z(17))
            h.indices = {size:z((4096+size-1)//size+15, 2) for size in (64,256,1216)}
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
                    assert h.p.cu[16] == h.p.cu[17]
            m = NS(actual_seq_lengths_q=list(range(1,86)), seq_lens_list=[7]*count+[0]*(85-count),
                   seq_lens=list(range(85)), _mtp_device_seq_lens=list(range(85)))
            result = compact(m)
            assert len(result.actual_seq_lengths_q) == 17
            assert result.actual_seq_lengths_q[-1] == 85
            assert result.seq_lens_list[:count] == [7]*count
            assert len(m.actual_seq_lengths_q) == 85
    m.seq_lens_list[16] = 7
    try:
        compact(m)
    except ValueError:
        pass
    else:
        raise AssertionError('compacted a live seventeenth request')
    print('PASS: C1/C8/C9/C16 publication, shrinking reuse, sentinel and draft compaction')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capsule', type=Path)
    check(parser.parse_args().capsule)
