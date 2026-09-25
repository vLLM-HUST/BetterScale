"""CPU-only layout fixture; execute installed connector statements, not NPU imports."""
import argparse
import ast
import json
import os
from pathlib import Path

os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    source = args.runtime / 'vllm_ascend/simple_kv_offload/worker.py'
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name == 'SimpleCPUOffloadNPUWorker')
    build = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                 and n.name == '_build_block_views')
    build.decorator_list = []
    flatten = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                   and n.name == '_flatten_kv_value')
    namespace = {'torch': torch}
    exec(compile(ast.Module(body=[flatten, build], type_ignores=[]), str(source), 'exec'), namespace)
    register = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                    and n.name == 'register_kv_caches')
    start = next(i for i, n in enumerate(register.body) if isinstance(n, ast.AnnAssign)
                 and isinstance(n.target, ast.Name) and n.target.id == 'unique_caches')
    end = next(i for i, n in enumerate(register.body) if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == 'per_tensor_bpb' for t in n.targets))
    registration = compile(ast.Module(body=register.body[start:end], type_ignores=[]), str(source), 'exec')

    class Receiver:
        _build_block_views = staticmethod(namespace['_build_block_views'])

    # One shared slot, real TP2/MTP2 byte geometry; synthetic N and alignment guard.
    n, c, s, guard = 4, 40960, 1048576, 256
    backing = torch.full((guard + n * (c + 2 * s) + guard,), 91, dtype=torch.uint8)
    raw = backing[guard:-guard]
    conv = raw[:n*c].view(torch.bfloat16).view(n, 4096, 5)
    ssm = raw[n*c:n*(c+s)].view(torch.float32).view(n, 16, 128, 128)
    k = raw[n*c:n*(c+s)].view(torch.bfloat16).view(n*16, 128, 1, 256)
    v = raw[n*(c+s):].view(torch.bfloat16).view(n*16, 128, 1, 256)
    assert ssm.data_ptr() == k.data_ptr()
    assert len({t.untyped_storage().data_ptr() for t in (conv, ssm, k, v)}) == 1
    results = {}
    for order, caches in (
        ('gdn_first', {'g0': [conv, ssm], 'g1': [conv, ssm], 'g2': [conv, ssm], 'fa': (k, v)}),
        ('fa_first', {'fa': (k, v), 'g0': [conv, ssm], 'g1': [conv, ssm], 'g2': [conv, ssm]}),
    ):
        namespace.update(self=Receiver(), num_blocks=n, kv_caches=caches)
        exec(registration, namespace)
        views = namespace['unique_caches']
        results[order] = {'keys': list(views), 'registered_bytes': sum(t.numel() for t in views.values())}
    # Assert the recorded failure signature. A future corrected implementation
    # should fail these assertions, prompting requalification rather than stale success.
    assert results['gdn_first']['registered_bytes'] == n*c
    assert results['fa_first']['registered_bytes'] == n*s//16

    # Independent address oracle: logical block b is three disjoint slices, NOT
    # raw[b*P:(b+1)*P]. Exercise changed destination ID and untouched neighbors.
    spans = [(0, c), (n*c, s), (n*(c+s), s)]
    src, dst = 1, 3
    raw.zero_()
    for marker, (base, width) in enumerate(spans, 11):
        raw[base+src*width:base+(src+1)*width].fill_(marker)
    snapshot = [raw[base+src*width:base+(src+1)*width].clone() for base, width in spans]
    before = raw.clone()
    for payload, (base, width) in zip(snapshot, spans):
        raw[base+dst*width:base+(dst+1)*width].copy_(payload)
    for marker, (base, width) in enumerate(spans, 11):
        for b in range(n):
            actual = raw[base+b*width:base+(b+1)*width]
            expected = torch.full_like(actual, marker) if b == dst else before[base+b*width:base+(b+1)*width]
            assert torch.equal(actual, expected)
    assert torch.all(backing[:guard] == 91) and torch.all(backing[-guard:] == 91)
    result = {
        'scope': 'CPU synthetic shared-slot fixture; exact installed registration AST; no NPU or DMA qualification',
        'source': str(source), 'logical_blocks': n, 'kernel_blocks_per_logical_block': 16,
        'conv_bytes_per_block': c, 'ssm_or_k_bytes_per_block': s,
        'raw_bytes': raw.numel(), 'relative_slab_offsets': [x[0] for x in spans],
        'ssm_k_exact_byte_alias': True, 'registration': results,
        'independent_three_region_copy': 'PASS: block 1 -> 3, neighbors and alignment guards intact',
    }
    rendered = json.dumps(result, indent=2) + '\n'
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end='')


if __name__ == '__main__':
    main()
