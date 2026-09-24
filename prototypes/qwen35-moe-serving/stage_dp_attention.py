"""Port the C16 MoE capsule's attention geometry from TP2 to TP1/DP2.

Expert communication stays native. Request counts, byte alignment, workspace
reservations and query envelopes are NOT head counts. All substitutions name
the affected expression rather than globally replacing numbers.
"""
import argparse
import difflib
import json
from pathlib import Path
import re
import shutil


def replace(source, changes):
    for old in changes:
        if old not in source:
            raise ValueError(f'Missing source expression: {old}')
    # Simultaneous substitution: new16 must not be replaced again as old16.
    pattern = '|'.join(re.escape(key) for key in sorted(changes, key=len, reverse=True))
    return re.sub(pattern, lambda match: changes[match.group()], source)


def stage(seed, output, ep):
    if output.exists():
        raise FileExistsError(output)
    gdn = 'package/betterscale/patches/qwen_gdn/'
    changes = {
        'service_adapter.py': {
            'parallel.enable_expert_parallel) == (2,1,1,False)':
                f'parallel.enable_expert_parallel) == (1,2,1,{ep})'},
        gdn+'preprocess.py': {
            'tl.arange(0, 8)': 'tl.arange(0, 16)',
            'off + 1024': 'off + 2048',
            '* 1024 + head': '* 2048 + head',
            'vd < 2048': 'vd < 4096',
            'XS + 2048 + vd': 'XS + 4096 + vd',
            '* 2048 + vd': '* 4096 + vd',
            'gh < 16': 'gh < 32', '* 16 + gh': '* 32 + gh',
            '(source_tokens, 4096)': '(source_tokens, 8192)',
            '(source_tokens, 16)': '(source_tokens, 32)',
            '(16,)': '(32,)', '(1, 8, t, 128)': '(1, 16, t, 128)',
            '(1, t, 8, 128)': '(1, t, 16, 128)',
            '(1, t, 16, 128)': '(1, t, 32, 128)',
            '(1, t, 16)': '(1, t, 32)'},
        gdn+'runtime.py': {
            '(1, 16, chunks, 128, 128)': '(1, 32, chunks, 128, 128)',
            '(1, 16, tokens, 128)': '(1, 32, tokens, 128)',
            '(requests, 16, 128, 128)': '(requests, 32, 128, 128)',
            't.kNumHead = 8': 't.kNumHead = 16', 't.vNumHead = 16': 't.vNumHead = 32',
            '(16, 128, 128)': '(32, 128, 128)',
            '(1, 8, self.T, 128)': '(1, 16, self.T, 128)',
            '(1, 16, self.T, 128)': '(1, 32, self.T, 128)',
            '(1, 16, self.T)': '(1, 32, self.T)',
            '(self.N, 16, 128, 128)': '(self.N, 32, 128, 128)'},
        gdn+'host.cpp': {
            '{1, 16, chunks, 128, 128}': '{1, 32, chunks, 128, 128}',
            '{1, 16, tokens, 128}': '{1, 32, tokens, 128}'},
        gdn+'decode_kv.py': {
            'H == 8 and K == V == 128 and v.shape[2] == 16':
                'H == 16 and K == V == 128 and v.shape[2] == 32'},
        'mixed_core.py': {
            'd < 2048': 'd < 4096', '* 2048 + d': '* 4096 + d',
            '(1, self.capacity, 16, 128)': '(1, self.capacity, 32, 128)',
            'triton.cdiv(self.capacity, 16), 16)': 'triton.cdiv(self.capacity, 16), 32)',
            'restore_kernel[(self.capacity, 3)]': 'restore_kernel[(self.capacity, 4)]'},
        'mixed_layout.py': {
            'head[None, :] < 16': 'head[None, :] < 32',
            '* 16 + head[None, :]': '* 32 + head[None, :]',
            '(1, 16, t)': '(1, 32, t)', '* 2048 + head': '* 4096 + head'},
        'draft_fia.py': {'(8,1,256)': '(16,2,256)', '(4096,8,256)': '(4096,16,256)'},
        'capacity_state_probe.py': {
            '16*WIDTH+1, 16, 128, 128': '16*WIDTH+1, 32, 128, 128',
            'WIDTH+2, 4096': 'WIDTH+2, 8192', '(4, 4096)': '(4, 8192)',
            'torch.randn(16)': 'torch.randn(32)',
            '(capacity, 4096)': '(capacity, 8192)',
            '(capacity, 16)': '(capacity, 32)',
            'capacity, 4096, dtype': 'capacity, 8192, dtype',
            'capacity, 16, dtype': 'capacity, 32, dtype',
            'torch.zeros(3, 4096)': 'torch.zeros(3, 8192)',
            'torch.zeros(16, 128, 128)': 'torch.zeros(32, 128, 128)',
            'y.split([1024,1024,2048]), [8,8,16]':
                'y.split([2048,2048,4096]), [16,16,32]'},
        'capacity_fia_probe.py': {
            'pages,128,256,device': 'pages,128,512,device',
            'width,8,256,device': 'width,16,256,device',
            'ptrs,rows,width,8,1,pages': 'ptrs,rows,width,16,2,pages',
            'num_heads=8,num_key_value_heads=1': 'num_heads=16,num_key_value_heads=2'},
        'package/betterscale/patches/qwen_fia/wave.py': {
            '            8,\n            1,': '            16,\n            2,',
            '(8, 1, 256)': '(16, 2, 256)', '(4096, 8, 256)': '(4096, 16, 256)'},
    }
    for name in [gdn+'chunk_wy.py', 'chunk_layout.py']:
        changes[name] = {'key_heads == 8': 'key_heads == 16',
                         '(1, tokens, 16, 128)': '(1, tokens, 32, 128)',
                         'heads, block = 16, 64': 'heads, block = 32, 64'}
    edits = {}
    diff = []
    for name, substitutions in changes.items():
        old = (seed/name).read_text()
        new = replace(old, substitutions)
        if name.endswith('.py'):
            compile(new, name, 'exec')
        edits[name] = new
        diff.extend(difflib.unified_diff(old.splitlines(True), new.splitlines(True),
                                       fromfile='attention-tp2/'+name,
                                       tofile='attention-dp2/'+name))
    shutil.copytree(seed, output, ignore=shutil.ignore_patterns('__pycache__', '*.log'))
    for name, content in edits.items():
        (output/name).write_text(content)
    (output/'native/libbs_gdn_host.so').unlink()
    (output/'attention-geometry.diff').write_text(''.join(diff))
    (output/'parallel-contract.json').write_text(json.dumps({
        'status': 'UNQUALIFIED', 'attention_tp': 1, 'attention_dp': 2,
        'expert_partition': 'EP2' if ep else 'TP2',
        'source': str(seed), 'gdn_heads': [16, 32], 'fia_heads': [16, 2],
        'max_requests_per_rank': 16, 'query_capacity_per_rank': 4096,
        'required_gates': ['rebuild and repin host adapter',
                           'GDN state oracle and FIA target/draft oracle',
                           'DP synchronized FULL prefill/mixed/decode',
                           'real MTP and rank-affine cold/warm/near256K retrieval'],
    }, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('seed', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--expert-parallel', action='store_true')
    args = parser.parse_args()
    stage(args.seed, args.output, args.expert_parallel)
