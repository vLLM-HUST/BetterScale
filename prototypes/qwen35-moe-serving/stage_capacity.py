"""Stage a separate, unqualified sixteen-request MoE capsule.

Exact substitutions preserve head geometry, ABI pointer counts and alignment.
The host adapter must be rebuilt; CPU checks do not qualify the new envelope.
"""
import argparse
import difflib
import json
from pathlib import Path
import shutil


def stage(seed, output):
    assert not output.exists()
    shutil.copytree(seed, output, ignore=shutil.ignore_patterns('__pycache__', '*.log'))
    changes = {
        'count_policy.py': [('BINS = (3, 6, 12, 24, 40)', 'BINS = (3, 6, 12, 24, 40, 48, 80)'),
                            ('8*WIDTH', '16*WIDTH'), ('len(lengths) <= 8', 'len(lengths) <= 16')],
        'service_adapter.py': [('max_num_seqs == 8', 'max_num_seqs == 16')],
        'service_metadata.py': [('torch.zeros(9,', 'torch.zeros(17,'), ('n <= 8', 'n <= 16')],
        'device_metadata.py': [('torch.zeros(9,', 'torch.zeros(17,')],
        'device_slots.py': [('meta.live <= 8', 'meta.live <= 16')],
        'host_metadata.py': [('n <= 8', 'n <= 16'), ('= 8  # permanent empty sentinel', '= 16  # permanent empty sentinel')],
        'mixed_core.py': [('torch.zeros(10,', 'torch.zeros(18,'), ('torch.full((9,', 'torch.full((17,'),
                          ('torch.zeros(9,', 'torch.zeros(17,'), ('torch.ones(9,', 'torch.ones(17,'),
                          ('torch.zeros(8*WIDTH,', 'torch.zeros(16*WIDTH,'),
                          ('len(lengths) <= 8', 'len(lengths) <= 16'), ('(10-len(', '(18-len(')],
        'draft_fia.py': [('only8 real requests', 'only16 real requests'),
                         ('<= 9:', '<= 17:'), ('[8:]', '[16:]'), ('[:8]', '[:16]'),
                         ('live ninth request', 'live seventeenth request')],
        'package/betterscale/patches/qwen_gdn/metadata.py': [
            ('requests=8', 'requests=16'), ('tokens if decode else 9', 'tokens if decode else 17'),
            ('// size + 7, 2', '// size + 15, 2'), ('tokens <= 8', 'tokens <= 16')],
        'package/betterscale/patches/qwen_gdn/graphs.py': [('num_reqs=8', 'num_reqs=16'), ('tokens > 8', 'tokens > 16')],
        'package/betterscale/patches/qwen_gdn/host.cpp': [('requests <= 9', 'requests <= 17'), ('requests == 9', 'requests == 17')],
        'package/betterscale/patches/qwen_fia/wave.py': [
            ('len(lengths) <= 9', 'len(lengths) <= 17'),
            ('2528 + 9 * 16 + 9 * columns * 4', '2528 + 17 * 16 + 17 * columns * 4'),
            ('t[2528:2600]', 't[2528:2664]'), ('t[2600:2672]', 't[2664:2800]'),
            ('t[2672:].view(torch.int32).view(9, columns)', 't[2800:].view(torch.int32).view(17, columns)')],
    }
    diff = []
    for name, replacements in changes.items():
        path = output / name
        old = new = path.read_text()
        for before, after in replacements:
            assert before in new, (name, before)
            new = new.replace(before, after)
        compile(new, name, 'exec') if name.endswith('.py') else None
        path.write_text(new)
        diff.extend(difflib.unified_diff(old.splitlines(True), new.splitlines(True),
                                        fromfile='capacity8/'+name, tofile='capacity16/'+name))
    from stage_draft_sampling import install_in_capsule
    install_in_capsule(output)
    # A stale host binary must never accidentally execute against enlarged PODs.
    (output/'native/libbs_gdn_host.so').unlink()
    (output/'capacity.diff').write_text(''.join(diff))
    (output/'capacity-contract.json').write_text(json.dumps({
        'status': 'UNQUALIFIED', 'max_requests': 16, 'sentinel_row': 16,
        'query_capacity': 4096, 'required_gates': [
            'rebuild host adapter and update its pin', 'CPU metadata and padding checks',
            'GDN state oracle with sixteen rows', 'FIA target/draft sixteen-row oracle',
            'real serving and FULL capture at new capacities'],
        'source': str(seed)}, indent=2)+'\n')


def stage32(seed, output):
    """Enlarge request axes only; leave head counts and byte strides untouched."""
    if output.exists():
        raise FileExistsError(output)
    gdn = 'package/betterscale/patches/qwen_gdn/'
    changes = {
        'count_policy.py': [('48, 80)', '48, 80, 96, 160)'),
                            ('16*WIDTH', '32*WIDTH'), ('len(lengths) <= 16', 'len(lengths) <= 32')],
        'service_adapter.py': [('max_num_seqs == 16', 'max_num_seqs == 32')],
        'service_metadata.py': [('torch.zeros(17,', 'torch.zeros(33,'), ('n <= 16', 'n <= 32')],
        'device_metadata.py': [('torch.zeros(17,', 'torch.zeros(33,')],
        'device_slots.py': [('meta.live <= 16', 'meta.live <= 32'),
                            ('tl.arange(0, 16)', 'tl.arange(0, 32)')],
        'host_metadata.py': [('n <= 16', 'n <= 32'),
                             ('= 16  # permanent empty sentinel', '= 32  # permanent empty sentinel')],
        'mixed_core.py': [('torch.zeros(18,', 'torch.zeros(34,'), ('torch.full((17,', 'torch.full((33,'),
                          ('torch.zeros(17,', 'torch.zeros(33,'), ('torch.ones(17,', 'torch.ones(33,'),
                          ('torch.zeros(16*WIDTH,', 'torch.zeros(32*WIDTH,'),
                          ('len(lengths) <= 16', 'len(lengths) <= 32'), ('(18-len(', '(34-len(')],
        'draft_fia.py': [('only16 real requests', 'only32 real requests'),
                         ('<= 17:', '<= 33:'), ('[16:]', '[32:]'), ('[:16]', '[:32]'),
                         ('live seventeenth request', 'live thirty-third request')],
        gdn+'metadata.py': [('requests=16', 'requests=32'), ('tokens if decode else 17', 'tokens if decode else 33'),
                            ('// size + 15, 2', '// size + 31, 2'), ('tokens <= 16', 'tokens <= 32')],
        gdn+'graphs.py': [('num_reqs=16', 'num_reqs=min(32, result.num_tokens)'), ('tokens > 16', 'tokens > 32')],
        gdn+'host.cpp': [('requests <= 17', 'requests <= 33'), ('requests == 17', 'requests == 33')],
        'package/betterscale/patches/qwen_fia/wave.py': [
            ('len(lengths) <= 17', 'len(lengths) <= 33'),
            ('2528 + 17 * 16 + 17 * columns * 4', '2528 + 33 * 16 + 33 * columns * 4'),
            ('t[2528:2664]', 't[2528:2792]'), ('t[2664:2800]', 't[2792:3056]'),
            ('t[2800:].view(torch.int32).view(17, columns)', 't[3056:].view(torch.int32).view(33, columns)')],
        'capacity_state_probe.py': [
            ('capacity = 48 if pure', 'capacity = 96 if pure'),
            ('16*WIDTH', '32*WIDTH'), ('reshape(16,WIDTH)', 'reshape(32,WIDTH)'),
            ('([3]*16,[True]*16,[1,2,3,1]*4)', '([3]*32,[True]*32,[1,2,3,1]*8)'),
            ('([1,2,3,1]*4,[True]*16,[3,1,2,3]*4)', '([1,2,3,1]*8,[True]*32,[3,1,2,3]*8)'),
            ('([256]*16,[False]*16,[1]*16)', '([128]*32,[False]*32,[1]*32)'),
            ('([3]*15+[4051],[True]*15+[False],[1,2,3]*5+[1])',
             '([3]*31+[4003],[True]*31+[False],[1,2,3]*10+[1,2])'),
            ('([2049]+[3]*14+[2005],[False]+[True]*14+[False],[1]*16)',
             '([2049]+[3]*30+[1957],[False]+[True]*30+[False],[1]*32)'),
            ('([3,17]*8,[True,False]*8,[2,1]*8)', '([3,17]*16,[True,False]*16,[2,1]*16)'),
            ('[True]*16)', '[True]*32)')],
        'capacity_fia_probe.py': [
            ('[1]*16+[capacity-16], [4096]*16+[0], [1]*16+[0]',
             '[1]*32+[capacity-32], [4096]*32+[0], [1]*32+[0]'),
            ('(48,2048,4096)', '(96,2048,4096)'),
            ('[3]*16', '[3]*32'), ('[4096]*16', '[4096]*32'), ('[1]*16', '[1]*32'),
            ('[262144]*16', '[262144]*32'), ('valid = 16 if draft_padding', 'valid = 32 if draft_padding'),
            ('[:16]', '[:32]'), ('(width-16)', '(width-32)')],
    }
    edits, diff = {}, []
    for name, substitutions in changes.items():
        old = new = (seed/name).read_text()
        for before, after in substitutions:
            if before not in new:
                raise ValueError((name, before))
            new = new.replace(before, after)
        if name.endswith('.py'):
            compile(new, name, 'exec')
        edits[name] = new
        diff.extend(difflib.unified_diff(old.splitlines(True), new.splitlines(True),
                                        fromfile='capacity16/'+name, tofile='capacity32/'+name))
    shutil.copytree(seed, output, ignore=shutil.ignore_patterns('__pycache__', '*.log'))
    for name, text in edits.items():
        (output/name).write_text(text)
    (output/'native/libbs_gdn_host.so').unlink()
    (output/'capacity32.diff').write_text(''.join(diff))
    (output/'capacity-contract.json').write_text(json.dumps({
        'status':'UNQUALIFIED', 'max_requests':32, 'sentinel_row':32,
        'query_capacity':4096, 'source':str(seed),
        'required_gates':['rebuild and repin host adapter', '32-row GDN and FIA oracles',
                          'C32 real MTP and context qualification', 'SWE engine load and KV admission'],
    }, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('seed', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--requests', type=int, choices=(16,32), default=16)
    args = parser.parse_args()
    (stage32 if args.requests == 32 else stage)(args.seed, args.output)
