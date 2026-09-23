"""Stage an isolated geometry-only pilot from the frozen September22 MTP source.

Not a product installer. Only listed head/stride sites change; hardware cores,
request capacities, bank/state ownership and MTP width remain untouched.
"""
import argparse
import difflib
import json
import re
from pathlib import Path
import shutil

def replace_exact(source, before, after):
    if before == '24':
        changed, count = re.subn(r'\b24\b', after, source)
        if not count:
            raise ValueError('missing standalone head-count literal24')
        return changed
    if before not in source:
        raise ValueError(f'missing exact source pattern: {before}')
    return source.replace(before, after)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('seed',type=Path)
    p.add_argument('output',type=Path)
    a=p.parse_args()
    assert not a.output.exists()
    shutil.copytree(a.seed,a.output,ignore=shutil.ignore_patterns('__pycache__'))
    gdn='package/betterscale/patches/qwen_gdn/'
    changes={
     gdn+'decode_kv.py': [('v.shape[2] == 24','v.shape[2] == 16')],
     gdn+'preprocess.py': [('5120','4096'),('3072','2048'),('24','16')],
     gdn+'runtime.py': [('(1, 24,','(1, 16,'),('(requests, 24,','(requests, 16,'),
                           ('t.vNumHead = 24','t.vNumHead = 16'),('(24, 128, 128)','(16, 128, 128)')],
     gdn+'host.cpp': [('{1, 24,','{1, 16,')],
     gdn+'chunk_wy.py': [('tokens, 24,','tokens, 16,'),('heads, block = 24,','heads, block = 16,')],
     'chunk_layout.py': [('tokens, 24,','tokens, 16,'),('heads, block = 24,','heads, block = 16,')],
     'mixed_layout.py': [('24','16'),('3072','2048')],
     'mixed_core.py': [('self.capacity, 24,','self.capacity, 16,'),('16), 24)]','16), 16)]'),
                       ('3072','2048')],
     'mixed_state_probe.py': [('5120','4096'),('3072','2048'),(', 24',', 16'),
                              ('randn(24)','randn(16)'),('zeros(24,','zeros(16,'),
                              ('[8,8,24]','[8,8,16]'),('repeat_interleave(3,0)','repeat_interleave(2,0)')],
    }
    # Remaining runtime shape contracts are value-head dimensions, not cores.
    changes[gdn+'runtime.py'] += [('(self.N, 24,','(self.N, 16,')]
    diffs=[]
    for name,replacements in changes.items():
     path=a.output/name;old=path.read_text();new=old
     for before,after in replacements:
      new=replace_exact(new,before,after)
     path.write_text(new)
     diffs.extend(difflib.unified_diff(old.splitlines(True),new.splitlines(True),
                                     fromfile='seed/'+name,tofile='moe/'+name))
    (a.output/'geometry.diff').write_text(''.join(diffs))
    (a.output/'geometry-contract.json').write_text(json.dumps({
     'status':'UNQUALIFIED','qk_heads':8,'value_heads':16,'channels':4096,
     'head_dimension':128,'mtp_tokens':2,'hardware_cores':24,'requests':8,
     'gdn_kernel':'unchanged owned-init H/O; new tiling heads; rebuilt host allocation',
     'oracles':'independent CPU recurrence, every candidate state and exact convolution history',
     'not_claimed':['FIA','MoE model forward','256K service','performance']},indent=2))
    print(a.output)


if __name__ == '__main__':
    main()
