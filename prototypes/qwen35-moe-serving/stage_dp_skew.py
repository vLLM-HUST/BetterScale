"""Allow a locally pure-verification rank inside a DP-promoted mixed graph.

Another rank's prefill can select the common large FULL envelope. Empty local
prefill uses existing empty sentinel chunks; restore never reads its unwritten
output, including padded model rows. Numerical kernels are unchanged.
"""
import argparse
import difflib
import json
from pathlib import Path
import shutil


def stage(seed, output):
    if output.exists():
        raise FileExistsError(output)
    name = 'host_metadata.py'
    old = (seed/name).read_text()
    guard = "        assert len(pre), 'pure verification belongs to its small graph, not mixed'\n"
    assert old.count(guard) == 1 and old.count('h.restore.fill(0)') == 1
    new = old.replace(guard, '').replace('h.restore.fill(0)',
        'h.restore.fill(0 if len(pre) else self.capacity)')
    shutil.copytree(seed, output, ignore=shutil.ignore_patterns('__pycache__', '*.log'))
    (output/name).write_text(new)
    (output/'dp-skew.diff').write_text(''.join(difflib.unified_diff(
        old.splitlines(True), new.splitlines(True), fromfile='local-only/'+name,
        tofile='dp-promoted/'+name)))
    p = output/'capacity_state_probe.py'
    s = p.read_text()
    s = s.replace("    pure = os.environ.get('PURE_VERIFY') == '1'",
                  "    pure = os.environ.get('PURE_VERIFY') == '1'\n"
                  "    promoted = os.environ.get('PROMOTED_VERIFY') == '1'")
    assert s.count('    if pure:\n') == 1
    s = s.replace('    if pure:\n', '    if pure or promoted:\n')
    s = s.replace('            actual = results[wave % 2].cpu()[0]',
                  '            actual = results[wave % 2].cpu()[0]\n'
                  '            if promoted:\n'
                  '                assert torch.isfinite(actual).all(), "Nonfinite padded model rows"')
    compile(s, str(p), 'exec')
    p.write_text(s)
    (output/'dp-skew-contract.json').write_text(json.dumps({
        'status':'UNQUALIFIED', 'source':str(seed),
        'required_gates':['pure verification in a4096-token published mixed graph',
                          'whole-pool state/conv comparison and finite padded output',
                          'rank-skewed real DP MTP serving'],
    }, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('seed', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    stage(args.seed, args.output)
