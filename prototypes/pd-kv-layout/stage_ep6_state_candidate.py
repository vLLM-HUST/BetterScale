"""Isolated EP6 State experiment; never relax the released TP2/DP1 admission."""
import argparse,json,shutil
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
assert not a.output.exists()
changes={
 'betterscale/models/qwen35/__init__.py': [('== (2, 1, 1, False, 1, 1)', '== (2, 3, 1, True, 1, 1)'),
   ('TP2/DP1/PP1 without EP or context parallelism','EXPERIMENTAL TP2/DP3/EP6/PP1 without context parallelism')],
 'betterscale/models/qwen35/small_fish_runtime.py':[('!= (2, 1, 1, False)','!= (2, 3, 1, True)')]}
assert (a.source/'target_only_diagnostic.py').is_file()
assert '_install_target_only_worker()' in (a.source/'betterscale/models/qwen35/state_address.py').read_text()
patched={}
for path,replacements in changes.items():
 source=(a.source/path).read_text()
 for old,new in replacements:
  assert source.count(old)==1,(path,old)
  source=source.replace(old,new)
 patched[path]=source
shutil.copytree(a.source,a.output,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
for path,source in patched.items():(a.output/path).write_text(source)
(a.output/'ep6-state-experiment.json').write_text(json.dumps(dict(source=str(a.source),changed_files=list(changes),
 scope='UNQUALIFIED full State DP3/TP2/EP6 target-only; draft startup retained; release unchanged'),indent=2)+'\n')
