"""Create an immutable old/new serving pair; never rewrite a shared capsule."""
import argparse,json,shutil
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True)
p.add_argument('--out',type=Path,required=True);a=p.parse_args()
a.out.mkdir(parents=True,exist_ok=False)
shutil.copytree(a.base,a.out/'baseline',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
shutil.copytree(a.out/'baseline',a.out/'candidate')
candidate=a.out/'candidate'
p=candidate/'mixed_core.py';text=p.read_text()
anchor='        # Fuse activation routing into preprocessing; recurrent state never moves.\n'
assert text.count(anchor)==1
text=text.replace(anchor,'        return self.after_conv(transformed, a, b, log, bias, state)\n\n'
                        '    def after_conv(self, transformed, a, b, log, bias, state):\n'
                        '        x = transformed\n'+anchor)
p.write_text(text)
worker=candidate/'package/betterscale/worker.py'
text=worker.read_text();anchor='install_mtp_probe()\n'
assert text.count(anchor)==1
worker.write_text(text.replace(anchor,anchor+'from service_fusion import install as install_qkv_conv\ninstall_qkv_conv()\n'))
for name in ('pool_adapter.py','service_fusion.py'):
 shutil.copy2(Path(__file__).parent/name,candidate/name)
(a.out/'source.json').write_text(json.dumps({'base':str(a.base.resolve()),
 'baseline':'byte copies excluding bytecode','candidate_changes':[
 'MixedCore.after_conv is extracted verbatim; original call remains valid',
 'Worker installs shape-qualified fusion by default after original MTP install',
 'two task-owned integration modules; no donor or model-weight edits'],
 'status':'STAGED, not NPU-qualified'},indent=2)+'\n')
