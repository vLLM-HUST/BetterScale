"""Freeze this experiment into a NEW capsule, with preflight before any NPU load.

Run on the execution host. Does not launch, alter installed donors or overwrite a
capsule. Seed must be the qualified banked-draft service, not an earlier fork.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f'Expected exactly one integration boundary: {old!r}')
    return text.replace(old,new)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('seed','destination','donor','prototype'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--device-lengths',action='store_true')
    args=parser.parse_args()
    seed,dest,donor,proto=(getattr(args,k).resolve() for k in ('seed','destination','donor','prototype'))
    if dest.exists() or dest.parent!=seed.parent or dest in (seed,donor,proto):
        raise ValueError('Destination must be a new capsule beside the qualified seed')
    files=['device_apc.py','device_apc_runner.patch']
    if args.device_lengths:files+=['draft_fia.py','device_metadata.py','device_slots.py']
    for path in [seed/'prompt.json',seed/'launch.sh',seed/'service_adapter.py',seed/'service_smoke.py',
                 seed/'helpers/admit_subset.py',donor/'worker/model_runner_v1.py',*[proto/f for f in files]]:
        if not path.is_file():raise FileNotFoundError(path)
    raw=json.loads((seed/'prompt.json').read_text())
    tokens=raw['prompt_token_ids'] if isinstance(raw,dict) else raw
    assert isinstance(tokens,list) and len(tokens)*10>=4096
    assert all(isinstance(t,int) and t>=0 for t in tokens)
    pins=list((seed/'package/betterscale').glob('qwen*pins.json'))
    target='vllm_ascend/worker/model_runner_v1.py'
    original=(donor/'worker/model_runner_v1.py').read_bytes()
    for pin in pins:
        for item in json.loads(pin.read_text())['source_files']:
            if item['path']==target:
                assert item['sha256']==hashlib.sha256(original).hexdigest(), 'seed/base source mismatch'
    dest.mkdir()
    for p in seed.glob('*.py'):shutil.copy2(p,dest/p.name)
    for name in ('launch.sh','prompt.json'):shutil.copy2(seed/name,dest/name)
    for name in ('package','helpers','triton','vllm-cache'):
        shutil.copytree(seed/name,dest/name)
    shutil.copytree(donor,dest/'runtime-source/vllm_ascend',ignore=shutil.ignore_patterns('__pycache__'))
    for name in files:shutil.copy2(proto/name,dest/name)
    subprocess.run(['patch','--batch','--fuzz=0','-p1','-i',str(dest/'device_apc_runner.patch')],
                   cwd=dest/'runtime-source',check=True)
    modified=(dest/'runtime-source'/target).read_bytes()
    function=next(n for n in ast.walk(ast.parse(modified)) if isinstance(n,ast.FunctionDef) and n.name=='execute_model')
    assert not any(isinstance(n,ast.Name) and n.id in ('mamba_bufs','preprocess_bufs') for n in ast.walk(function))
    for pin in (dest/'package/betterscale').glob('qwen*pins.json'):
        data=json.loads(pin.read_text())
        for item in data['source_files']:
            if item['path']==target:item['sha256']=hashlib.sha256(modified).hexdigest()
        data['experimental_source_patch']='device_apc_runner.patch; base commit retained, source explicitly differs'
        pin.write_text(json.dumps(data,indent=2)+'\n')
    p=dest/'service_adapter.py'
    text=replace_once(p.read_text(),'        qwen_gdn.install()',
        '        qwen_gdn.install()\n        from device_apc import install as install_device_apc\n        install_device_apc()')
    if args.device_lengths:
        text=replace_once(text,'    qwen.after_init = lambda worker:qwen_fia.install()',
            '    def after_init(worker):\n        qwen_fia.install()\n        from draft_fia import install as install_draft_fia\n        from device_metadata import install as install_device_metadata\n        install_draft_fia()\n        install_device_metadata()\n    qwen.after_init = after_init')
    p.write_text(text)
    p=dest/'service_smoke.py';text=p.read_text()
    late="        raw = json.loads((root/'prompt.json').read_text())\n        tokens = raw['prompt_token_ids'] if isinstance(raw,dict) else raw\n"
    text=replace_once(text,late,'')
    text=replace_once(text,'    port = 32492',"    raw = json.loads((root/'prompt.json').read_text())\n    tokens = raw['prompt_token_ids'] if isinstance(raw,dict) else raw\n    assert len(tokens)*10 >= 4096\n    port = 32492")
    text=text.replace("qualification='performance-only; text differences retained, not APC qualification'",
                      "qualification='experimental device continuation; strict APC boundary gates'")
    p.write_text(text)
    p=dest/'launch.sh';text=replace_once(p.read_text(),seed.name,dest.name)
    text=replace_once(text,'$CAPSULE/package:$base/runtime-source','$CAPSULE/package:$CAPSULE/runtime-source:$base/runtime-source')
    p.write_text(text)
    for p in dest.glob('*.py'):compile(p.read_text(),str(p),'exec')
    compile(modified,str(dest/'runtime-source'/target),'exec')
    print('CAPSULE PREFLIGHT PASS',dest,flush=True)


if __name__=='__main__':main()
