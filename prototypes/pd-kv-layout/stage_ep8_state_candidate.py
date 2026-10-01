"""Copy the qualified experimental EP6 package into an explicit EP8 candidate.

No installed/released package changes, arithmetic changes, or MTP qualification.
The source must already contain the target-only and FIA-padding qualification.
"""
import argparse
import json
from pathlib import Path
import shutil


def stage(source,output):
    changes={
        "betterscale/models/qwen35/__init__.py":(
            "== (2, 3, 1, True, 1, 1)","== (2, 4, 1, True, 1, 1)"),
        "betterscale/models/qwen35/small_fish_runtime.py":(
            "!= (2, 3, 1, True)","!= (2, 4, 1, True)"),
    }
    if output.exists():raise FileExistsError(output)
    if not (source/"target_only_diagnostic.py").is_file():
        raise ValueError("Missing target-only experiment")
    adapter=source/"betterscale/patches/qwen_fia/context_parallel/adapter.py"
    if "def target_metadata(" not in adapter.read_text():
        raise ValueError("Missing qualified positive-padding adapter")
    patched={}
    for path,(old,new) in changes.items():
        text=(source/path).read_text()
        if text.count(old)!=1:raise ValueError("Unexpected topology guard: "+path)
        patched[path]=text.replace(old,new).replace(
            "EXPERIMENTAL TP2/DP3/EP6/PP1","EXPERIMENTAL TP2/DP4/EP8/PP1")
    shutil.copytree(source,output,ignore=shutil.ignore_patterns("__pycache__","*.pyc"))
    for path,text in patched.items():(output/path).write_text(text)
    (output/"ep8-state-experiment.json").write_text(json.dumps(dict(
        source=str(source),changed_files=list(changes),
        scope="Experimental DP4 TP2 EP8, target-only; draft startup retained, not MTP qualified"),indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();stage(args.source,args.output)
