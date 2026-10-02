"""Copy qualified task capsules; overlay only the reviewed cache source seam."""
import argparse
from pathlib import Path
import shutil

FILES=("worker.py", "models/qwen35/seat_scheduler.py", "models/qwen35/resident_leases.py",
       "models/qwen35/cache_actions.py", "models/qwen35/cache_engine.py",
       "models/qwen35/cache_pages.py", "models/qwen35/cache_policy.py",
       "models/qwen35/cache_worker.py", "models/qwen35/state_dma.py", "live/runtime/host_state.py",
       "live/runtime/page_state.py", "live/runtime/page_transport.py")

def stage(repo,baseline,output,*,mtp=False):
    if output.exists():raise ValueError("Never overwrite a running/frozen capsule")
    shutil.copytree(baseline,output,ignore=shutil.ignore_patterns("__pycache__"))
    for name in FILES + (("models/qwen35/state_address.py","models/qwen35/draft_fia.py") if mtp else ()):
        shutil.copy2(repo/"src/betterscale"/name,output/"betterscale"/name)

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo",type=Path,required=True)
    p.add_argument("--baseline",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--mtp",action="store_true")
    a=p.parse_args();stage(a.repo,a.baseline,a.output,mtp=a.mtp)
