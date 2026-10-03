"""Copy qualified task capsules; overlay only the reviewed cache source seam."""
import argparse
from pathlib import Path
import shutil

FILES=("worker.py", "models/qwen35/state_backend.py", "models/qwen35/state_slots.py", "models/qwen35/seat_scheduler.py", "models/qwen35/resident_leases.py",
       "models/qwen35/cache_actions.py", "models/qwen35/cache_engine.py",
       "models/qwen35/cache_pages.py", "models/qwen35/cache_policy.py",
       "models/qwen35/cache_worker.py", "models/qwen35/state_dma.py", "live/runtime/host_state.py",
       "live/runtime/page_state.py", "live/runtime/page_transport.py",
       'models/qwen35/execution_capacity.py',
       'models/qwen35/count_policy.py',
       'models/qwen35/mixed_core.py',
       'models/qwen35/service_metadata.py',
       'models/qwen35/host_metadata.py',
       'models/qwen35/device_metadata.py',
       'models/qwen35/device_slots.py',
       'models/qwen35/integration.py',
       'patches/qwen_gdn/metadata.py',
       'patches/qwen_fia/wave.py',
       'patches/qwen_fia/context_parallel/plan.py',
       'patches/qwen_fia/context_parallel/adapter.py')

def stage(repo,baseline,output,*,mtp=False):
    if output.exists():raise ValueError("Never overwrite a running/frozen capsule")
    shutil.copytree(baseline,output,ignore=shutil.ignore_patterns("__pycache__"))
    for name in FILES + (("models/qwen35/state_address.py","models/qwen35/draft_fia.py","models/qwen35/draft_banks.py") if mtp else ()):
        shutil.copy2(repo/"src/betterscale"/name,output/"betterscale"/name)
    # Preserve the qualified capsule's DP/EP admission, changing only its row
    # envelope; replacing this whole module would silently revert DP4 to DP1.
    entry=output/"betterscale/models/qwen35/__init__.py"
    value=entry.read_text()
    old="s.max_num_seqs == 16"
    if value.count(old)!=1:
        raise ValueError("Unknown capsule execution admission")
    value=value.replace(old,"s.max_num_seqs == EXECUTION")
    value += "\nfrom .execution_capacity import EXECUTION\n"
    entry.write_text(value)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo",type=Path,required=True)
    p.add_argument("--baseline",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--mtp",action="store_true")
    a=p.parse_args();stage(a.repo,a.baseline,a.output,mtp=a.mtp)
