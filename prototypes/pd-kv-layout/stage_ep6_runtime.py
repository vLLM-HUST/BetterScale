"""Stage only the two pinned Ascend A2/EP6 uneven-linear admission corrections.

No MC2, ALLTOALL, EPLB, quantization or shared donor mutation is admitted.
"""
import argparse,json,shutil,subprocess
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();root=Path(__file__).resolve().parents[2]
assert not a.output.exists() and not a.output.resolve().is_relative_to(a.source.resolve())
files={
'vllm_ascend/ops/fused_moe/fused_moe.py':('''        if local_num_experts != expected_local_num_experts:''','''        if (moe_config.ep_size == 6 and placement_moe_config.num_experts == 256
                and not eplb_config.dynamic_eplb and not eplb_config.expert_map_path
                and self.global_redundant_expert_num == 0):
            expected_local_num_experts = 42 + int(moe_config.ep_rank < 4)
        if local_num_experts != expected_local_num_experts:'''),
'vllm_ascend/ops/fused_moe/token_dispatcher.py':('''            first_expert_idx = get_ep_group().rank_in_group * self.num_experts_local''','''            ep = get_ep_group()
            if ep.world_size == 6 and len(expert_map) == 256 and global_redundant_expert_num == 0:
                expected_count = 42 + int(ep.rank_in_group < 4)
                if self.num_experts_local != expected_count:
                    raise ValueError("Unqualified EP6 local expert count")
                first_expert_idx = ep.rank_in_group * 42 + min(ep.rank_in_group, 4)
            else:
                first_expert_idx = ep.rank_in_group * self.num_experts_local''')}
patched={}
for name,(old,new) in files.items():
 source=(a.source/name).read_text();original=subprocess.check_output(['git','show','9bf964cb4b87c8cd0d6852c41a55b3c29711fa95:'+name],cwd=root/'upstream/vllm-ascend',text=True)
 assert source==original,name
 assert source.count(old)==1,name
 patched[name]=source.replace(old,new)
# Native probe must not inherit the State/MTP-patched model runner.
runner='vllm_ascend/worker/model_runner_v1.py'
assert (a.source/runner).read_text()==subprocess.check_output(['git','show','9bf964cb4b87c8cd0d6852c41a55b3c29711fa95:'+runner],cwd=root/'upstream/vllm-ascend',text=True)
shutil.copytree(a.source/'vllm_ascend',a.output/'vllm_ascend',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
for name,source in patched.items():(a.output/name).write_text(source)
(a.output/'ep6-candidate.json').write_text(json.dumps(dict(donor='9bf964cb4b87c8cd0d6852c41a55b3c29711fa95',
    changed_files=list(files),scope='UNQUALIFIED A2 BF16 EP6 linear256 no EPLB/LoRA',source_runtime=str(a.source)),indent=2)+'\n')
print(a.output)
