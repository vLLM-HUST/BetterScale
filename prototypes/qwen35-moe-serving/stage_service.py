"""Stage the unqualified MoE FULL candidate from the geometry-only GDN pilot.

Keep donor BF16 MoE routing/grouped-GEMM/collectives; dense-only MC2 is not a
MoE implementation. Native FIA planning remains fail-closed on its actual ABI.
"""
import argparse
import difflib
import json
from pathlib import Path
import shutil

p = argparse.ArgumentParser()
p.add_argument('seed', type=Path)
p.add_argument('output', type=Path)
a = p.parse_args()
assert not a.output.exists()
shutil.copytree(a.seed, a.output, ignore=shutil.ignore_patterns(
    '__pycache__', 'runtime-source', '*.log', 'receipt.json'))
changes = {
    'package/betterscale/models/__init__.py': [
        ('hf.model_type == "qwen3_5_text"', 'hf.model_type == "qwen3_5_moe_text"')],
    'service_adapter.py': [
        ('qwen_gdn, qwen_fia, qwen_mc2', 'qwen_gdn, qwen_fia'),
        ('(64,5120,256)', '(40,2048,256)'),
        ('(16,48,128,128)', '(16,32,128,128)'),
        ('max_num_batched_tokens <= 2048', 'max_num_batched_tokens <= 4096'),
        ('qwen_mc2.install(worker.model_runner.model)',
         '# Native BF16 MoE and row collectives: no dense-only MC2 override.')],
    'package/betterscale/patches/qwen_layout.py': [
        ('count != 48', 'count != 30'), ('expected48 Qwen27', 'expected30 Qwen35MoE')],
    'package/betterscale/patches/qwen_fia/wave.py': [
        ('            12,\n            2,', '            8,\n            1,'),
        ('(12, 2, 256)', '(8, 1, 256)'), ('(2048, 12, 256)', '(4096, 8, 256)')],
    'draft_fia.py': [
        ('(12,2,256)', '(8,1,256)'), ('(2048,12,256)', '(4096,8,256)')],
    'apc_boundary.py': [
        ("model_type == 'qwen3_5_text'", "model_type == 'qwen3_5_moe_text'")],
    'package/betterscale/patches/qwen_gdn/graphs.py': [
        ('1024, 1536, 2048)', '1024, 1536, 2048, 4096)')],
    'package/betterscale/patches/qwen_gdn/preprocess.py': [
        ('0 < t <= 2048', '0 < t <= 4096')],
    'mixed_state_probe.py': [
        ('import torch_npu\n', 'import torch_npu\ntorch.npu.set_device(0)\n'),
        ('    torch.npu.set_device(0)\n', '')],
}
diff = []
for name, replacements in changes.items():
    path = a.output / name
    old = new = path.read_text()
    for before, after in replacements:
        assert before in new, (name, before)
        new = new.replace(before, after)
    path.write_text(new)
    diff.extend(difflib.unified_diff(old.splitlines(True), new.splitlines(True),
                                   fromfile='gdn-pilot/'+name, tofile='moe-full/'+name))
(a.output/'service-geometry.diff').write_text(''.join(diff))
(a.output/'service-contract.json').write_text(json.dumps({
    'status':'UNQUALIFIED', 'model':'Qwen3.5-35B-A3B', 'dtype':'BF16',
    'mtp':2, 'max_context':262144, 'target_graph':'FULL prefill/mixed/decode',
    'moe':'donor native BF16 routing/grouped GEMM/shared experts/TP collectives',
    'excluded':'dense27-only MC2 row projection override',
    'required_gates':['mixed GDN state oracle', 'new FIA head geometry oracle',
                      'changing MoE routing under graph replay',
                      'actual FULL prefill and mixed profiles',
                      'real weight and near-limit context qualification']}, indent=2)+'\n')
