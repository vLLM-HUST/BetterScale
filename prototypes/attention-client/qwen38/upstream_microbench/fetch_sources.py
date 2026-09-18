"""Fetch exact upstream leaves into an untracked experiment directory."""

import argparse
import json
from pathlib import Path
from urllib.request import urlopen

PINS = {
    "sglang": (
        "sgl-project/sgl-kernel-npu",
        "7ede76da2f3e12fa16f7cad3161b199b24aeb6e1",
    ),
    "ascend": ("vllm-project/vllm-ascend", "9082742026e64e0311b4f4913850817dbc752875"),
}
FILES = {
    "sglang": [
        "python/sgl_kernel_npu/sgl_kernel_npu/qwen3_8_flash_next/" + x + ".py"
        for x in ("mqa", "sparse_attention", "hc", "expansion")
    ],
    "ascend": [
        "vllm_ascend/models/qwen4_exp/lightning_indexer.py",
        "vllm_ascend/ops/triton/qwen4_exp/qsa.py",
    ],
}
p = argparse.ArgumentParser()
p.add_argument("output", type=Path)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
manifest = []
for family, paths in FILES.items():
    repo, sha = PINS[family]
    target = a.output / family
    target.mkdir()
    for path in paths + ["LICENSE"]:
        url = f"https://raw.githubusercontent.com/{repo}/{sha}/{path}"
        data = urlopen(url, timeout=45).read()
        dest = target / Path(path).name
        dest.write_bytes(data)
        manifest.append(
            dict(
                repo=repo,
                sha=sha,
                path=path,
                local=str(dest.relative_to(a.output)),
                url=url,
            )
        )
(a.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(a.output)
