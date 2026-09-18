"""Matched SEND experiment: keep Cube/client binaries; rebuild Vector workers."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
from channel_layout import ChannelLayout
from expert_partition import ExpertPartition

p = argparse.ArgumentParser()
p.add_argument("base", type=Path)
p.add_argument("output", type=Path)
p.add_argument("--route-ready", action="store_true")
p.add_argument("--pipeline", action="store_true")
a = p.parse_args()
if a.pipeline and a.route_ready:
    p.error("pipelined export is currently qualified only without --route-ready")
shutil.copytree(a.base, a.output)
here = Path(__file__).resolve().parent
abi = json.loads((a.output / "abi.json").read_text())
layout = ChannelLayout.from_abi(abi)
partition = ExpertPartition(layout.owners)
groups = (partition.slots + 3) // 4 * 4
ends_storage = (groups + 7) // 8 * 8
text = (
    (here / "server_workers.hpp")
    .read_text()
    .replace("e < 128", "e < LOCAL_EXPERTS")
    .replace("offset < 256", f"offset < {ends_storage * 2}")
)
if layout.sources > 2:
    text = (
        text.replace("cfg[4 + c]", "SourcePointer(cfg, c)")
        .replace("cfg[2 + c]", "OutputPointer(cfg, c)")
        .replace("c < 2", "c < SOURCES")
        .replace(
            "int layer = desc[0] ? desc[2] : desc[MAP + 2];",
            "int layer = DescriptorLayer(desc);",
        )
    )
for enabled, macro in (
    (abi["batch_activate"], "QWEN38_BATCH_ACTIVATE"),
    (a.route_ready, "QWEN38_ROUTE_READY"),
    (a.pipeline, "QWEN38_PIPELINED_EXPORT"),
):
    if enabled:
        text = f"#define {macro} 1\n" + text
(a.output / "source/server_workers.hpp").write_text(text)
shutil.copyfile(here / "quant_export.hpp", a.output / "source/quant_export.hpp")
env = dict(
    os.environ,
    OUTPUT_DIR=str(a.output.resolve()),
    SOURCE=str(a.output.resolve() / "source/persistent_vector.cpp"),
    OBJECT_NAME="persistent_vector",
    LAUNCH_SOURCE=str(here / "launch.cpp"),
)
with (a.output / "export.build.log").open("w") as log:
    subprocess.run(
        ["bash", str(here.parent / "device-service/build.sh")],
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        check=True,
    )
abi.update(route_ready=a.route_ready, pipelined_export=a.pipeline)
(a.output / "abi.json").write_text(json.dumps(abi, indent=2) + "\n")
