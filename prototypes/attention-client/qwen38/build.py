"""Freeze a Qwen38 ABI closure, reusing the existing priority coordinator.

Only the worker data plane and matrix type change. Existing Next binaries and
published Worker defaults are not rewritten. No generated file is a source API.
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
from channel_layout import ChannelLayout
from expert_partition import ExpertPartition

p = argparse.ArgumentParser()
p.add_argument("output", type=Path)
p.add_argument("--rows", type=int, default=32)
p.add_argument("--owners", type=int, choices=(3, 4), default=4)
p.add_argument("--sources", type=int, choices=(2, 4, 5), default=2)
a = p.parse_args()
layout = ChannelLayout(a.rows, a.owners, a.sources)
partition = ExpertPartition(a.owners)
groups = (partition.slots + 3) // 4 * 4
ends_storage = (groups + 7) // 8 * 8
repo = Path(__file__).resolve().parents[3]
local = Path(__file__).resolve().parent
base = local.parent / "device-service"
out = a.output.resolve()
source = out / "source"
source.mkdir(parents=True, exist_ok=True)
for name in ("actual_gmm.cpp", "priority_policy.hpp"):
    shutil.copyfile(base / name, source / name)
for name in ("server_workers.hpp", "quant_vector.hpp", "quant_gmm.cpp", "launch.cpp"):
    shutil.copyfile(local / name, source / name)
protocol = (base / "persistent_protocol.hpp").read_text()
assert "HIDDEN = 2048, INNER = 512" in protocol
protocol = protocol.replace(
    "HIDDEN = 2048, INNER = 512", "HIDDEN = 2560, INNER = 640"
).replace("LAYERS = 48", "LAYERS = 49")
protocol = protocol.replace("TOKENS = 32", f"TOKENS = {layout.rows}")
protocol = protocol.replace(
    "LOCAL_EXPERTS = 128", f"LOCAL_EXPERTS = {partition.slots}"
).replace("GROUPS = 128", f"GROUPS = {groups}")
protocol = protocol.replace(
    "constexpr int MAP =",
    f"constexpr int SOURCE_SCALES = {layout.scales}, SOURCE_PAYLOAD = {layout.payload};\nconstexpr int MAP =",
)
(source / "persistent_protocol.hpp").write_text("#define QWEN_NEXT 1\n" + protocol)
coordinator = (base / "persistent_vector.cpp").read_text()
assert coordinator.count("struct Slot {") == 1
coordinator = coordinator[coordinator.index("struct Slot {") :].replace(
    "n > 32", "n > TOKENS"
)
# Route IDs belong to the persistent slot, not the AIV function stack.
# Only the coordinator accesses this scratch; workers consume published maps.
coordinator = coordinator.replace("int ids[2][ROUTES],", "__gm__ int32_t *ids[2]; int")
coordinator = coordinator.replace(
    "Slot s[2];",
    """Slot s[2];
  for (int slot = 0; slot < 2; ++slot)
    for (int c = 0; c < 2; ++c)
      s[slot].ids[c] = (__gm__ int32_t *)slots[slot * 16 + 14] + c * ROUTES;""",
)
(source / "persistent_vector.cpp").write_text(
    '#include "server_workers.hpp"\n#include "priority_policy.hpp"\n' + coordinator
)
shutil.copyfile(local / "server_cube.cpp", source / "persistent_cube.cpp")
for filename in ("server_workers.hpp", "persistent_cube.cpp"):
    path = source / filename
    text = path.read_text().replace("e < 128", "e < LOCAL_EXPERTS")
    text = text.replace("offset < 256", f"offset < {ends_storage * 2}")
    path.write_text(text)
client = (local.parent / "qwen-next/client_kernel.cpp").read_text()
assert "constexpr int H = 2048" in client
client = client.replace("constexpr int H = 2048", "constexpr int H = 2560").replace(
    "ROUTES = 320", f"ROUTES = {layout.routes}"
)
old = """  for (int row = 0; row < n; ++row) {
    io.Read((__gm__ int32_t *)hidden + row * H / 2, H / 2);
    io.Write(src + 1024 + row * H / 2, H / 2);
  }"""
assert old in client
client = client.replace(
    old,
    """  bool quantized = cfg[5] < 48;
  int width = H / (quantized ? 4 : 2);
  for (int row = 0; row < n; ++row) {
    io.Read((__gm__ int32_t *)hidden + row * width, width);
    io.Write(src + 1024 + row * width, width);
    if (quantized) {
      // cfg16 names the padded native FP32 input-scale tensor.
      io.Read((__gm__ int32_t *)cfg[16] + (row / 8) * 8, 8);
      int scaleBits = io.ub.GetValue(row % 8);
      for (int j = 0; j < 8; ++j) io.ub.SetValue(j, j ? 0 : scaleBits);
      io.Write(src + 512 + row * 8, 8);
    }
  }""",
)
client = client.replace("src + 1024", f"src + {layout.payload}").replace(
    "src + 512", f"src + {layout.scales}"
)
old_collect = """  io.Read((__gm__ int32_t *)topk, (n * TOPK + 7) / 8 * 8);
  int ids[ROUTES];
  for (int i = 0; i < n * TOPK; ++i)
    ids[i] = io.ub.GetValue(i);
  for (int i = GetBlockIdx(); i < n * TOPK; i += GetBlockNum()) {
    int owner = ids[i] / 128;"""
assert old_collect in client
client = client.replace(
    old_collect,
    """  for (int i = GetBlockIdx(); i < n * TOPK; i += GetBlockNum()) {
    io.Read((__gm__ int32_t *)topk + (i / 8) * 8, 8);
    int owner = io.ub.GetValue(i % 8) / 128;""",
    1,
)
# The first INT8 gate uses complete-owner collect, not the old H2048 token-pull
# UB layout or BF16-only early-return pipeline. Do not ship a malformed export.
start = client.index('extern "C" __global__ __aicore__ void\nneural_collect_reduce')
end = client.index('extern "C" __global__ __aicore__ void\nneural_retire', start)
client = client[:start] + client[end:]
client = client.replace("META(neural_collect_reduce)", "")
client = client.replace("owner < 4", f"owner < {a.owners}").replace(
    "/ 128", f"/ {partition.slots}"
)
(source / "client_kernel.cpp").write_text(client)
if a.sources > 2:
    from topology_codegen import expand_sources

    expand_sources(source, a.sources)
env = dict(os.environ, OUTPUT_DIR=str(out), LAUNCH_SOURCE=str(source / "launch.cpp"))
for script, unit, name in (
    ("build.sh", "persistent_vector.cpp", "persistent_vector"),
    ("build.sh", "client_kernel.cpp", "queue_service"),
    ("build_actual_gmm.sh", "persistent_cube.cpp", "persistent_cube"),
):
    with (out / (name + ".build.log")).open("w") as log:
        subprocess.run(
            ["bash", str(base / script)],
            env=dict(env, SOURCE=str(source / unit), OBJECT_NAME=name),
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
(out / "abi.json").write_text(
    json.dumps(
        dict(
            version=5 if a.sources > 2 else (3 if a.owners == 4 else 4),
            kernel_timeout_us=1200000000,
            model="qwen38",
            server_config_words=29 if a.sources > 2 else 27,
            client_config_words=17,
            weight_pointer_columns=4,
            hidden=2560,
            inner=640,
            topk=10,
            owners=a.owners,
            sources=a.sources,
            rows=layout.rows,
            source_scale_words=layout.scales,
            source_payload_words=layout.payload,
            target_input="native_dynamic_int8_and_fp32_scale",
            mtp_input="bf16",
            prefix_pipeline=False,
        ),
        indent=2,
    )
    + "\n"
)
print(out, flush=True)
