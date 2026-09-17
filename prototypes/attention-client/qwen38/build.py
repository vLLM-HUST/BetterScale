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

p = argparse.ArgumentParser()
p.add_argument("output", type=Path)
a = p.parse_args()
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
(source / "persistent_protocol.hpp").write_text("#define QWEN_NEXT 1\n" + protocol)
coordinator = (base / "persistent_vector.cpp").read_text()
assert coordinator.count("struct Slot {") == 1
coordinator = coordinator[coordinator.index("struct Slot {") :]
(source / "persistent_vector.cpp").write_text(
    '#include "server_workers.hpp"\n#include "priority_policy.hpp"\n' + coordinator
)
shutil.copyfile(local / "server_cube.cpp", source / "persistent_cube.cpp")
client = (local.parent / "qwen-next/client_kernel.cpp").read_text()
assert "constexpr int H = 2048" in client
client = client.replace("constexpr int H = 2048", "constexpr int H = 2560")
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
# The first INT8 gate uses complete-owner collect, not the old H2048 token-pull
# UB layout or BF16-only early-return pipeline. Do not ship a malformed export.
start = client.index('extern "C" __global__ __aicore__ void\nneural_collect_reduce')
end = client.index('extern "C" __global__ __aicore__ void\nneural_retire', start)
client = client[:start] + client[end:]
client = client.replace("META(neural_collect_reduce)", "")
(source / "client_kernel.cpp").write_text(client)
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
            version=2,
            kernel_timeout_us=1200000000,
            model="qwen38",
            server_config_words=27,
            client_config_words=17,
            weight_pointer_columns=4,
            hidden=2560,
            inner=640,
            topk=10,
            owners=4,
            rows=32,
            target_input="native_dynamic_int8_and_fp32_scale",
            mtp_input="bf16",
            prefix_pipeline=False,
        ),
        indent=2,
    )
    + "\n"
)
print(out, flush=True)
