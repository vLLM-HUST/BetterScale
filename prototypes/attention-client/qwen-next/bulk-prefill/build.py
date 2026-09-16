"""Freeze an isolated bulk-row server variant; do not widen the serving ABI.

The released prototype remains 32 rows. Expanding its route map also requires
moving hidden payload beyond the IDs, resizing both staging slots and packing
the coordinator's retained expert IDs into int16 UB storage (512 experts fit).
Workers retain only their strided map entries. All transformed sources are retained.
"""

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import subprocess

p = argparse.ArgumentParser()
p.add_argument("output", type=Path)
p.add_argument("--rows", type=int, choices=(256, 512, 1024), default=1024)
a = p.parse_args()
rows = a.rows
payload = ((64 + rows * 10 + 1023) // 1024) * 1024


def aligned(n):
    return ((n + 2 * 1024**2 - 1) // (2 * 1024**2)) * 2 * 1024**2


repo = Path(__file__).resolve().parents[4]
source = repo / "prototypes/attention-client/device-service"
out = a.output.resolve()
frozen = out / "source"
frozen.mkdir(parents=True, exist_ok=True)
for name in (
    "persistent_vector.cpp",
    "persistent_cube.cpp",
    "persistent_protocol.hpp",
    "streaming_gmm.hpp",
    "actual_gmm.cpp",
    "launch.cpp",
    "persistent_service.py",
):
    text = (source / name).read_text()
    if name == "persistent_protocol.hpp":
        assert "TOKENS = 32" in text
        text = text.replace("TOKENS = 32", f"TOKENS = {rows}")
    if name == "persistent_vector.cpp":
        for old, new in [
            ("n > 32", f"n > {rows}"),
            ("c * 32", f"c * {rows}"),
            ("+ 1024 + row * HIDDEN", f"+ {payload} + row * HIDDEN"),
        ]:
            assert old in text, old
            text = text.replace(old, new)
    if name == "persistent_vector.cpp":
        old = "int ids[2][ROUTES], live = 0, boundary = 0;"
        assert old in text
        text = text.replace(
            old, "LocalTensor<int16_t> ids; int live = 0, boundary = 0;"
        )
        old = """      s.ids[c][i] = io.words.GetValue(i);
      if (s.ids[c][i] < 0 || s.ids[c][i] >= EXPERTS)
        return -1;"""
        new = """      int id = io.words.GetValue(i);
      if (id < 0 || id >= EXPERTS) return -1;
      s.ids.SetValue(c * ROUTES + i, id);"""
        assert old in text
        text = text.replace(old, new)
        text = text.replace("s.ids[c][i]", "s.ids.GetValue(c * ROUTES + i)")
        text = text.replace(
            "pipe.InitBuffer(buf, 32768);", "pipe.InitBuffer(buf, 65536);"
        )
        text = text.replace(
            "  Slot s[2];",
            """  Slot s[2];
  TBuf<TPosition::VECCALC> retained;
  io.pipe.InitBuffer(retained, 4 * ROUTES * sizeof(int16_t));
  for (int i = 0; i < 2; ++i)
    s[i].ids = retained.Get<int16_t>()[i * 2 * ROUTES];""",
        )
        edits = [
            ("int maps[2][ROUTES]", "int maps[2][(ROUTES + VW - 1) / VW]"),
            (
                "for (int r = 0; r < ROUTES; ++r)\n      maps[c][r] = io.words.GetValue(8 + r);",
                "for (int r = worker; r < ROUTES; r += VW)\n      maps[c][r / VW] = io.words.GetValue(8 + r);",
            ),
            ("maps[c][route]", "maps[c][route / VW]"),
            ("int map[ROUTES];", "int map[(ROUTES + VW - 1) / VW];"),
            (
                "for (int i = 0; i < ROUTES; ++i)\n          map[i] = io.words.GetValue(8 + i);",
                "for (int i = worker; i < ROUTES; i += VW)\n          map[i / VW] = io.words.GetValue(8 + i);",
            ),
            ("map[route]", "map[route / VW]"),
        ]
        for old, new in edits:
            assert old in text, old
            text = text.replace(old, new)
    if name in ("persistent_vector.cpp", "persistent_cube.cpp"):
        text = "#define QWEN_NEXT 1\n" + text
    if name == "persistent_service.py":
        for old, new in [
            ("64 * topk, 32 * topk + 8", f"{2*rows} * topk, {rows} * topk + 8"),
            ("(2, 32, 2048)", f"(2, {rows}, 2048)"),
        ]:
            assert old in text, old
            text = text.replace(old, new)
    (frozen / name).write_text(text)
for name, unit, obj in [
    ("build.sh", "persistent_vector.cpp", "persistent_vector"),
    ("build_actual_gmm.sh", "persistent_cube.cpp", "persistent_cube"),
]:
    text = (source / name).read_text()
    text = re.sub(r"^ROOT=.*$", "ROOT=" + shlex.quote(str(repo)), text, flags=re.M)
    script = frozen / name
    script.write_text(text)
    env = dict(
        os.environ,
        OUTPUT_DIR=str(out),
        SOURCE=str(frozen / unit),
        OBJECT_NAME=obj,
        LAUNCH_SOURCE=str(frozen / "launch.cpp"),
    )
    subprocess.run(["bash", str(script)], env=env, check=True)
(out / "geometry.json").write_text(
    json.dumps(
        dict(
            rows=rows,
            topk=10,
            hidden=2048,
            payload_words=payload,
            source_bytes=aligned((payload + rows * 1024) * 4),
            output_bytes=aligned((64 + rows * 10 * (1024 + 16)) * 4),
        ),
        indent=2,
    )
)
