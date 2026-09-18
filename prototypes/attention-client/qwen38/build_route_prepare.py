"""Bounded route-preparation experiments; none changes the production builder."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess


def prepare_tiled_routes(text):
    old = """    for (int i = 0; i < n * TOPK; ++i) {
      s.ids[c][i] = io.words.GetValue(i);
      if (s.ids[c][i] < 0 || s.ids[c][i] >= EXPERTS)
        return -1;
    }"""
    new = """    for (int i = 0; i < n * TOPK; ++i) {
      int expert = io.words.GetValue(i);
      if (expert < 0 || expert >= EXPERTS) return -1;
    }
    // Keep the complete validated frame in slot-owned GM with one DMA.
    io.Write(s.ids[c], (n * TOPK + 7) / 8 * 8);"""
    assert text.count(old) == 1
    text = text.replace(old, new)
    start = text.index("__aicore__ inline void Group(")
    end = text.index("__aicore__ inline void Command(", start)
    group = text[start:end]
    # The live map consumes at most (1024*10+8)*4 bytes. A separate 1KiB
    # route tile at48KiB survives its scalar edits and metadata publication.
    helper = """static_assert((ROUTES + 8) * 4 <= 49152, "Replan route tile UB for larger capacities");
__aicore__ inline void RouteTile(Transfer &io, __gm__ int32_t *ids, int n) {
  GlobalTensor<int32_t> gm;
  gm.SetGlobalBuffer(ids);
  DataCopy(io.words[12288], gm, (n + 7) / 8 * 8);
  SetFlag<HardEvent::MTE2_S>(EVENT_ID3);
  WaitFlag<HardEvent::MTE2_S>(EVENT_ID3);
}
"""
    old = """    if (s.gen[c])
      for (int i = 0; i < s.rows[c] * TOPK; ++i)
        if (s.ids[c][i] / LOCAL_EXPERTS == owner)
          ++count[(SINGLE_LAYER ? 0 : s.layer[c] * LOCAL_EXPERTS) +
                  s.ids[c][i] % LOCAL_EXPERTS];"""
    new = """    if (s.gen[c]) {
      int routes = s.rows[c] * TOPK;
      for (int begin = 0; begin < routes; begin += 256) {
        int size = ScalarMin(256, routes - begin);
        RouteTile(io, s.ids[c] + begin, size);
        for (int i = 0; i < size; ++i) {
          int expert = io.words.GetValue(12288 + i);
          if (expert / LOCAL_EXPERTS == owner)
            ++count[(SINGLE_LAYER ? 0 : s.layer[c] * LOCAL_EXPERTS) + expert % LOCAL_EXPERTS];
        }
      }
    }"""
    assert group.count(old) == 1
    group = group.replace(old, new)
    old = """    if (s.gen[c])
      for (int i = 0; i < s.rows[c] * TOPK; ++i)
        if (s.ids[c][i] / LOCAL_EXPERTS == owner)
          io.words.SetValue(
              8 + i, cursor[(SINGLE_LAYER ? 0 : s.layer[c] * LOCAL_EXPERTS) +
                            s.ids[c][i] % LOCAL_EXPERTS]++);"""
    new = """    if (s.gen[c]) {
      int routes = s.rows[c] * TOPK;
      for (int begin = 0; begin < routes; begin += 256) {
        int size = ScalarMin(256, routes - begin);
        RouteTile(io, s.ids[c] + begin, size);
        for (int i = 0; i < size; ++i) {
          int expert = io.words.GetValue(12288 + i);
          if (expert / LOCAL_EXPERTS == owner)
            io.words.SetValue(8 + begin + i,
              cursor[(SINGLE_LAYER ? 0 : s.layer[c] * LOCAL_EXPERTS) + expert % LOCAL_EXPERTS]++);
        }
      }
    }"""
    assert group.count(old) == 1
    group = group.replace(old, new)
    return text[:start] + helper + group + text[end:]


def prepare_local_routes(text):
    old = """      s.ids[c][i] = io.words.GetValue(i);
      if (s.ids[c][i] < 0 || s.ids[c][i] >= EXPERTS)
        return -1;"""
    new = """      int expert = io.words.GetValue(i);
      if (expert < 0 || expert >= EXPERTS) return -1;
      // This slot-private scratch is read only by Group, never by clients or
      // workers. Resolve ownership once; preserve original route positions.
      s.ids[c][i] = expert / LOCAL_EXPERTS == cfg[7]
          ? (SINGLE_LAYER ? 0 : layer * LOCAL_EXPERTS) + expert % LOCAL_EXPERTS
          : -1;"""
    assert text.count(old) == 1
    text = text.replace(old, new)
    old = """      for (int i = 0; i < s.rows[c] * TOPK; ++i)
        if (s.ids[c][i] / LOCAL_EXPERTS == owner)
          ++count[(SINGLE_LAYER ? 0 : s.layer[c] * LOCAL_EXPERTS) +
                  s.ids[c][i] % LOCAL_EXPERTS];"""
    new = """      for (int i = 0; i < s.rows[c] * TOPK; ++i) {
        int localExpert = s.ids[c][i];
        if (localExpert >= 0) ++count[localExpert];
      }"""
    assert text.count(old) == 1
    text = text.replace(old, new)
    old = """      for (int i = 0; i < s.rows[c] * TOPK; ++i)
        if (s.ids[c][i] / LOCAL_EXPERTS == owner)
          io.words.SetValue(
              8 + i, cursor[(SINGLE_LAYER ? 0 : s.layer[c] * LOCAL_EXPERTS) +
                            s.ids[c][i] % LOCAL_EXPERTS]++);"""
    new = """      for (int i = 0; i < s.rows[c] * TOPK; ++i) {
        int localExpert = s.ids[c][i];
        if (localExpert >= 0) io.words.SetValue(8 + i, cursor[localExpert]++);
      }"""
    assert text.count(old) == 1
    return text.replace(old, new)


def prepare_counted_routes(text):
    text = prepare_local_routes(text)
    sources = "SOURCES" if "c < SOURCES" in text else "2"
    assert text.count("struct Slot {") == 1
    text = text.replace(
        "struct Slot {", f"struct Slot {{\n  int counts[{sources}][GROUPS];"
    )
    old = """    for (int i = 0; i < n * TOPK; ++i) {
      int expert = io.words.GetValue(i);"""
    new = """    for (int g = 0; g < GROUPS; ++g) s.counts[c][g] = 0;
    for (int i = 0; i < n * TOPK; ++i) {
      int expert = io.words.GetValue(i);"""
    assert text.count(old) == 1
    text = text.replace(old, new)
    old = """      s.ids[c][i] = expert / LOCAL_EXPERTS == cfg[7]
          ? (SINGLE_LAYER ? 0 : layer * LOCAL_EXPERTS) + expert % LOCAL_EXPERTS
          : -1;"""
    new = """      int localExpert = expert / LOCAL_EXPERTS == cfg[7]
          ? (SINGLE_LAYER ? 0 : layer * LOCAL_EXPERTS) + expert % LOCAL_EXPERTS
          : -1;
      s.ids[c][i] = localExpert;
      if (localExpert >= 0) ++s.counts[c][localExpert];"""
    assert text.count(old) == 1
    text = text.replace(old, new)
    old = """      for (int i = 0; i < s.rows[c] * TOPK; ++i) {
        int localExpert = s.ids[c][i];
        if (localExpert >= 0) ++count[localExpert];
      }"""
    new = """      for (int g = 0; g < GROUPS; ++g) count[g] += s.counts[c][g];"""
    assert text.count(old) == 1
    return text.replace(old, new)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("base", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument(
        "--strategy",
        choices=("local_ids", "local_counts", "ub_tiles"),
        required=True,
    )
    a = p.parse_args()
    shutil.copytree(a.base, a.output)
    here = Path(__file__).resolve().parent
    source = a.output.resolve() / "source/persistent_vector.cpp"
    transforms = {
        "local_ids": prepare_local_routes,
        "local_counts": prepare_counted_routes,
        "ub_tiles": prepare_tiled_routes,
    }
    source.write_text(transforms[a.strategy](source.read_text()))
    env = dict(
        os.environ,
        OUTPUT_DIR=str(a.output.resolve()),
        SOURCE=str(source),
        OBJECT_NAME="persistent_vector",
        LAUNCH_SOURCE=str(here / "launch.cpp"),
    )
    with (a.output / "route-prepare.build.log").open("w") as log:
        subprocess.run(
            ["bash", str(here.parent / "device-service/build.sh")],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    abi = json.loads((a.output / "abi.json").read_text())
    abi["route_prepare"] = a.strategy
    (a.output / "abi.json").write_text(json.dumps(abi, indent=2) + "\n")
