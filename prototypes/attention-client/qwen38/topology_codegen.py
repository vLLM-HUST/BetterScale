"""Expand source mailboxes independently of the two persistent staging slots.

Applied only to an explicit ABI5 build, after freezing the qualified two-source
closure. Keep the Cube/Vector two-slot protocol intact. Source pointers move to
cfg27/28 tables; per-source completion counters and32-word traces are versioned.
"""

from pathlib import Path
import re


def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Unexpected frozen protocol fragment: {old!r}")
    return text.replace(old, new)


def expand_sources(directory: Path, sources: int):
    if sources not in (4, 5):
        raise ValueError("expanded server admits4 or5 source mailboxes")
    path = directory / "persistent_protocol.hpp"
    text = path.read_text().replace(
        "CAPACITY = ROUTES * 2", "CAPACITY = ROUTES * SOURCES"
    )
    text = once(
        text,
        "constexpr int TOKENS =",
        f"constexpr int SOURCES = {sources};\nconstexpr int TOKENS =",
    )
    text += """
namespace Persistent {
__aicore__ inline int64_t SourcePointer(__gm__ int64_t *cfg, int source) {
  return ((__gm__ int64_t *)cfg[27])[source];
}
__aicore__ inline int64_t OutputPointer(__gm__ int64_t *cfg, int source) {
  return ((__gm__ int64_t *)cfg[28])[source];
}
__aicore__ inline int SumSources(const int *values) {
  int result = 0;
  for (int source = 0; source < SOURCES; ++source) result += values[source];
  return result;
}
__aicore__ inline bool SameLayer(const int *generations, const int *layers, int layer) {
  for (int source = 0; source < SOURCES; ++source)
    if (generations[source] && layers[source] != layer) return false;
  return true;
}
__aicore__ inline int DescriptorLayer(__gm__ int32_t *desc) {
  for (int source = 0; source < SOURCES; ++source) {
    Refresh(desc + source * MAP);
    if (desc[source * MAP]) return desc[source * MAP + 2];
  }
  return -1;
}
}
"""
    path.write_text(text)
    path = directory / "priority_policy.hpp"
    text = path.read_text().replace("j < 2", "j < SOURCES").replace("% 2", "% SOURCES")
    text = once(text, "turn = 1 - source;", "turn = (source + 1) % SOURCES;")
    path.write_text(text)
    path = directory / "persistent_vector.cpp"
    text = path.read_text()
    # These are source arrays, not Slot s[2], command phases or buffer indices.
    names = "gen|rows|layer|ids|pending|generations|layers|priorities|urgency|claimed|finished|closed"
    text = re.sub(rf"\b({names})\[2\](?=\s*[;=,])", r"\1[SOURCES]", text)
    text = text.replace("c < 2", "c < SOURCES")
    text = once(
        text,
        "for (int j = 0; j < 2; ++j) {\n    int c = (first + j) % 2;",
        "for (int j = 0; j < SOURCES; ++j) {\n    int c = (first + j) % SOURCES;",
    )
    text = once(
        text,
        "bool occupied = s.gen[0] || s.gen[1];",
        "bool occupied = SumSources(s.gen) != 0;",
    )
    text = once(
        text,
        "(SINGLE_LAYER && ((s.gen[0] && s.layer[0] != layer) ||\n                                       (s.gen[1] && s.layer[1] != layer)))",
        "(SINGLE_LAYER && !SameLayer(s.gen, s.layer, layer))",
    )
    text = text.replace("slot.rows[0] + slot.rows[1]", "SumSources(slot.rows)")
    start = text.index("          int fields[16] = {slot.gen[0],")
    end = text.index("          slot.stage = EMPTY;", start)
    text = text[:start] + """          int fields[32] = {};
          for (int source = 0; source < SOURCES; ++source) {
            fields[source] = slot.gen[source];
            fields[SOURCES + source] = slot.gen[source] ? slot.layer[source] : -1;
            fields[2 * SOURCES + source] = slot.rows[source];
            slot.gen[source] = slot.rows[source] = 0;
          }
          fields[3 * SOURCES] = slot.live;
          fields[3 * SOURCES + 1] = vs;
          fields[3 * SOURCES + 2] = slot.priority;
          fields[3 * SOURCES + 3] = slot.ticket;
          fields[3 * SOURCES + 4] = slot.serviceRank;
          for (int i = 0; i < 32; ++i) io.words.SetValue(i, fields[i]);
          io.Write((__gm__ int32_t *)cfg[8] + (waves % (cfg[6] * 2)) * 32, 32);
          ++waves;
""" + text[end:]
    text = once(
        text,
        "if (cfg[24] ? (closed[0] && closed[1])\n                : (finished[0] == cfg[6] && finished[1] == cfg[6]))",
        "if (cfg[24] ? SumSources(closed) == SOURCES\n                : SumSources(finished) == SOURCES * cfg[6])",
    )
    text = once(
        text,
        "ctrl[STATUS * LINE + 4] = finished[0];\n      ctrl[STATUS * LINE + 5] = finished[1];\n      ctrl[STATUS * LINE + 6] = promotions;",
        "for (int source = 0; source < SOURCES; ++source)\n        ctrl[STATUS * LINE + 4 + source] = finished[source];\n      ctrl[STATUS * LINE + 4 + SOURCES] = promotions;",
    )
    path.write_text(text)
    for name in ("persistent_vector.cpp", "server_workers.hpp", "persistent_cube.cpp"):
        path = directory / name
        text = (
            path.read_text()
            .replace("cfg[4 + c]", "SourcePointer(cfg, c)")
            .replace("cfg[2 + c]", "OutputPointer(cfg, c)")
        )
        text = text.replace("c < 2", "c < SOURCES")
        text = text.replace(
            "int layer = desc[0] ? desc[2] : desc[MAP + 2];",
            "int layer = DescriptorLayer(desc);",
        )
        path.write_text(text)
