"""Disconnected TP2/DP4 overlay; never mutate the pinned source or base runtime.

The old research runtime's distributed facade equates TP with WORLD. Keep all
imported function identities, adding selected TP/DP coordinators in its source.
Native EP remains WORLD8. Everything else points at the immutable input closure.
"""

import argparse
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("source", type=Path)
p.add_argument("output", type=Path)
a = p.parse_args()
source = a.source.resolve()
out = a.output.resolve()
if out.exists():
    raise FileExistsError(out)
relative = Path("livemodule/llm/distributed.py")
text = (source / relative).read_text()
text += "\n# Installed by the explicit colocated topology bootstrap before model load.\n_BETTERSCALE_GROUPS = {}\n"
for name in (
    "get_tp_group",
    "get_otp_group",
    "get_flashcomm2_otp_group",
    "get_mlp_tp_group",
):
    old = f"def {name}() -> GroupCoordinator:\n    return _world()"
    if text.count(old) != 1:
        raise ValueError(f"Unexpected selected runtime function {name}")
    text = text.replace(
        old,
        f'def {name}() -> GroupCoordinator:\n    return _BETTERSCALE_GROUPS.get("tp") or _world()',
    )
old = "def get_dp_group() -> GroupCoordinator:\n    return _singleton()"
if text.count(old) != 1:
    raise ValueError("Unexpected DP facade")
text = text.replace(
    old,
    'def get_dp_group() -> GroupCoordinator:\n    return _BETTERSCALE_GROUPS.get("dp") or _singleton()',
)


# Only the ancestors of the edited leaf need physical directories.
def mirror(src, dst, parts):
    dst.mkdir()
    for child in src.iterdir():
        if child.name == parts[0]:
            if len(parts) > 1:
                mirror(child, dst / child.name, parts[1:])
            else:
                (dst / child.name).write_text(text)
        else:
            (dst / child.name).symlink_to(child, target_is_directory=child.is_dir())


out.parent.mkdir(parents=True, exist_ok=True)
mirror(source, out, relative.parts)
print(out)
