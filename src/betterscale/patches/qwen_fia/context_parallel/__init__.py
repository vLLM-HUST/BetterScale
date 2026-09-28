"""Qualified balanced decode attention; inert native-artifact admission."""

import hashlib
import json
from pathlib import Path


def configure(env):
    """Qwen35 default; preserve explicit diagnostic disable and library override."""
    choice = env.setdefault("BETTERSCALE_CONTEXT_PARALLEL", "1")
    if choice not in ("0", "1"):
        raise ValueError("BETTERSCALE_CONTEXT_PARALLEL must be 0 or 1")
    if choice == "0":
        return
    root = Path(__file__).parent
    library = Path(env.get("BETTERSCALE_CP_LIBRARY", root / "libbs_fia_cp.so")).resolve()
    contract = json.loads((root / "native.json").read_text())
    if not library.is_file():
        raise FileNotFoundError(
            f"Missing balanced attention library: {library}; install the complete distribution"
        )
    if hashlib.sha256(library.read_bytes()).hexdigest() != contract["sha256"]:
        raise ValueError("BETTERSCALE_CP_LIBRARY must name the qualified balanced attention library")
    env["BETTERSCALE_CP_LIBRARY"] = str(library)
