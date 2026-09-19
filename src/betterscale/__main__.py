"""Launch the qualified Qwen service with packaged native artifacts."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys


def prepare(model, devices, port, cache_dir):
    """Prepare a new process; never preload CANN into this interpreter."""
    pair = devices.split(",")
    if (
        len(pair) != 2
        or len(set(pair)) != 2
        or any(not item.isdigit() or int(item) not in range(8) for item in pair)
    ):
        raise ValueError("--devices needs two distinct device IDs, e.g. 0,1")
    if not 1 <= port <= 65535:
        raise ValueError("--port must be in 1..65535")
    root = Path(__file__).parent
    env = os.environ.copy()
    env.update(
        QWEN_MODEL_PATH=str(model),
        ASCEND_RT_VISIBLE_DEVICES=devices,
        SERVING_PORT=str(port),
        VLLM_CACHE_ROOT=str(cache_dir.resolve()),
        PYTHON=sys.executable,
    )
    libraries = {
        "BETTERSCALE_GDN_LIBRARY": root / "patches/qwen_gdn/libbs_gdn.so",
        "BETTERSCALE_GDN_HOST_LIBRARY": root / "patches/qwen_gdn/libbs_gdn_host.so",
        "BETTERSCALE_FIA_LIBRARY": root / "patches/qwen_fia/libbs_fia.so",
    }
    for name, bundled in libraries.items():
        path = Path(env.get(name, str(bundled))).resolve()
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {name}: {path}; install the complete distribution"
            )
        env[name] = str(path)
    # serve.sh sets queue mode, AIV and LD_PRELOAD before the new Python process;
    # Worker independently verifies donor pins and all three artifact digests.
    return ["bash", str(root / "patches/qwen_gdn/serve.sh")], env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    qwen = sub.add_parser("serve-qwen", help="Qwen27 BF16 TP2 / FULL / APC / no MTP")
    qwen.add_argument("model", type=Path)
    qwen.add_argument(
        "--devices", required=True, help="two idle 910B2 devices, e.g. 0,1"
    )
    qwen.add_argument("--port", type=int, default=8000)
    qwen.add_argument(
        "--cache-dir", type=Path, default=Path.home() / ".cache/betterscale/qwen27"
    )
    args = parser.parse_args()
    if platform.system() != "Linux" or platform.machine() != "aarch64":
        parser.error(
            "This qualified native package requires Linux/aarch64 Ascend 910B2"
        )
    if not args.model.is_dir():
        parser.error("model must be an existing local Qwen27 checkpoint directory")
    try:
        command, env = prepare(args.model, args.devices, args.port, args.cache_dir)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    root = Path(__file__).parent / "patches"
    for variable, patch, key in (
        ("BETTERSCALE_GDN_LIBRARY", "qwen_gdn", "sha256"),
        ("BETTERSCALE_GDN_HOST_LIBRARY", "qwen_gdn", "host_sha256"),
        ("BETTERSCALE_FIA_LIBRARY", "qwen_fia", "sha256"),
    ):
        expected = json.loads((root / patch / "native.json").read_text())[key]
        if hashlib.sha256(Path(env[variable]).read_bytes()).hexdigest() != expected:
            parser.error(f"Unqualified native artifact: {variable}")
    os.execvpe(command[0], command, env)


if __name__ == "__main__":
    main()
