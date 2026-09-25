"""Launch the qualified Qwen service with packaged native artifacts."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys


def prepare(
    model,
    devices,
    port,
    cache_dir,
    *,
    runtime="native",
    context_tokens=512,
    resident_seats=20,
    token_pages=64,
    distributed_port=29535,
):
    """Prepare a new process; never preload CANN into this interpreter."""
    if runtime == "live":
        return prepare_live(
            model,
            devices,
            port,
            cache_dir,
            context_tokens,
            resident_seats,
            token_pages,
            distributed_port,
        )
    if runtime != "native":
        raise ValueError(f"unknown runtime: {runtime}")
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


def prepare_live(
    model,
    devices,
    port,
    cache_dir,
    context_tokens,
    resident_seats,
    token_pages,
    distributed_port,
):
    """Inert launch admission: no torch, donor, device or native artifact imports."""
    selected = devices.split(",")
    if (
        len(selected) not in (1, 2)
        or len(set(selected)) != len(selected)
        or any(not d.isdigit() or int(d) not in range(8) for d in selected)
    ):
        raise ValueError("live requires one or two distinct 910B2 device IDs")
    if not (
        1 <= port <= 65535
        and 1 <= distributed_port <= 65535
        and port != distributed_port
    ):
        raise ValueError("HTTP and distributed ports must be distinct and in 1..65535")
    if not (3 <= context_tokens <= 4096 and resident_seats > 0 and token_pages > 0):
        raise ValueError("invalid live context or State capacity")
    config = json.loads((model / "config.json").read_text())
    text = config.get("text_config", config)
    envelope = (
        text.get("model_type"),
        text.get("num_hidden_layers"),
        text.get("hidden_size"),
        len(selected),
    )
    if envelope not in (
        ("qwen3_5_text", 24, 1024, 1),
        ("qwen3_5_moe_text", 40, 2048, 2),
    ):
        raise ValueError("live supports Qwen3.5-0.8B TP1 or 35B-A3B TP2 only")
    env = os.environ.copy()
    env.update(
        ASCEND_RT_VISIBLE_DEVICES=devices,
        PYTHON=sys.executable,
        VLLM_CACHE_ROOT=str((cache_dir / "live").resolve()),
        BETTERSCALE_LIVE_TP=str(len(selected)),
        BETTERSCALE_LIVE_DISTRIBUTED_PORT=str(distributed_port),
    )
    return [
        "bash",
        str(Path(__file__).parent / "live/llm/qwen35/serve.sh"),
        str(model),
        "--port",
        str(port),
        "--context-tokens",
        str(context_tokens),
        "--resident-seats",
        str(resident_seats),
        "--token-pages",
        str(token_pages),
    ], env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    qwen = sub.add_parser(
        "serve-qwen", help="qualified native Qwen27 or experimental live Qwen35"
    )
    qwen.add_argument("model", type=Path)
    qwen.add_argument(
        "--runtime",
        choices=("native", "live"),
        default="native",
        help="native (default); live owns Qwen35 State and full-model graphs",
    )
    qwen.add_argument(
        "--devices",
        required=True,
        help="idle 910B2 IDs; native/35B live: two, 0.8B live: one",
    )
    qwen.add_argument("--port", type=int, default=8000)
    qwen.add_argument(
        "--cache-dir", type=Path, default=Path.home() / ".cache/betterscale/qwen27"
    )
    qwen.add_argument("--live-context-tokens", type=int, default=512)
    qwen.add_argument("--live-resident-seats", type=int, default=20)
    qwen.add_argument("--live-token-pages", type=int, default=64)
    qwen.add_argument("--live-distributed-port", type=int, default=29535)
    args = parser.parse_args()
    if platform.system() != "Linux" or platform.machine() != "aarch64":
        parser.error(
            "This qualified native package requires Linux/aarch64 Ascend 910B2"
        )
    if not args.model.is_dir():
        parser.error(
            "model must be an existing supported local Qwen checkpoint directory"
        )
    try:
        command, env = prepare(
            args.model,
            args.devices,
            args.port,
            args.cache_dir,
            runtime=args.runtime,
            context_tokens=args.live_context_tokens,
            resident_seats=args.live_resident_seats,
            token_pages=args.live_token_pages,
            distributed_port=args.live_distributed_port,
        )
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    if args.runtime == "live":
        os.execvpe(command[0], command, env)
        return
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
