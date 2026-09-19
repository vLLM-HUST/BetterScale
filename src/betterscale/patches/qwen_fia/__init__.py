"""Qwen27-only native wave FIA planning; separate from DSV4 and native Qwen."""

import ctypes
import hashlib
import json
import os
from pathlib import Path


def check_library():
    if os.environ.get("TASK_QUEUE_ENABLE") != "0":
        raise ValueError("Wave FIA requires TASK_QUEUE_ENABLE=0")
    path = Path(os.environ.get("BETTERSCALE_FIA_LIBRARY", ""))
    contract = json.loads(Path(__file__).with_name("native.json").read_text())
    if (
        not path.is_file()
        or hashlib.sha256(path.read_bytes()).hexdigest() != contract["sha256"]
    ):
        raise ValueError(
            "BETTERSCALE_FIA_LIBRARY must name the qualified native planner"
        )
    # Setting LD_PRELOAD after startup or merely dlopen'ing the library is not
    # sufficient: interception must already be bound process-wide before CANN.
    library = ctypes.CDLL(str(path.resolve()))
    try:
        process_entry = ctypes.CDLL(None).plan_native_queries
    except AttributeError as exc:
        raise RuntimeError(
            "Preload BETTERSCALE_FIA_LIBRARY before process startup"
        ) from exc
    if (
        ctypes.cast(process_entry, ctypes.c_void_p).value
        != ctypes.cast(library.plan_native_queries, ctypes.c_void_p).value
    ):
        raise RuntimeError("FIA planner does not match the preloaded process binding")
    return library


def install():
    library = check_library()
    from .wave import install as install_wave

    install_wave(library)
