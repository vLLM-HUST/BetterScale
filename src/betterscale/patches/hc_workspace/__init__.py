"""Select the qualified HC-pre host tiler before native custom-op registration."""

import hashlib
import json
import logging
from multiprocessing.util import Finalize
import os
from pathlib import Path
import platform
import shutil
import tempfile

log = logging.getLogger("vllm.betterscale.hc_workspace")
# The directory must outlive every replay and lazy tiling-library load. One
# process owns one complete vendor tree; no edits to the user's installation,
# cross-process cache invalidation, device allocations, or additional graph pool.
_envelope = None


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prepare_vendor(source, destination, library, manifest):
    for relative in manifest["changed"]:
        if _digest(source / relative) != manifest["original_library_sha256"]:
            raise RuntimeError("HC-pre native donor differs from the qualified build")
    if _digest(library) != manifest["candidate_library_sha256"]:
        raise RuntimeError("HC-pre packaged native library is missing or corrupted")
    # Dereference links: replacing a vendor's absolute symlink must NEVER write
    # back through to the donor. Preserve the complete API/config/kernel closure.
    shutil.copytree(source, destination, symlinks=False)
    for relative in manifest["changed"]:
        shutil.copyfile(library, destination / relative)


def install():
    """TP Worker.__init__, after pin/config admission, before NPUWorker.__init__.

    Native register_ascend_customop -> enable_custom_op -> bootstrap_custom_op_env
    uses _CUSTOM_OP_BASE_DIR. Merely prepending an OPP path is insufficient: its
    bootstrap could prepend the original vendor again and silently lose the fix.
    No device kernel or numerical operation changes; only HC-pre workspace size.
    """
    global _envelope
    if _envelope is not None:
        return
    import vllm_ascend.utils as native

    # Native camem imports the extension while importing NPUWorker. That loads
    # the unchanged API, not the host tiler; it is not a late-install signal.
    # Refuse actual custom-op enablement or a previously loaded vendor tiler.
    if native._CUSTOM_OP_ENABLED is not None or any(
        "custom_transformer/op_impl/ai_core/tbe/op_tiling/" in line
        for line in Path("/proc/self/maps").read_text().splitlines()
    ):
        raise RuntimeError(
            "HC-pre must be installed before native custom-op registration"
        )
    if platform.system() != "Linux" or platform.machine() != "aarch64":
        raise RuntimeError("HC-pre native package requires Linux aarch64")
    opp = os.environ.get("ASCEND_OPP_PATH")
    if (
        not opp
        or "Version=9.0.1" not in (Path(opp) / "version.info").read_text().splitlines()
    ):
        raise RuntimeError(
            "HC-pre native package requires the qualified CANN 9.0.1 OPP"
        )
    here = Path(__file__).parent
    manifest = json.loads((here / "native.json").read_text())
    suffix = Path("_cann_ops_custom/vendors/custom_transformer")
    source = Path(native._CUSTOM_OP_BASE_DIR) / suffix
    envelope = tempfile.TemporaryDirectory(prefix="betterscale-hc-")
    try:
        destination = Path(envelope.name) / suffix
        _prepare_vendor(
            source, destination, here / "libcust_opmaster_rt2.0.so", manifest
        )
    except BaseException:
        envelope.cleanup()
        raise
    # One vendor name, one coherent tree. Preserve unrelated custom vendors.
    paths = os.environ.get("ASCEND_CUSTOM_OPP_PATH", "").split(":")
    paths = [p for p in paths if p and Path(p).resolve() != source.resolve()]
    os.environ["ASCEND_CUSTOM_OPP_PATH"] = ":".join([str(destination), *paths])
    native._CUSTOM_OP_BASE_DIR = envelope.name
    _envelope = envelope
    # multiprocessing workers exit through os._exit: Python's weakref/atexit
    # cleanup alone is skipped. Register with their own orderly exit protocol.
    Finalize(None, envelope.cleanup, exitpriority=10)
    log.info("HC-pre workspace floor removed; private native vendor=%s", destination)
