"""Qwen35 baseline entry; install its capture policy before config finalization.

This is an alias of the original BetterScale Worker, not a new execution loop.
The packaged State-backed launcher selects it before native graph initialization.
"""

from vllm.config import CUDAGraphMode
from vllm.config.compilation import CompilationConfig


def _install_capture_policy():
    original = CompilationConfig.adjust_cudagraph_sizes_for_spec_decode
    if getattr(original, "_betterscale_qwen35_full", False):
        return

    def adjust(self, *args, **kwargs):
        # Qualified mixed FULL keys need not be multiples of the MTP width.
        if self.cudagraph_mode == CUDAGraphMode.FULL:
            return
        return original(self, *args, **kwargs)

    adjust._betterscale_qwen35_full = True
    CompilationConfig.adjust_cudagraph_sizes_for_spec_decode = adjust


_install_capture_policy()
from .worker import Worker  # noqa: E402,F401
