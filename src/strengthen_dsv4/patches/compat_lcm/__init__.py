"""K5/TP joint graph-bucket alignment; independent of target/draft graph patches."""

from vllm.config import CompilationConfig

_original_adjust_sizes = CompilationConfig.adjust_cudagraph_sizes_for_spec_decode
_installed = False


def _adjust_joint_alignment(self, uniform_decode_query_len, tensor_parallel_size):
    import math

    alignment = uniform_decode_query_len
    if self.pass_config.enable_sp:
        alignment = math.lcm(alignment, tensor_parallel_size)
    return _original_adjust_sizes(self, alignment, tensor_parallel_size)


def install():
    """Install before the runner computes capture buckets; does not change K."""
    global _installed
    if not _installed:
        CompilationConfig.adjust_cudagraph_sizes_for_spec_decode = (
            _adjust_joint_alignment
        )
        _installed = True
