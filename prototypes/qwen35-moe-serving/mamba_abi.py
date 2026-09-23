"""Bridge pinned V1 Mamba postprocess arguments to Ascend's SD-only kernel.

vLLM752a3a50 added DS-row metadata and V2 request mapping; Ascend9bf964c still
installs the earlier SD/V1 kernel. Qwen uses SD convolution state and V1. Reject
other semantics rather than silently discarding a layout or indexing contract.
"""


class V1SDPostprocess:
    def __init__(self, kernel):
        self.kernel = kernel

    def __getitem__(self, grid):
        launch = self.kernel[grid]

        def run(*args, CONV_STATE_DIM_FIRST, **kwargs):
            if len(args) != 18 or CONV_STATE_DIM_FIRST or args[16] is not None:
                raise ValueError("Qwen Mamba ABI adapter requires V1, SD, no request mapping")
            if set(kwargs) != {"block_size", "COPY_BLOCK_SIZE"}:
                raise ValueError("Unqualified Mamba postprocess launch arguments")
            # Drop DS-only row count/stride (13,14) and absent V2 mapping (16).
            return launch(*args[:13], args[15], args[17], **kwargs)

        return run


def install():
    # Load the donor override first; otherwise it can overwrite this adapter.
    from vllm_ascend.patch.worker import patch_mamba_utils  # noqa: F401
    from vllm.v1.worker import mamba_utils

    original = mamba_utils.postprocess_mamba_fused_kernel
    if isinstance(original, V1SDPostprocess):
        return
    expected_tail = ["num_accepted_tokens_out_ptr", "num_reqs", "block_size", "COPY_BLOCK_SIZE"]
    if original.arg_names[-4:] != expected_tail or len(original.arg_names) != 17:
        raise RuntimeError("Unqualified Ascend Mamba postprocess ABI")
    mamba_utils.postprocess_mamba_fused_kernel = V1SDPostprocess(original)
