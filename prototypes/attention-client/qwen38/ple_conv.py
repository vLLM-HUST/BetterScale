"""Keep PLE's existing depthwise convolution on graph-capable ACLNN dispatch."""

import torch
import torch_npu
from torch import nn


class GraphPLEConv(nn.Conv1d):
    def __init__(self, original):
        super().__init__(
            original.in_channels,
            original.out_channels,
            original.kernel_size,
            stride=original.stride,
            padding=original.padding,
            dilation=original.dilation,
            groups=original.groups,
            bias=original.bias is not None,
            padding_mode=original.padding_mode,
            device=original.weight.device,
            dtype=original.weight.dtype,
        )
        self.weight = original.weight
        self.bias = original.bias

    def forward(self, inputs):
        # Scope dispatch selection to this leaf. INT8 NZ expert/QSA paths retain
        # their internal-format setting; this does not reformat their weights.
        previous = torch_npu._C._npu_getOption("ALLOW_INTERNAL_FORMAT")
        if previous not in (b"enable", b"disable"):
            raise RuntimeError(
                "PLE graph dispatch requires an explicit internal-format policy"
            )
        try:
            torch.npu.config.allow_internal_format = False
            return super().forward(inputs)
        finally:
            torch_npu._C._npu_setOption({"ALLOW_INTERNAL_FORMAT": previous.decode()})
