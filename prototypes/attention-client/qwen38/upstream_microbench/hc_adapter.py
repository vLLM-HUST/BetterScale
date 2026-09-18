"""Bounded HC leaf borrowing without moving work across the expert boundary.

The pinned upstream module is supplied by the caller; no installed package is
patched. Unlike replacing the entire HC transaction, this keeps injection-gate
projection/preparation before the attention/expert call, and injection afterward.
This remains a microbench candidate, not a qualified whole-model default.
"""

import torch
import triton
from livemodule.arch.ascend.llm.qwen38.residual import AscendQwen38ResidualBackend


class BorrowedHCLeaves(AscendQwen38ResidualBackend):
    def __init__(self, upstream):
        self.upstream = upstream

    def normalize(self, value, weight, group_size, eps):
        if (
            value.ndim == 2
            and value.shape[0] <= 32
            and self.upstream.can_run_norm(value, weight, group_size)
        ):
            return self.upstream.grouped_norm(value, weight, group_size, eps)
        return super().normalize(value, weight, group_size, eps)

    def mix(self, logits, normalized, count):
        if (
            normalized.ndim != 2
            or normalized.shape[0] > 32
            or normalized.dtype != torch.bfloat16
            or not normalized.is_contiguous()
            or not logits.is_contiguous()
            or count != 4
            or normalized.shape[1] != 10240
        ):
            return super().mix(logits, normalized, count)
        rows, hidden = normalized.shape[0], normalized.shape[1] // count
        out = torch.empty(
            (rows, hidden), device=normalized.device, dtype=normalized.dtype
        )
        self.upstream._mix[(rows, triton.cdiv(hidden, 256))](
            normalized,
            logits,
            out,
            count,
            hidden,
            triton.next_power_of_2(count),
            256,
            enable_fp_fusion=False,
        )
        return out
