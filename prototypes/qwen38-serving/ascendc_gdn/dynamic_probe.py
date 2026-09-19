"""Reuse the proven full-pipeline oracle; replace only dynamic H/O with our fork."""

import os, runpy
from pathlib import Path
import torch, torch_npu
from vllm_ascend.utils import enable_custom_op

assert enable_custom_op()
torch.npu.set_device(0)
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

init_device_properties_triton()
from vllm_ascend.ops.triton.fla import chunk
from runtime import Kernels

engine = Kernels(os.environ["ASCENDC_GDN_LIB"], 512, 5, 12)


def forward_h(*, k, w, u, g, initial_state, cu_seqlens, chunk_indices, **kw):
    kh, wh, uh, gh = [x.transpose(1, 2).contiguous() for x in (k, w, u, g)]
    engine.launch(
        "h",
        [
            kh,
            wh,
            uh,
            gh,
            initial_state,
            cu_seqlens,
            chunk_indices,
            engine.h,
            engine.v,
            engine.final,
            engine.ws,
            engine.th,
        ],
    )
    # Keep temporary tensor allocations alive through the paired O call.
    engine.inputs = (kh, gh, cu_seqlens, chunk_indices, wh, uh)
    return engine.h, engine.v, engine.final


def forward_o(*, q, **kw):
    kh, gh, cu, indices, *_ = engine.inputs
    qh = q.transpose(1, 2).contiguous()
    engine.launch(
        "o",
        [qh, kh, engine.v, engine.h, gh, cu, indices, engine.o, engine.ws, engine.to],
    )
    return engine.o.transpose(1, 2).contiguous()


chunk.chunk_gated_delta_rule_fwd_h = forward_h
chunk.chunk_fwd_o = forward_o
runpy.run_path(
    str(Path(__file__).with_name("dynamic_gdn_probe.py")), run_name="__main__"
)
