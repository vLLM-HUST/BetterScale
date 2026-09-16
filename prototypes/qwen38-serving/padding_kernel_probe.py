"""Bounded kernel isolation: real512 versus zero-padded513/576/1024.

No model load, no timing claim. PCP1 and num_decodes0 stand in for the enclosing
model forward context; native Ascend kernels and Qwen TP2 local head shapes stay.
"""

import json
import os
from pathlib import Path
from types import SimpleNamespace as NS
import torch
import torch_npu
from vllm_ascend.utils import enable_custom_op

assert enable_custom_op()
torch.npu.set_device(0)
torch.manual_seed(17)
from vllm_ascend.ops.triton.fla import chunk
from vllm_ascend.ops.gdn_attn_builder import _build_non_spec_chunked_prefill_metadata

chunk.get_forward_context = lambda: NS(attn_metadata=NS(num_decodes=0))
chunk.get_pcp_group = lambda: NS(world_size=1)
builder = NS(
    vllm_config=NS(
        model_config=NS(hf_text_config=NS(linear_num_value_heads=48)),
        parallel_config=NS(tensor_parallel_size=2),
    )
)
root = Path(os.environ["CAPSULE"])
results = []


def compare(a, b):
    return dict(
        close=bool(torch.allclose(a, b, atol=0.01, rtol=0.01)),
        max_abs=float((a.float() - b.float()).abs().max().item()),
        finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
    )


with torch.inference_mode():
    for initial in [False, True]:
        x = torch.randn(1024, 5120, device="npu", dtype=torch.bfloat16)
        w = torch.randn(4, 5120, device="npu", dtype=torch.bfloat16)
        cache = torch.randn(8, 3, 5120, device="npu", dtype=torch.bfloat16)
        cu = torch.tensor([0, 512], device="npu", dtype=torch.int32)
        slots = torch.tensor([[3, 4, 5]], device="npu", dtype=torch.int32)
        mode = torch.tensor([initial], device="npu", dtype=torch.bool)
        conv_reference = None
        for length in [512, 513, 576, 1024]:
            state = cache.clone()
            out = torch.empty_like(x[:length])
            torch.ops._C_ascend.npu_causal_conv1d_custom(
                out,
                x[:length],
                w,
                conv_state=state,
                bias_opt=None,
                query_start_loc_opt=cu,
                cache_indices_opt=slots,
                initial_state_mode_opt=mode,
                num_accepted_tokens_opt=None,
                activation_mode=1,
                pad_slot_id=-1,
                run_mode=0,
            )
            torch.npu.synchronize()
            if conv_reference is None:
                conv_reference = (out.clone(), state.clone())
            results.append(
                dict(
                    kind="conv",
                    initial=initial,
                    length=length,
                    output=compare(out[:512], conv_reference[0]),
                    state=compare(state, conv_reference[1]),
                )
            )
        values = dict(
            q=torch.randn(1, 512, 8, 128, device="npu", dtype=torch.bfloat16),
            k=torch.randn(1, 512, 8, 128, device="npu", dtype=torch.bfloat16),
            v=torch.randn(1, 512, 24, 128, device="npu", dtype=torch.bfloat16),
            g=-torch.rand(1, 512, 24, device="npu", dtype=torch.float32) * 0.1,
            beta=torch.rand(1, 512, 24, device="npu", dtype=torch.bfloat16),
        )
        state = (
            torch.randn(1, 24, 128, 128, device="npu", dtype=torch.float32)
            if initial
            else torch.zeros(1, 24, 128, 128, device="npu", dtype=torch.float32)
        )
        reference = None
        for length in [512, 513, 576, 1024]:
            inputs = {}
            for name, v in values.items():
                z = torch.zeros((1, length, *v.shape[2:]), device="npu", dtype=v.dtype)
                z[:, :512].copy_(v)
                inputs[name] = z
            cpu = torch.tensor([0, length], dtype=torch.int32)
            meta = _build_non_spec_chunked_prefill_metadata(
                builder, cpu, torch.device("npu")
            )
            out, after = chunk.chunk_gated_delta_rule(
                **inputs,
                initial_state=state.clone(),
                output_final_state=True,
                cu_seqlens=cpu.to("npu"),
                prebuilt_meta=meta,
                head_first=False,
                use_qk_l2norm_in_kernel=True,
            )
            torch.npu.synchronize()
            if reference is None:
                reference = (out.clone(), after.clone())
            results.append(
                dict(
                    kind="recurrence",
                    initial=initial,
                    length=length,
                    output=compare(out[:, :512], reference[0]),
                    state=compare(after, reference[1]),
                )
            )
            (root / "receipt.json").write_text(
                json.dumps(dict(status="RUNNING", results=results), indent=2)
            )
(root / "receipt.json").write_text(
    json.dumps(dict(status="COMPLETE", results=results), indent=2)
)
print(json.dumps(results), flush=True)
