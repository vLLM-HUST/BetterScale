"""Prototype only: one-time NZ storage for the 64 unquantized gate/up weights."""
def apply(model):
    import torch
    import torch_npu

    modules = [(name, layer) for name, layer in model.named_modules()
               if name.endswith('.mlp.gate_up_proj')]
    assert len(modules) == 64, [name for name, _ in modules]
    torch.npu.config.allow_internal_format = True
    with torch.inference_mode():
        for name, layer in modules:
            w = layer.weight
            assert w.dtype == torch.bfloat16 and tuple(w.shape) == (17408, 5120), (name, w.shape, w.dtype)
            packed = torch_npu.npu_format_cast(w, 29)
            assert torch.equal(torch_npu.npu_format_cast(packed, 2), w), name
            w.data = packed
    torch.npu.synchronize()
    print('PROTOTYPE_NZ_GATE_UP PASS', len(modules), flush=True)
