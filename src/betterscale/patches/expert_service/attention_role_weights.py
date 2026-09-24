"""Construct Qwen35 attention ranks without ever allocating routed expert HBM.

Scoped native loader hooks retain parameter metadata and native loaded-name
validation. They do not alter attention/shared expert weights or routing.
The serving interception must be installed before the first model forward.
"""
from contextlib import contextmanager
import torch
from .model_geometry import GEOMETRY as G


@contextmanager
def attention_only_weights():
    from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import UnquantizedFusedMoEMethod
    from vllm_ascend.ops.fused_moe.fused_moe import AscendUnquantizedFusedMoEMethod
    from vllm.model_executor.models.qwen3_5 import Qwen3_5Model
    from vllm.model_executor.model_loader import utils as loader_utils
    # Ascend overrides postprocessing and holds an early-imported context alias.
    # Patch that exact consumer too, not just the upstream definition.
    from vllm_ascend.patch.worker import patch_process_weights_after_loading as ascend_loader
    original_ascend_context = ascend_loader.device_loading_context
    original_context = loader_utils.device_loading_context
    assert G.name == 'qwen35'
    original_create = UnquantizedFusedMoEMethod.create_weights
    original_process = AscendUnquantizedFusedMoEMethod.process_weights_after_loading
    from vllm.model_executor.models.qwen3_5_mtp import Qwen3_5MultiTokenPredictor
    loaders = [Qwen3_5Model] + ([Qwen3_5MultiTokenPredictor] if G.total_layers == 41 else [])
    original_loads = {cls: cls.load_weights for cls in loaders}
    created_layers = set()
    receipt = dict(created_layers=0, create_weight_calls=0, skipped_checkpoint_tensors=0,
                   skipped_checkpoint_bytes=0, startup_full_model_load_still_required=False)

    def shape_only(shape, dtype):
        return torch.nn.Parameter(torch.empty_strided(shape, (0,)*len(shape),
                                  dtype=dtype, device='cpu'), requires_grad=False)

    def create(method, layer, num_experts, hidden_size, intermediate_size_per_partition,
               params_dtype, **attrs):
        assert (num_experts, hidden_size, intermediate_size_per_partition) == (G.experts, G.hidden, G.inner)
        assert params_dtype == torch.bfloat16 and method.moe.is_act_and_mul and not method.moe.has_bias
        # Native pre-load orientation; each CPU placeholder has only2 storage bytes.
        for name, shape in [('w13_weight', (G.experts, 2*G.inner, G.hidden)),
                            ('w2_weight', (G.experts, G.hidden, G.inner))]:
            param = shape_only(shape, params_dtype)
            for key, value in attrs.items():
                setattr(param, key, value)
            param._expert_shape_only = True
            layer.register_parameter(name, param)
        # Ascend 0.23 initializes these same modules twice (upstream + backend).
        created_layers.add(layer)
        receipt['created_layers'] = len(created_layers)
        receipt['create_weight_calls'] += 1

    @contextmanager
    def loading_context(module, target_device):
        params = list(module.parameters())
        if any(getattr(p, '_expert_shape_only', False) for p in params):
            # Native CPU-offload handling would otherwise materialize every
            # logical element of our zero-stride metadata tensor on the NPU.
            assert all(p.device.type != 'cpu' or getattr(p, '_expert_shape_only', False) for p in params)
            yield module
        else:
            with original_context(module, target_device):
                yield module

    def process(method, layer):
        # Preserve the exact objects (including native aliases), change only
        # metadata to Ascend's post-load orientation. No CPU/NPU expert payload.
        for name in ('w13_weight', 'w2_weight'):
            param = getattr(layer, name)
            assert param.device.type == 'cpu' and param.untyped_storage().nbytes() == 2
            param.data = param.data.transpose(1, 2)

    def wrap_load(original_load):
        def load(model, weights):
            def attention_weights():
                for name, tensor in weights:
                    if '.mlp.experts.' in name:
                        receipt['skipped_checkpoint_tensors'] += 1
                        receipt['skipped_checkpoint_bytes'] += tensor.numel()*tensor.element_size()
                        continue
                    yield name, tensor
            loaded = original_load(model, attention_weights())
            loaded.update(name for name, _ in model.named_parameters() if '.mlp.experts.' in name)
            return loaded
        return load

    UnquantizedFusedMoEMethod.create_weights = create
    AscendUnquantizedFusedMoEMethod.process_weights_after_loading = process
    for cls, original in original_loads.items():
        cls.load_weights = wrap_load(original)
    loader_utils.device_loading_context = loading_context
    ascend_loader.device_loading_context = loading_context
    try:
        yield receipt
        assert receipt['created_layers'] == len(G.layers), receipt
        assert receipt['skipped_checkpoint_tensors'] == 80 + (3*G.experts if G.total_layers == 41 else 0), receipt
        assert receipt['skipped_checkpoint_bytes'] == len(G.layers)*G.experts*3*G.hidden*G.inner*2, receipt
    finally:
        UnquantizedFusedMoEMethod.create_weights = original_create
        AscendUnquantizedFusedMoEMethod.process_weights_after_loading = original_process
        for cls, original in original_loads.items():
            cls.load_weights = original
        loader_utils.device_loading_context = original_context
        ascend_loader.device_loading_context = original_ascend_context


def native_layer_reference(original, method, frame, layer):
    """One layer's native routed MoE shadow; never retain the whole expert model."""
    from dataclasses import replace
    import torch_npu
    from .checkpoint import Checkpoint
    gate, up, down = Checkpoint().experts(layer)
    w1 = torch_npu.npu_format_cast(torch.cat((gate, up), dim=1).transpose(1, 2).contiguous(), 29)
    w2 = torch_npu.npu_format_cast(down.transpose(1, 2).contiguous(), 29)
    reference = replace(frame, weights=replace(frame.weights, w1=w1, w2=w2))
    return original(method, reference).routed_out
