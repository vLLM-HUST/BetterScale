"""Captured device metadata program for the explicit stable-K5 producer.

No host H2D/readback is allowed in this stage. Host-owned query geometry is
already published by the runner; exact lengths/positions remain device State.
Native CPU tiling maxima are conservatively fixed at the admitted model limit.
The native builder remains the independent oracle and fallback.
"""

from copy import copy
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten


class DeviceOnly(TorchDispatchMode):
    def __init__(self, device_type="npu"):
        super().__init__()
        self.device_type = device_type

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        tensors = [
            v for v in tree_flatten((args, kwargs))[0] if isinstance(v, torch.Tensor)
        ]
        destination = kwargs.get("device")
        target = torch.device(destination).type if destination is not None else None
        devices = {v.device.type for v in tensors}
        assert not (
            target == self.device_type and "cpu" in devices
        ), f"Unbanked metadata ingress: {func}"
        assert not (
            target == "cpu" and self.device_type in devices
        ), f"Metadata readback: {func}"
        assert not (
            func._schema.name == "aten::_local_scalar_dense"
            and self.device_type in devices
        ), f"Metadata scalar readback: {func}"
        if func._schema.name == "aten::copy_" and len(args) > 1:
            assert (
                args[0].device.type == args[1].device.type
            ), f"Unbanked metadata copy: {func}"
        return func(*args, **kwargs)


class DecodeMetadata:
    def __init__(self, producer):
        self.producer = producer
        self.r = r = producer.r
        self.native = r._build_attention_metadata
        self.pool = torch.npu.graph_pool_handle()
        self.entries = {}
        self.replays = 0
        r._build_attention_metadata = self.build

    def capture(self, **kwargs):
        """Startup only, with the same native fields/CPU carrier as serving."""
        r = self.r
        n = kwargs["num_reqs"]
        key = (
            n,
            kwargs["num_reqs_padded"],
            kwargs["num_tokens_padded"],
            r.optimistic_seq_lens_cpu.data_ptr(),
        )
        assert key not in self.entries
        upper = r.optimistic_seq_lens_cpu.clone()
        graph = torch.npu.NPUGraph()
        try:
            r.optimistic_seq_lens_cpu[:n].fill_(r.max_model_len)
            with DeviceOnly():
                self.native(**kwargs)
            torch.npu.synchronize()
            with torch.npu.graph(graph, pool=self.pool), DeviceOnly():
                output = self.native(**kwargs)
        finally:
            r.optimistic_seq_lens_cpu.copy_(upper)
        self.entries[key] = graph, output

    def build(self, *args, **kwargs):
        r = self.r
        # The normal execute_model caller uses keywords. Dummy/capture and
        # non-admitted calls retain native semantics without touching the cache.
        if (
            args
            or self.producer.active is None
            or not r._cross_step_bounds.admitted
            or kwargs.get("for_cudagraph_capture")
            or kwargs.get("ubatch_slices") is not None
        ):
            return self.native(*args, **kwargs)
        n = kwargs["num_reqs"]
        nr = kwargs.get("num_reqs_padded") or n
        nt = kwargs.get("num_tokens_padded") or kwargs["num_tokens"]
        # Local K5 does not imply a small GLOBAL DP wave: another rank can
        # force prefill-sized collective padding. That target uses the native
        # fallback, not DecodePair. Do not lazily capture every large metadata
        # shape on the live serving critical path either.
        if nt > r.max_num_reqs * 6:
            return self.native(*args, **kwargs)
        assert kwargs["num_tokens"] == n * 6 and kwargs["max_query_len"] == 6
        assert kwargs["use_spec_decode"] and not r.cache_config.kv_sharing_fast_prefill
        assert not r.model_config.enable_return_routed_experts
        # CPU carrier identity matters: native source slots alternate even when
        # an intervening prefill/dummy wave is not admitted by this producer.
        key = (n, nr, nt, r.optimistic_seq_lens_cpu.data_ptr())
        r._cross_step_bounds.build_args = (args, kwargs)
        if key not in self.entries:
            # A new runtime shape is a fallback, never permission to stop the
            # cluster and capture. Startup enumerates the supported envelope.
            return self.native(*args, **kwargs)
        graph, output = self.entries[key]
        graph.replay()  # capture is not a committed first invocation
        self.replays += 1
        # DSpark receives a fresh envelope: it can mutate its own per-invocation
        # fields without corrupting the cached construction template.
        metadata, common = output
        common = copy(common) if common is not None else None
        if common is not None:
            common.max_seq_len = int(r.optimistic_seq_lens_cpu[:n].max())
        # Prefill/native calls may have replaced DSpark's per-group views.
        for gid, table in enumerate(r.input_batch.block_table.block_tables):
            r.drafter.set_per_group_attn_metadata(
                gid, table.get_device_tensor()[:nr], table.slot_mapping.gpu[:nt]
            )
        return metadata, common
