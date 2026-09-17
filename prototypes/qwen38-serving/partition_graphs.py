"""Bounded partition identity for native ACL graphs and FIA task resources.

The pinned worker submits model forwards serially. A scoped native resource bank
is selected for the whole host forward (including parameter update and replay),
then restored; no device synchronization or request-slot remapping is introduced.
This is not a concurrent-thread or multi-runner adapter.
"""

from contextlib import contextmanager
from dataclasses import dataclass

from vllm.forward_context import BatchDescriptor


@dataclass(frozen=True)
class PartitionDescriptor(BatchDescriptor):
    partition: tuple[int, ...] = ()


def descriptor_for(partition):
    return PartitionDescriptor(
        num_tokens=sum(partition), num_reqs=len(partition), partition=partition
    )


def install(signatures):
    from vllm.config import CUDAGraphMode
    from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

    original_init = CudagraphDispatcher.initialize_cudagraph_keys

    def initialize(self, *args, **kwargs):
        result = original_init(self, *args, **kwargs)
        keys = self.cudagraph_keys[CUDAGraphMode.FULL]
        totals = {sum(p) for p in signatures}
        keys.difference_update(k for k in tuple(keys) if k.num_tokens in totals)
        keys.update(descriptor_for(p) for p in signatures)
        return result

    original_warmup = NPUModelRunner._warmup_and_capture

    def warmup(self, desc, *args, **kwargs):
        assert getattr(self, "_capture_partition", None) is None
        self._capture_partition = getattr(desc, "partition", None)
        try:
            return original_warmup(self, desc, *args, **kwargs)
        finally:
            self._capture_partition = None

    CudagraphDispatcher.initialize_cudagraph_keys = initialize
    NPUModelRunner._warmup_and_capture = warmup


@contextmanager
def graph_resources(runner, descriptor):
    import vllm_ascend.compilation.acl_graph as acl

    partition = getattr(descriptor, "partition", None)
    if partition is None:
        yield
        return
    if not hasattr(runner, "_partition_resources"):
        runner._partition_resources = {}
    if partition not in runner._partition_resources:
        n = sum(partition)
        runner._partition_resources[partition] = acl.GraphParams(
            events={n: []}, workspaces={n: None}, handles={n: []}, attn_params={n: []}
        )
    assert not getattr(runner, "_partition_resources_active", False)
    runner._partition_resources_active = True
    previous = acl._graph_params
    acl._graph_params = runner._partition_resources[partition]
    try:
        yield
    finally:
        acl._graph_params = previous
        runner._partition_resources_active = False
