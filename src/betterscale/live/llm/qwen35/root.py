"""Qwen35 model State root; numerical execution is not yet a serving route."""

from torch import nn

from betterscale.live import (
    ElasticStateCapacity,
    ExactStateCapacity,
    LiveModule,
    StateCapacityUnit,
    StateDomain,
)

from .state import AttentionState, Capacity, ContinuationState, GDNState, Geometry


class QwenStateRoot(LiveModule):
    """Model State root; numerical execution roots compose these leaves.

    No token history or max-context KV array per seat, no native blockpool and
    no implicit checkpoint/offload pool. execute width never multiplies State.
    """

    def __init__(self, geometry: Geometry, capacity: Capacity):
        super().__init__()
        self.geometry, self.capacity = geometry, capacity
        self.residents = StateDomain(ExactStateCapacity(capacity.resident_seats))
        self.pages = StateDomain(
            ElasticStateCapacity(
                StateCapacityUnit(minimum_units=capacity.execution_seats)
            )
            if capacity.token_pages is None
            else ExactStateCapacity(capacity.token_pages)
        )
        self.target = nn.ModuleDict(
            {
                str(i): GDNState(geometry, capacity, self.residents)
                if kind == "linear_attention"
                else AttentionState(geometry, capacity, self.pages)
                for i, kind in enumerate(geometry.layer_types)
            }
        )
        self.draft = AttentionState(geometry, capacity, self.pages)
        self.continuation = ContinuationState(geometry, capacity, self.residents)

    def attach_consumers(self, target, draft):
        if set(target) != set(self.target):
            raise ValueError("handoff must cover every target leaf exactly once")
        consumers = [target[key] for key in self.target] + [draft]
        if len({id(c) for c in consumers}) != len(consumers):
            raise ValueError("distinct State leaves cannot share a mutable cache field")
        leaves = [*self.target.values(), self.draft]
        # Validate the whole attachment before publishing any partial choice.
        for leaf, consumer in zip(leaves, consumers, strict=True):
            leaf.validate_consumer(consumer)
        for leaf, consumer in zip(leaves, consumers, strict=True):
            leaf.borrow_into(consumer)


def state_budget_bytes(geometry, capacity, page_ceiling):
    """Charge the declarations themselves, not a duplicated model-size formula.

    This temporary tree only declares State; no tensor storage or graph is made.
    The ceiling limits useful shared pages, not reservations for individual seats.
    """
    import math

    import torch

    from betterscale.live import LiveRuntime, live_runtime

    with live_runtime(LiveRuntime()):
        declaration = QwenStateRoot(geometry, capacity)
    return sum(
        math.prod(
            state.physical_shape(
                page_ceiling
                if state.domain is declaration.pages
                else capacity.resident_seats
            )
        )
        * torch.empty((), dtype=state.storage_dtype, device="meta").element_size()
        for _, state in declaration.named_states()
    )
