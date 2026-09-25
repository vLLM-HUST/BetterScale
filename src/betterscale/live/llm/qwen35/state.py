"""Experimental Qwen35 State declarations, realized by the BetterScale-owned live root.

No allocator, native runner, cache manager, model math or request scheduler lives
here. The numerical consumer may borrow a generation's tensors after activation.
"""

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn
from betterscale.live import (
    LiveModule,
    StateTensor,
)


def positive(name, value):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True)
class Geometry:
    layer_types: tuple[str, ...]
    kv_heads: int
    attention_head_dim: int
    gdn_key_heads: int
    gdn_value_heads: int
    gdn_key_dim: int
    gdn_value_dim: int
    conv_kernel: int
    hidden_size: int
    draft_layers: int = 1

    def __post_init__(self):
        for name in (
            "kv_heads",
            "attention_head_dim",
            "gdn_key_heads",
            "gdn_value_heads",
            "gdn_key_dim",
            "gdn_value_dim",
            "conv_kernel",
            "hidden_size",
            "draft_layers",
        ):
            positive(name, getattr(self, name))
        if not self.layer_types or set(self.layer_types) != {
            "linear_attention",
            "full_attention",
        }:
            raise ValueError("this vertical requires the Qwen35 hybrid layer sequence")
        if self.gdn_value_heads % self.gdn_key_heads:
            raise ValueError("GDN value heads must be divisible by key heads")
        if self.conv_kernel < 2 or self.draft_layers != 1:
            raise ValueError(
                "this vertical admits one MTP layer and convolution history"
            )

    @classmethod
    def from_config(cls, config: Mapping, *, tensor_parallel_size=1):
        """Explicit dense TP1 / MoE TP2 geometry; no topology/device discovery."""
        text = config.get("text_config", config)
        if (text.get("model_type"), tensor_parallel_size) not in (
            ("qwen3_5_text", 1),
            ("qwen3_5_moe_text", 2),
        ):
            raise ValueError("live model geometry supports dense TP1 or MoE TP2 only")
        for field in (
            "num_key_value_heads",
            "linear_num_key_heads",
            "linear_num_value_heads",
        ):
            if text[field] % tensor_parallel_size:
                raise ValueError(f"{field} must divide evenly across TP")
        if text.get("mamba_ssm_dtype", "float32") != "float32":
            raise ValueError("first gate requires FP32 recurrent State")
        layers = tuple(text["layer_types"])
        if len(layers) != text["num_hidden_layers"]:
            raise ValueError("layer_types and num_hidden_layers disagree")
        return cls(
            layers,
            text["num_key_value_heads"] // tensor_parallel_size,
            text["head_dim"],
            text["linear_num_key_heads"] // tensor_parallel_size,
            text["linear_num_value_heads"] // tensor_parallel_size,
            text["linear_key_head_dim"],
            text["linear_value_head_dim"],
            text["linear_conv_kernel_dim"],
            text["hidden_size"],
            text["mtp_num_hidden_layers"],
        )

    @property
    def conv_channels(self):
        return (
            2 * self.gdn_key_heads * self.gdn_key_dim
            + self.gdn_value_heads * self.gdn_value_dim
        )


@dataclass(frozen=True)
class Capacity:
    execution_seats: int
    resident_seats: int
    page_tokens: int = 128
    token_pages: int | None = None
    speculative_tokens: int = 2

    def __post_init__(self):
        for name in (
            "execution_seats",
            "resident_seats",
            "page_tokens",
            "speculative_tokens",
        ):
            positive(name, getattr(self, name))
        if self.execution_seats > self.resident_seats:
            raise ValueError("resident seats must cover the execution envelope")
        if self.token_pages is not None:
            positive("token_pages", self.token_pages)
        if self.speculative_tokens != 2:
            raise ValueError("initial consumer contract is MTP2")


class NumericalState(LiveModule):
    """One numerical leaf owns declarations; an optional old leaf only borrows.

    This is the binding seam, not authority to run an old native scheduler or
    capture. All consumers must be quiescent before the containing root closes.
    """

    def __init__(self):
        super().__init__()
        object.__setattr__(self, "_consumer", None)
        self._published = None

    def validate_consumer(self, consumer):
        if self._consumer is not None or any(
            s.is_bound for _, s in self.named_states()
        ):
            raise RuntimeError(
                "consumer must be attached once, before State activation"
            )
        if not isinstance(consumer, nn.Module):
            raise TypeError("numerical consumer must be an explicit torch module")
        cache = getattr(consumer, "kv_cache", None)
        if cache is not None and not (
            isinstance(cache, (tuple, list)) and len(cache) == 0
        ):
            raise ValueError("cannot adopt or overwrite preallocated native KV")
        if getattr(consumer, "state_binding_abi", None) != self.binding_abi:
            raise ValueError("consumer has not declared this State addressing ABI")

    def borrow_into(self, consumer):
        self.validate_consumer(consumer)
        # Do not register the borrowed donor as a child or collect its State.
        object.__setattr__(self, "_consumer", consumer)

    def numerical_tensors(self):
        raise NotImplementedError

    def _initialize_live_generation(self):
        if self._consumer is None:
            return
        cache = getattr(self._consumer, "kv_cache", None)
        if cache is not None and not (
            isinstance(cache, (tuple, list)) and len(cache) == 0
        ):
            raise RuntimeError("native consumer allocated KV before State handoff")
        value = self.numerical_tensors()
        # One publication after all values resolve. No allocation in this hook.
        self._consumer.kv_cache = value
        self._published = value

    def _release_live_generation(self):
        if self._published is not None:
            if self._consumer.kv_cache is not self._published:
                raise RuntimeError(
                    "foreign writer replaced generation-owned KV binding"
                )
            self._consumer.kv_cache = []
            self._published = None

    def _rebind_live_state(self):
        # Fitting is not admitted with attached consumers: silently changing
        # tensors would leave native pointer tables/captures on retired storage.
        if self._consumer is not None:
            raise RuntimeError(
                "consumer rebind requires explicit descriptor retirement"
            )


class AttentionState(NumericalState):
    binding_abi = "qwen35-paged-fa-v1"

    def __init__(self, geometry, capacity, domain):
        super().__init__()
        for name in ("key", "value"):
            self.register_state(
                name,
                StateTensor(
                    role=f"attention-{name}",
                    requirement="BF16 token pages",
                    block_shape=(
                        capacity.page_tokens,
                        geometry.kv_heads,
                        geometry.attention_head_dim,
                    ),
                    storage_dtype=torch.bfloat16,
                    domain=domain,
                ),
            )

    def numerical_tensors(self):
        return self.key.tensor, self.value.tensor


class GDNState(NumericalState):
    binding_abi = "qwen35-seat-gdn-kv-v1"

    def __init__(self, geometry, capacity, domain):
        super().__init__()
        self.candidates = capacity.speculative_tokens + 1
        self.register_state(
            "conv",
            StateTensor(
                role="gdn-convolution",
                requirement="Ascend SD extended history",
                block_shape=(
                    geometry.conv_kernel - 1 + capacity.speculative_tokens,
                    geometry.conv_channels,
                ),
                storage_dtype=torch.bfloat16,
                domain=domain,
            ),
        )
        self.register_state(
            "recurrent",
            StateTensor(
                role="gdn-recurrent",
                requirement="owned K-V candidate matrix",
                block_shape=(
                    geometry.gdn_value_heads,
                    geometry.gdn_key_dim,
                    geometry.gdn_value_dim,
                ),
                storage_dtype=torch.float32,
                domain=domain,
                physical_blocks_per_logical_block=self.candidates,
            ),
        )

    def numerical_tensors(self):
        # Conv indexes resident directly; recurrence indexes resident*(K+1)+candidate.
        # Never pass this pair to a consumer that assumes a common block index.
        return self.conv.tensor, self.recurrent.tensor


class ContinuationState(LiveModule):
    def __init__(self, geometry, capacity, domain):
        super().__init__()
        self.register_state(
            "anchor_hidden",
            StateTensor(
                role="target-hidden-at-committed-boundary",
                requirement="bounded MTP seed; valid only with the committed anchor",
                block_shape=(geometry.hidden_size,),
                storage_dtype=torch.bfloat16,
                domain=domain,
            ),
        )
        for name in ("resident_epoch", "target_cursor", "draft_cursor", "anchor_token"):
            self.register_state(
                name,
                StateTensor(
                    role=name,
                    requirement="resident continuation",
                    block_shape=(),
                    storage_dtype=torch.int64,
                    domain=domain,
                ),
            )
        self.register_state(
            "selection",
            StateTensor(
                role="accepted-input-count",
                requirement="one-based GDN candidate selection",
                block_shape=(),
                storage_dtype=torch.int32,
                domain=domain,
            ),
        )
        self.register_state(
            "proposal",
            StateTensor(
                role="next-MTP-proposal",
                requirement="resident continuation",
                block_shape=(capacity.speculative_tokens,),
                storage_dtype=torch.int64,
                domain=domain,
            ),
        )

    def _initialize_live_generation(self):
        self.selection.tensor.fill_(1)
        self.anchor_token.tensor.fill_(-1)
        self.anchor_hidden.tensor.zero_()
        self.proposal.tensor.fill_(-1)
        for name in ("resident_epoch", "target_cursor", "draft_cursor"):
            getattr(self, name).tensor.zero_()

    def _rebind_live_state(self):
        self._initialize_live_generation()
