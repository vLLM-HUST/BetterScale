"""Declared State for the baseline's actual attention consumers, not new math.

The caller owns runner metadata/address installation and graph retirement.
This module only declares, binds and accounts for the storage it lends them.
"""

import re

from betterscale.live import StateTensor
from betterscale.live.llm.qwen35.root import QwenStateRoot
from betterscale.live.llm.qwen35.state import GDNState, empty_cache
import torch


class BaselineStateRoot(QwenStateRoot):
    def __init__(self, geometry, capacity, *, consumers, draft_names):
        super().__init__(geometry, capacity)
        # Prefill writes the selected recurrent candidate in place, but resets
        # the small convolution history. Its selector is therefore independent.
        self.register_state(
            "conv_selection",
            StateTensor(
                role="convolution-accepted-window",
                requirement="one-based extended-history selection",
                block_shape=(),
                storage_dtype=torch.int32,
                domain=self.residents,
            ),
        )
        draft_names = set(draft_names)
        if len(draft_names) != 1 or not draft_names <= consumers.keys():
            raise ValueError("baseline requires exactly one identified draft FA leaf")
        target = {}
        self.layer_states = {}
        for name, consumer in consumers.items():
            if name in draft_names:
                leaf = self.draft
            else:
                match = re.search(r"(?:^|\.)layers\.(\d+)\.", name)
                if match is None or match[1] not in self.target or match[1] in target:
                    raise ValueError(f"unrecognized or duplicate target State: {name}")
                target[match[1]] = consumer
                leaf = self.target[match[1]]
            if not empty_cache(getattr(consumer, "kv_cache", None)):
                raise ValueError(f"native cache already allocated: {name}")
            self.layer_states[name] = leaf
        if set(target) != set(self.target):
            raise ValueError("baseline State must cover every target layer")
        # The binding ABI must be explicitly accepted by the addressing adapter.
        # Do not stamp it onto arbitrary native modules merely to pass validation.
        self.attach_consumers(target, consumers[next(iter(draft_names))])

    def _initialize_live_generation(self):
        # New resident GDN reads must never observe allocator garbage. FA pages
        # are initialized by their normal token writers and masked by lengths.
        for leaf in self.target.values():
            if isinstance(leaf, GDNState):
                for tensor in leaf.numerical_tensors():
                    tensor.zero_()
        self.conv_selection.tensor.fill_(1)

    def _rebind_live_state(self):
        raise RuntimeError("baseline State rebind requires retirement of runner graphs")

    def kv_caches(self):
        self._require_active("borrow native cache tensors from")
        return {name: leaf._published for name, leaf in self.layer_states.items()}
