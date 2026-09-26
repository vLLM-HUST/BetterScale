"""Incremental real-weight execution over the declared model State root.

This first eager vertical is synchronous and consumes explicit physical page
addresses. No native Worker, KV allocator or scheduler participates. Graphs and
resident admission are separate integration gates, not implicit fallbacks.
"""

from contextlib import nullcontext
from dataclasses import replace

import torch

from . import numerics
from .residents import ResidentTable
from .root import QwenStateRoot


class QwenExecutionRoot(QwenStateRoot):
    def __init__(
        self, geometry, capacity, target_model, draft_model, *, greedy_only=False
    ):
        super().__init__(geometry, capacity)
        self.greedy_only = greedy_only
        self.target_model = target_model
        self.draft_model = draft_model
        if len(target_model.model.layers) != len(geometry.layer_types):
            raise ValueError("weight model and State geometry disagree")

    def _initialize_live_generation(self):
        for state in self.target.values():
            for tensor in state.numerical_tensors():
                tensor.zero_()
        for tensor in self.draft.numerical_tensors():
            tensor.zero_()
        self.residents_table = ResidentTable(
            replace(self.capacity, token_pages=self.pages.capacity),
            clear_seat=self.clear_seat,
        )

    def _rebind_live_state(self):
        self._initialize_live_generation()

    @torch.inference_mode()
    def clear_seat(self, seat):
        self._require_active("clear resident on")
        epoch = int(self.continuation.resident_epoch.tensor[seat].item()) + 1
        self.clear_state_blocks((seat,), domain=self.residents)
        self.continuation.resident_epoch.tensor[seat] = epoch
        self.continuation.anchor_token.tensor[seat] = -1
        self.continuation.selection.tensor[seat] = 1
        self.continuation.proposal.tensor[seat].fill_(-1)
        torch.npu.synchronize(self.live_device)

    @torch.inference_mode()
    def generate(self, prompt, max_new_tokens, **kwargs):
        from .generation import generate

        # This eager entry serializes close and callers until all writers drain.
        # It is a correctness vertical, not a concurrent-serving scheduler.
        with self._live_lock:
            self._require_active("generate on")
            return generate(self, prompt, max_new_tokens, **kwargs)

    @torch.inference_mode()
    def target_step(self, tokens, *, seat, position, slots, accepted=1):
        self._require_active("execute target on")
        if not (
            0 <= seat < self.capacity.resident_seats
            and 1 <= len(tokens) <= 3
            and 1 <= accepted <= 3
            and position >= 0
        ):
            raise ValueError("invalid resident/candidate/short-wave geometry")
        if len(slots) != position + len(tokens) or len(set(slots)) != len(slots):
            raise ValueError(
                "explicit unique KV slots must cover the complete prefix and wave"
            )
        ids = torch.tensor(tokens, dtype=torch.long, device=self.live_device)
        pos = torch.arange(position, position + len(tokens), device=self.live_device)
        # Text mRoPE: temporal/height/width positions coincide.
        positions = pos.expand(3, -1).contiguous()
        read = torch.tensor(slots, dtype=torch.long, device=self.live_device)
        write = read[-len(tokens) :]
        hidden, logits = self._target_forward(
            ids, positions, write, read, seat, accepted
        )
        # Eager entry returns only after writers finish. Captured invocation
        # ownership will replace this synchronous boundary, not coexist with it.
        torch.npu.synchronize(self.live_device)
        return hidden, logits

    @torch.inference_mode()
    def draft_step(self, tokens, hidden_seed, *, position, slots):
        """Shifted IDs pair with target (or previous draft) hidden at same position.

        The initial MTP pass leaves target positions unchanged; only subsequent
        recursive proposal steps advance them. Cache validity is the caller's
        committed-prefix boundary, not the last speculative write position.
        """
        self._require_active("execute draft on")
        if not (1 <= len(tokens) <= 3 and position >= 0):
            raise ValueError("invalid draft wave")
        if len(slots) != position + len(tokens) or len(set(slots)) != len(slots):
            raise ValueError("draft slots must cover its valid prefix and wave")
        if hidden_seed.shape != (len(tokens), self.geometry.hidden_size):
            raise ValueError("MTP needs one hidden seed per shifted input")
        ids = torch.tensor(tokens, dtype=torch.long, device=self.live_device)
        pos = torch.arange(position, position + len(tokens), device=self.live_device)
        positions = pos.expand(3, -1).contiguous()
        read = torch.tensor(slots, dtype=torch.long, device=self.live_device)
        hidden, logits = self._draft_forward(
            ids, hidden_seed, positions, read[-len(tokens) :], read
        )
        torch.npu.synchronize(self.live_device)
        return hidden, logits

    def _target_forward(
        self, ids, positions, write, read, seat=0, accepted=1, candidate_metadata=None
    ):
        with self._numerical_context(self.target_model, ids, is_draft=False):
            hidden = self.target_model.model.embed_tokens(ids)
            residual = None
            for i, layer in enumerate(self.target_model.model.layers):
                hidden, residual = numerics.decoder(
                    layer,
                    self.target[str(i)],
                    hidden,
                    residual,
                    positions,
                    write,
                    read,
                    self.geometry,
                    seat,
                    accepted,
                    candidate_metadata,
                )
            hidden, _ = self.target_model.model.norm(hidden, residual)
            logits = self._sample(self.target_model, hidden)
            return hidden, logits

    def _draft_forward(self, ids, hidden_seed, positions, write, read):
        with self._numerical_context(self.draft_model, ids, is_draft=True):
            model = self.draft_model.model
            embedding = model.pre_fc_norm_embedding(model.embed_tokens(ids))
            hidden = model.pre_fc_norm_hidden(hidden_seed)
            hidden = model.fc(torch.cat((embedding, hidden), dim=-1))
            hidden, residual = numerics.decoder(
                model.layers[0],
                self.draft,
                hidden,
                None,
                positions,
                write,
                read,
                self.geometry,
                0,
                1,
            )
            hidden, _ = model.norm(hidden, residual)
            logits = self._sample(self.draft_model, hidden)
            return hidden, logits

    def _sample(self, model, hidden):
        if self.greedy_only:
            # Same exact greedy reduction for target and MTP. TP communicates
            # only value/index pairs, never the complete vocabulary.
            return model.logits_processor.get_top_tokens(model.lm_head, hidden)
        return model.compute_logits(hidden)

    def _numerical_context(self, model, ids, *, is_draft):
        config = getattr(model.model, "config", None)
        if getattr(config, "model_type", None) != "qwen3_5_moe_text":
            return nullcontext()
        from vllm.config import get_current_vllm_config
        from vllm_ascend.ascend_forward_context import set_ascend_forward_context

        return set_ascend_forward_context(
            None,
            get_current_vllm_config(),
            num_tokens=ids.shape[0],
            # Match native MTP: its predictor has no target start_layer. The
            # donor has_layer_idx predicate caches the first target observation.
            model_instance=None if is_draft else model,
            is_draft_model=is_draft,
            input_ids=ids,
        )
