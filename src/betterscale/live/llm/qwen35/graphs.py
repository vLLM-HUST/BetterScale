"""Full-model target/draft graphs, owned by the model's LiveModule generation.

Fixed short-wave shapes and a bounded attention envelope are intentional first
qualification limits. No native runner capture or partial eager fallback exists.
"""

from dataclasses import dataclass
import torch
from betterscale.live import GraphCallSchema, MetaTensor, construct_meta_tensors
from .execution import QwenExecutionRoot


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelCallSchema(GraphCallSchema):
    capture_state_blocks: dict


class QwenLiveLLMRoot(QwenExecutionRoot):
    def __init__(
        self,
        geometry,
        capacity,
        target_model,
        draft_model,
        *,
        context_tokens=128,
        greedy_only=False,
    ):
        super().__init__(
            geometry, capacity, target_model, draft_model, greedy_only=greedy_only
        )
        if not 3 <= context_tokens <= 4096:
            raise ValueError("bounded graph context must be in 3..4096")
        self.context_tokens = context_tokens
        for name, width in (
            ("target1", 1),
            ("target3", 3),
            ("draft1", 1),
            ("draft2", 2),
        ):
            self.register_meta_tensor(
                name + "_hidden",
                MetaTensor((width, geometry.hidden_size), dtype=torch.bfloat16),
            )
            self.register_meta_tensor(
                name + "_logits",
                MetaTensor(
                    (width,) if greedy_only else (width, target_model.model.vocab_size),
                    dtype=torch.int64 if greedy_only else torch.bfloat16,
                ),
            )
            ids = torch.ones(width, dtype=torch.long)
            pos = torch.arange(width).expand(3, -1).contiguous()
            read = torch.zeros(context_tokens, dtype=torch.long)
            read[:width] = torch.arange(width)
            write = torch.arange(width)
            if name.startswith("target"):
                args = (
                    ids,
                    pos,
                    read,
                    write,
                    torch.tensor([0, width], dtype=torch.int32),
                    torch.tensor([[0]], dtype=torch.int32),
                    torch.tensor([[0, 1, 2]]),
                    torch.ones(1, dtype=torch.int32),
                    name,
                )
                entry = self.target_forward
            else:
                args = (
                    ids,
                    pos,
                    read,
                    write,
                    torch.zeros(width, geometry.hidden_size, dtype=torch.bfloat16),
                    name,
                )
                entry = self.draft_forward
            self.register_graph(
                name,
                entry=entry,
                schema=ModelCallSchema(
                    args=args,
                    kwargs={},
                    capture_state_blocks={self.residents: (0,), self.pages: (0,)},
                ),
            )

    def _outputs(self, name):
        return getattr(self, name + "_hidden").tensor, getattr(
            self, name + "_logits"
        ).tensor

    def _target1_metadata(self, context):
        return self._outputs("target1")

    def _target3_metadata(self, context):
        return self._outputs("target3")

    def _draft1_metadata(self, context):
        return self._outputs("draft1")

    def _draft2_metadata(self, context):
        return self._outputs("draft2")

    def target_forward(
        self, ids, positions, read, write, cu, conv_slots, candidates, accepted, name
    ):
        out_hidden, out_logits = construct_meta_tensors(
            getattr(self, "_" + name + "_metadata"), context=None
        )
        hidden, logits = self._target_forward(
            ids,
            positions,
            write,
            read,
            candidate_metadata=(cu, conv_slots, candidates, accepted),
        )
        out_hidden.copy_(hidden)
        out_logits.copy_(logits)

    def draft_forward(self, ids, positions, read, write, seed, name):
        out_hidden, out_logits = construct_meta_tensors(
            getattr(self, "_" + name + "_metadata"), context=None
        )
        hidden, logits = self._draft_forward(ids, seed, positions, write, read)
        out_hidden.copy_(hidden)
        out_logits.copy_(logits)

    def _inputs(self, tokens, position, slots):
        if not 0 <= position < len(slots) <= self.context_tokens:
            raise ValueError("call exceeds the owned graph context envelope")
        if len(slots) != position + len(tokens) or len(set(slots)) != len(slots):
            raise ValueError("KV slots must cover exactly the valid prefix and wave")
        ids = torch.tensor(tokens, dtype=torch.long, device="cpu")
        positions = (
            torch.arange(position, position + len(tokens), device="cpu")
            .expand(3, -1)
            .contiguous()
        )
        read = torch.zeros(self.context_tokens, dtype=torch.long, device="cpu")
        read[: len(slots)] = torch.tensor(slots, dtype=torch.long, device="cpu")
        write = read[position : len(slots)].clone()
        return ids, positions, read, write

    def _run(self, name, args):
        stream = torch.npu.current_stream(self.live_device)
        invocation = self.replay(name, *args, stream=stream)
        stream.synchronize()
        # These copies are readers of the graph's output bank too. Retire only
        # after they complete, so a synchronous call cannot race root.close().
        result = tuple(t.clone() for t in self._outputs(name))
        stream.synchronize()
        invocation.retire()
        return result

    @torch.inference_mode()
    def target_step(self, tokens, *, seat, position, slots, accepted=1):
        if (
            len(tokens) not in (1, 3)
            or not 0 <= seat < self.capacity.resident_seats
            or not 1 <= accepted <= 3
        ):
            raise ValueError("unqualified target graph shape or resident selection")
        name = "target" + str(len(tokens))
        args = self._inputs(tokens, position, slots) + (
            torch.tensor([0, len(tokens)], dtype=torch.int32, device="cpu"),
            torch.tensor([[seat]], dtype=torch.int32, device="cpu"),
            torch.tensor(
                [[seat * 3 + i for i in range(3)]], dtype=torch.long, device="cpu"
            ),
            torch.tensor([accepted], dtype=torch.int32, device="cpu"),
            name,
        )
        return self._run(name, args)

    @torch.inference_mode()
    def draft_step(self, tokens, hidden_seed, *, position, slots):
        if len(tokens) not in (1, 2):
            raise ValueError("unqualified draft graph shape")
        name = "draft" + str(len(tokens))
        args = self._inputs(tokens, position, slots) + (hidden_seed, name)
        return self._run(name, args)
