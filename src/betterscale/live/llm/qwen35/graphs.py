"""Full-model target/draft graphs, owned by the model's LiveModule generation.

Fixed short-wave shapes and a bounded attention envelope are intentional first
qualification limits. No native runner capture or partial eager fallback exists.
"""

from dataclasses import dataclass
from functools import partial

import torch

from betterscale.live import GraphCallSchema, MetaTensor, construct_meta_tensors

from .execution import QwenExecutionRoot


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelCallSchema(GraphCallSchema):
    capture_state_blocks: dict
    graph_pool_key: object


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
        self._metadata_actions = {}
        # All waves are serial on one stream; only retained output banks cross
        # calls. The common backend may therefore reuse transient graph scratch.
        self._graph_pool = object()
        self.batch_sizes = tuple(
            2**i for i in range(capacity.execution_seats.bit_length())
        )
        if capacity.token_pages is not None and capacity.token_pages < max(
            self.batch_sizes
        ):
            raise ValueError("capture needs one physical token page per execution lane")
        for batch in self.batch_sizes:
            for kind, width in (
                ("target", 1),
                ("target", 3),
                ("draft", 1),
                ("draft", 2),
            ):
                self._declare_call(kind, width, batch)

    @staticmethod
    def call_name(kind, width, batch):
        return f"{kind}{width}" + (f"_b{batch}" if batch > 1 else "")

    def _declare_call(self, kind, width, batch):
        name = self.call_name(kind, width, batch)
        self._metadata_actions[name] = partial(self._output_metadata, name)
        rows = batch * width
        self.register_meta_tensor(
            name + "_hidden",
            MetaTensor((rows, self.geometry.hidden_size), dtype=torch.bfloat16),
        )
        self.register_meta_tensor(
            name + "_logits",
            MetaTensor(
                (rows,)
                if self.greedy_only
                else (rows, self.target_model.model.vocab_size),
                dtype=torch.int64 if self.greedy_only else torch.bfloat16,
            ),
        )
        ids = torch.ones(rows, dtype=torch.long)
        pos = torch.arange(width).repeat(batch).expand(3, -1).contiguous()
        read = torch.zeros(batch, self.context_tokens, dtype=torch.long)
        write = (
            torch.arange(batch)[:, None] * self.capacity.page_tokens
            + torch.arange(width)[None, :]
        )
        read[:, :width] = write
        args = (ids, pos, read, write.flatten())
        if kind == "target":
            args += (
                torch.arange(batch + 1, dtype=torch.int32) * width,
                torch.arange(batch, dtype=torch.int32)[:, None],
                torch.arange(batch * 3).reshape(batch, 3),
                torch.ones(batch, dtype=torch.int32),
                name,
            )
            entry = self.target_forward
        else:
            args += (
                torch.zeros(rows, self.geometry.hidden_size, dtype=torch.bfloat16),
                name,
            )
            entry = self.draft_forward
        self.register_graph(
            name,
            entry=entry,
            schema=ModelCallSchema(
                args=args,
                kwargs={},
                graph_pool_key=self._graph_pool,
                capture_state_blocks={
                    self.residents: tuple(range(batch)),
                    self.pages: tuple(range(batch)),
                },
            ),
        )

    def _outputs(self, name):
        return getattr(self, name + "_hidden").tensor, getattr(
            self, name + "_logits"
        ).tensor

    def _output_metadata(self, name, context):
        return self._outputs(name)

    def target_forward(
        self, ids, positions, read, write, cu, conv_slots, candidates, accepted, name
    ):
        out_hidden, out_logits = construct_meta_tensors(
            self._metadata_actions[name], context=None
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
            self._metadata_actions[name], context=None
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
    def batch_step(self, steps):
        """Execute real rows only; the scheduler decomposes odd counts into buckets."""
        self._require_active("execute batch on")
        batch = len(steps)
        if batch not in self.batch_sizes:
            raise ValueError("unqualified graph batch size")
        kind, width = steps[0].kind, len(steps[0].tokens)
        if (
            kind not in ("target", "draft")
            or width not in ((1, 3) if kind == "target" else (1, 2))
            or any(step.kind != kind or len(step.tokens) != width for step in steps)
        ):
            raise ValueError("unqualified mixed graph wave")
        inputs = [
            self._inputs(s.tokens, s.metadata["position"], s.metadata["slots"])
            for s in steps
        ]
        writes = [slot for s in steps for slot in s.metadata["slots"][-width:]]
        if len(set(writes)) != len(writes):
            raise ValueError("batch writers cannot alias token State")
        name = self.call_name(kind, width, batch)
        args = (
            torch.cat([v[0] for v in inputs]),
            torch.cat([v[1] for v in inputs], dim=1),
            torch.stack([v[2] for v in inputs]),
            torch.cat([v[3] for v in inputs]),
        )
        if kind == "target":
            seats = [s.metadata["seat"] for s in steps]
            accepted = [s.metadata.get("accepted", 1) for s in steps]
            if (
                len(set(seats)) != batch
                or any(not 0 <= seat < self.capacity.resident_seats for seat in seats)
                or any(not 1 <= count <= 3 for count in accepted)
            ):
                raise ValueError("invalid resident or candidate selection")
            args += (
                torch.arange(batch + 1, dtype=torch.int32) * width,
                torch.tensor(seats, dtype=torch.int32)[:, None],
                torch.tensor([[seat * 3 + i for i in range(3)] for seat in seats]),
                torch.tensor(accepted, dtype=torch.int32),
                name,
            )
        else:
            if any(
                s.hidden_seed.shape != (width, self.geometry.hidden_size) for s in steps
            ):
                raise ValueError("MTP needs one hidden seed per shifted input")
            args += (torch.cat([s.hidden_seed for s in steps]), name)
        hidden, logits = self._run(name, args)
        return [
            (hidden[i * width : (i + 1) * width], logits[i * width : (i + 1) * width])
            for i in range(batch)
        ]

    @torch.inference_mode()
    def target_step(self, tokens, *, seat, position, slots, accepted=1):
        from .generation import ModelStep

        return self.batch_step(
            [
                ModelStep(
                    "target",
                    tokens,
                    {
                        "seat": seat,
                        "position": position,
                        "slots": slots,
                        "accepted": accepted,
                    },
                )
            ]
        )[0]

    @torch.inference_mode()
    def draft_step(self, tokens, hidden_seed, *, position, slots):
        from .generation import ModelStep

        return self.batch_step(
            [
                ModelStep(
                    "draft", tokens, {"position": position, "slots": slots}, hidden_seed
                )
            ]
        )[0]
