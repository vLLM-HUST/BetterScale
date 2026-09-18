"""Whole-model prefill capture with persistent inputs and device-authored topology.

Unlike the donor27 patch this runner already owns graphable GDN metadata.
Keep cold/continuation Python specializations separate; lengths and inactive
seats remain device inputs. No CPU validity check or token conversion is captured.
"""

import torch
from livemodule.llm.forward_context import ForwardContext
from livemodule.llm.qwen35.batch import Qwen35DeviceBatchTopology as Topology


def forward_prefill(engine, ids, lengths, before, is_fresh):
    b, width = ids.shape
    active = lengths.gt(0)
    fresh = before.eq(0) & active
    pos = (
        before[:, None]
        + torch.arange(width, device="npu")[None]
        - width
        + lengths[:, None]
    ).clamp_min(0)
    topology = Topology(
        before + lengths,
        lengths,
        active,
        engine.blocks,
        width,
        is_fresh,
        continuation_prefill=not is_fresh,
        query_padding=True,
    )
    # Continuation prefill reads canonical GDN row0; speculative decode
    # selects a candidate and a convolution slice. Materialize only those
    # endpoints before switching phase. PLE consumes its own selector later.
    accepted = engine.root.request_state.accepted_tokens.tensor[:b].clamp(
        1, engine.k + 1
    )
    for layer in engine.root.model.language_model.layers:
        gdn = getattr(layer, "linear_attn", None)
        if gdn is None:
            continue
        rows = (
            torch.arange(b, device="npu") * (engine.k + 1)
            + gdn.conv_state.leading_physical_blocks
        )
        indices = rows[:, None] + torch.arange(engine.k + 1, device="npu")[None]
        _, conv, recurrent = gdn._backend._accepted_state(gdn, indices, accepted)
        old_conv = gdn.conv_state.tensor.index_select(0, rows)
        canonical = torch.zeros_like(old_conv)
        canonical[..., : gdn.conv_kernel_size - 1] = conv
        gdn.conv_state.tensor.index_copy_(
            0, rows, torch.where(active[:, None, None], canonical, old_conv)
        )
        old_rec = gdn.recurrent_state.tensor.index_select(0, rows)
        gdn.recurrent_state.tensor.index_copy_(
            0, rows, torch.where(active[:, None, None, None], recurrent, old_rec)
        )
    context = ForwardContext({}, {}, {})
    context.batch_topology = topology
    engine.cfg.remote_expert_priority = 1
    engine.generation.add_(1)
    engine.status.zero_()
    with context.activate():
        hidden, multi, _, valid = engine.root.forward_request_owned_continuous_ple(
            ids,
            positions=pos,
            mailbox=engine.mailbox,
            generation=engine.generation,
            response_payload=engine.response,
            status=engine.status,
            slot_ids=torch.arange(engine.lanes, device="npu") // width,
            request_generations=engine.identities,
            destination_generations=engine.identities,
        )
    # Pair each real input token with its immediately preceding target row.
    # Right padding is NOT a preceding row: replace that boundary explicitly.
    previous = torch.cat((engine.multi, multi[:, :-1]), dim=1)
    rowids = torch.arange(b, device="npu")
    previous[rowids, (width - lengths).clamp_max(width - 1)] = engine.multi[:, 0]
    mtp_lengths = (lengths - fresh.to(lengths.dtype)).clamp_min(0)
    mtp_topology = Topology(
        (before + lengths - 1).clamp_min(0),
        mtp_lengths,
        mtp_lengths.gt(0),
        engine.blocks,
        width,
        is_fresh,
        continuation_prefill=not is_fresh,
        query_padding=True,
    )
    engine.serving.draft_cabin._run_mtp(
        ids, (pos - 1).clamp_min(0), previous, mtp_topology
    )
    engine.multi.copy_(torch.where(active[:, None, None], multi[:, -1:], engine.multi))
    tokens = engine.root.compute_top_tokens(hidden[:, -1:])[:, 0]
    valid_mask = topology.query_valid.reshape(-1)
    return tokens, (valid[: b * width] | ~valid_mask).all(), context


class PrefillGraphRunner:
    """One input bank per static prefill kind; replay owns all derived metadata.

    This deliberately serial qualification path does not promise H2D overlap.
    Host publication precedes replay on the same stream. Graph outputs and the
    ForwardContext stay alive until reset; no host value determines work inside
    replay. Mailbox generations are never restored by the state shadow.
    """

    def __init__(self, engine):
        import os

        self.engine = engine
        self.width = min(512, engine.lanes // engine.batch)
        self.enabled = os.environ.get("QWEN38_PREFILL_GRAPH", "0") == "1"
        self.shadow_enabled = os.environ.get("QWEN38_PREFILL_SHADOW", "0") == "1"
        self.banks = {}
        self.receipts = []
        for fresh in (True, False):
            self.banks[fresh] = dict(
                ids=torch.zeros(
                    engine.batch, self.width, dtype=torch.int64, device="npu"
                ),
                lengths=torch.zeros(engine.batch, dtype=torch.int64, device="npu"),
                before=torch.zeros(engine.batch, dtype=torch.int64, device="npu"),
                graph=None,
            )

    def publish(self, chunks, cursors):
        if len(chunks) != self.engine.batch or len(cursors) != self.engine.batch:
            raise ValueError("prefill seat count changed")
        if max(map(len, chunks)) > self.width or min(cursors) < 0:
            raise ValueError("prefill input exceeds bucket or has negative cursor")
        fresh = all(c == 0 for c in cursors)
        bank = self.banks[fresh]
        ids = torch.zeros(self.engine.batch, self.width, dtype=torch.int64)
        for row, values in enumerate(chunks):
            if values:
                ids[row, -len(values) :] = torch.tensor(values, dtype=torch.int64)
        bank["ids"].copy_(ids)
        bank["lengths"].copy_(torch.tensor(list(map(len, chunks)), dtype=torch.int64))
        bank["before"].copy_(torch.tensor(cursors, dtype=torch.int64))
        return fresh, bank

    def forward(self, fresh, bank):
        return forward_prefill(
            self.engine, bank["ids"], bank["lengths"], bank["before"], fresh
        )

    def run(self, chunks, cursors):
        fresh, bank = self.publish(chunks, cursors)
        if bank["graph"] is None:
            output = self.forward(fresh, bank)
        else:
            bank["graph"].replay()
            output = bank["output"]
        tokens, valid, _ = output
        if not bool(valid.cpu()):
            raise RuntimeError(
                f"PLE prefill publication failed: status={self.engine.status.cpu().tolist()}, "
                f"generation={self.engine.generation.cpu().tolist()}"
            )
        return tokens.cpu().tolist()

    def capture(self):
        if not self.enabled:
            return
        # Independent graph pools first: safe lifecycle qualification before
        # attempting cross-graph scratch pooling. Graph memory is reported below.
        for fresh, bank in self.banks.items():
            cursors = [0 if fresh else 3] * self.engine.batch
            self.publish([[9707, 11, 1879]] * self.engine.batch, cursors)
            self.forward(fresh, bank)
            torch.npu.synchronize()
            stream = torch.npu.Stream()
            stream.wait_stream(torch.npu.current_stream())
            bank["graph"] = torch.npu.NPUGraph()
            with torch.npu.stream(stream):
                with torch.npu.graph(bank["graph"]):
                    bank["output"] = self.forward(fresh, bank)
            stream.synchronize()
            if self.shadow_enabled:
                for length in (2, 1, self.width):
                    self.shadow(fresh, bank, length)
        print(
            {
                "prefill_full_graph": True,
                "shadow_enabled": self.shadow_enabled,
                "shadows": self.receipts,
                "allocated": torch.npu.memory_allocated(),
                "reserved": torch.npu.memory_reserved(),
            },
            flush=True,
        )

    def shadow(self, fresh, bank, length):
        """Same-state eager/replay on changed lengths, including an inactive seat.

        Snapshot model State and recurrent carry, not mailbox generation/receipt
        counters. The protocol continues monotonically while math state rewinds.
        Compare every retained state, not just the sampled next token.
        """
        engine = self.engine
        tensors = [(name, state.tensor) for name, state in engine.root.named_states()]
        tensors.append(("trace.multi", engine.multi))
        chunks = [
            [9707] * length if row % 2 == 0 else [] for row in range(engine.batch)
        ]
        self.publish(chunks, [0 if fresh else 5] * engine.batch)
        saved = [(name, value, value.clone()) for name, value in tensors]
        eager = self.forward(fresh, bank)
        expected_tokens = eager[0].clone()
        if not bool(eager[1].cpu()):
            raise AssertionError("eager prefill PLE failed")
        expected = [(name, value.clone()) for name, value in tensors]
        for _, value, original in saved:
            value.copy_(original)
        bank["graph"].replay()
        torch.npu.synchronize()
        actual = bank["output"]
        if not bool(actual[1].cpu()) or not torch.equal(actual[0], expected_tokens):
            raise AssertionError("prefill replay validity/token mismatch")
        max_rl2, exact = 0.0, 0
        for (name, value), (_, reference) in zip(tensors, expected):
            if torch.equal(value, reference):
                exact += 1
                continue
            if not value.is_floating_point():
                raise AssertionError(f"prefill integer state mismatch: {name}")
            error = float((value.float() - reference.float()).norm().cpu())
            scale = float(reference.float().norm().cpu())
            rl2 = error / max(scale, 1e-9)
            if not rl2 <= 0.001:
                # Distinguish replay corruption from an already non-repeatable
                # eager data plane (e.g. concurrent arrival-order reductions).
                # This is failure evidence, never permission to relax the gate.
                graph_error = dict(
                    state=name,
                    fresh=fresh,
                    length=length,
                    dtype=str(value.dtype),
                    shape=list(value.shape),
                    graph_rl2=rl2,
                    graph_max_abs=float(
                        (value.float() - reference.float()).abs().max().cpu()
                    ),
                )
                for _, target, original in saved:
                    target.copy_(original)
                repeated = self.forward(fresh, bank)
                torch.npu.synchronize()
                graph_error["eager_repeat_valid"] = bool(repeated[1].cpu())
                graph_error["eager_repeat_rl2"] = float(
                    (value.float() - reference.float()).norm().cpu()
                ) / max(scale, 1e-9)
                import json

                path = (
                    engine.args.directory
                    / f"prefill-shadow-failure-source{engine.args.source}.json"
                )
                path.write_text(json.dumps(graph_error, indent=2) + "\n")
                raise AssertionError(f"prefill state mismatch: {graph_error}")
            max_rl2 = max(max_rl2, rl2)
        self.receipts.append(
            dict(
                fresh=fresh,
                length=length,
                states=len(tensors),
                exact=exact,
                max_rl2=max_rl2,
            )
        )

    def close(self):
        for bank in self.banks.values():
            if bank["graph"] is not None:
                bank["graph"].reset()
        self.banks.clear()
