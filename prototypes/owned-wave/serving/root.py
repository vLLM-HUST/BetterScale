"""Fixed graph portfolio, variable resident State, native Qwen numerical bodies."""

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
import os
from functools import partial
import torch
from livemodule import (
    LiveModule,
    MetaTensor,
    StateTensor,
    StateDomain,
    ExactStateCapacity,
)
from livemodule.runtime.meta_tensor import construct_meta_tensors
from livemodule.runtime.phase import LivePhase, current_live_phase
from live_root import OwnedRoot, WaveSchema
from vllm_ascend.compilation import acl_graph
from vllm_ascend.attention import attention_v1


@dataclass(frozen=True)
class AttentionWaveSchema(WaveSchema):
    # Only same-bank/same-shape FD variants share transient allocation. All
    # cross-wave values already live in retained State or frame ingress/metadata.
    graph_pool_key: object


class Lengths(MetaTensor):
    def __init__(self, values):
        self.values = tuple(values)

    def construct(self):
        return torch.tensor(self.values, dtype=torch.int32, device="cpu")


class ServingRoot(OwnedRoot):
    """Host allocates/authorizes; the compute graph alone writes resident truth.

    State columns: cursor, anchor, remaining, phase, generation, prompt_len,
    generated. Phase: empty0/prefill1/decode2/done3. One prefill resident per
    wave; decode batches all authorized residents. Padding has no KV/state authority.
    """

    def __init__(self, bundle, backend, *, residents, max_length, chunks):
        LiveModule.__init__(self)
        self.bundle = bundle
        self.residents, self.max_length = residents, max_length
        self.domain = StateDomain(ExactStateCapacity(1))
        dev = bundle.device
        self.progress = torch.zeros((residents, 7), dtype=torch.int64, device=dev)
        self.tables = torch.zeros(
            (residents, bundle.table.shape[1]), dtype=torch.int32, device=dev
        )
        self.egress = torch.zeros((2, residents, 8), dtype=torch.int64, device=dev)
        for i, buffer in enumerate(
            [*bundle.pools, self.progress, self.tables, self.egress]
        ):
            raw = buffer.view(torch.uint8).flatten()
            state = StateTensor(
                role=f"serving-state-{i}",
                requirement="exclusive adopted arena",
                block_shape=(raw.numel(),),
                storage_dtype=torch.uint8,
                domain=self.domain,
            )
            self.register_state(f"state_{i}", state)
            backend.buffers[state] = raw
        self.frames, self.params, self.actions = {}, {}, {}
        self.static_attention = None
        if os.environ.get("OWNED_STATIC_FIA", "1") == "1":
            from static_attention import StaticAttention

            if os.environ.get("OWNED_HOST_FIA", "0") == "1":
                from host_attention import HostAttention

                self.static_attention = HostAttention(self)
            else:
                self.static_attention = StaticAttention(self)
        self.update_stream = (
            torch.npu.Stream() if self.static_attention is None else None
        )
        self.execution_stream = (
            torch.npu.Stream() if hasattr(self.static_attention, "prepare") else None
        )
        capture_pools = {}
        self.forward_calls = self.shadow_actions = 0
        entries = []
        for kind, count in [("p", q) for q in chunks] + [("d", residents)]:
            entries.append((kind, count, False))
            if hasattr(
                self.static_attention, "fd_eligible"
            ) and self.static_attention.fd_eligible(kind, count):
                entries.append((kind, count, True))
        for kind, count, fd in entries:
            for bank in range(2):
                key = f"{kind}{count}b{bank}" + ("fd" if fd else "")
                m = self.make_metadata(count, count, decode=kind == "d")
                rows = residents if kind == "d" else 1
                m.block_tables = torch.zeros(
                    (rows, bundle.table.shape[1]), dtype=torch.int32, device=dev
                )
                if kind == "d":
                    from vllm_ascend.attention.attention_mask import (
                        AttentionMaskBuilder,
                    )

                    m.attn_mask = AttentionMaskBuilder(dev).get_attention_mask(
                        True, bundle.config.model_config
                    )
                    m.num_decodes = residents
                    m.max_query_len = 1
                    m.query_start_loc = torch.arange(
                        residents + 1, dtype=torch.int32, device=dev
                    )
                    m.actual_seq_lengths_q = list(range(1, residents + 1))
                frame = dict(
                    kind=kind,
                    fd=fd,
                    count=count,
                    bank=bank,
                    metadata=m,
                    lengths=[count] if kind == "p" else [1] * residents,
                    ids=torch.zeros(count, dtype=torch.int64, device=dev),
                    # seq, slot, gen, start, hit, promptlen, outputlimit
                    command=torch.tensor(
                        [0, 0, 1, 1, 0, count, 1], dtype=torch.int64, device=dev
                    ),
                    block_row=torch.zeros(
                        bundle.table.shape[1], dtype=torch.int32, device=dev
                    ),
                    generations=torch.zeros(residents, dtype=torch.int64, device=dev),
                )
                frame["device_q_lengths"] = torch.tensor(
                    m.actual_seq_lengths_q, dtype=torch.int64, device=dev
                )
                frame["device_kv_lengths"] = torch.ones(
                    rows, dtype=torch.int64, device=dev
                )
                if hasattr(self.static_attention, "prepare"):
                    frame["tiling"] = torch.zeros(2528, dtype=torch.uint8, device=dev)
                self.frames[key] = frame
                self.params[key] = acl_graph.GraphParams(
                    {count: []}, {count: None}, {count: []}, {count: []}
                )
                self.actions[key] = partial(self.construct, key)
                schema = WaveSchema((key,), {}, {self.domain: (0,)})
                if hasattr(self.static_attention, "prepare"):
                    base = key.removesuffix("fd")
                    owner = capture_pools.setdefault(base, object())
                    schema = AttentionWaveSchema((key,), {}, {self.domain: (0,)}, owner)
                self.register_graph(
                    key,
                    entry=self.prefill if kind == "p" else self.decode,
                    schema=schema,
                )

    def construct(self, key, context):
        self.shadow_actions += 1
        f = self.frames[key]
        m = f["metadata"]
        m.seq_lens = Lengths(f["lengths"]).tensor
        m.seq_lens_cpu = m.seq_lens
        m.seq_lens_list = list(f["lengths"])
        return m

    @contextmanager
    def registry(self, key):
        old_params, old_keys = acl_graph._graph_params, attention_v1._ATTN_KEYS_BUFFER
        acl_graph._graph_params = self.params[key]
        attention_v1._ATTN_KEYS_BUFFER = list(self.bundle.layer_names)
        try:
            yield
        finally:
            acl_graph._graph_params, attention_v1._ATTN_KEYS_BUFFER = (
                old_params,
                old_keys,
            )

    def run_model(self, ids, positions, m, key, *, last_only):
        self.forward_calls += 1
        with (
            self.registry(key),
            (
                self.static_attention.scope(key)
                if self.static_attention
                else nullcontext()
            ),
            self.context(
                m, ids.shape[0], capturing=current_live_phase() == LivePhase.CAPTURE
            ),
        ):
            hidden = self.bundle.model(
                input_ids=ids,
                positions=positions,
                intermediate_tensors=None,
                inputs_embeds=None,
            )
        if last_only:
            last = self.frames[key]["device_q_lengths"] - 1
            hidden = hidden.index_select(0, last)
        logits = self.bundle.model.compute_logits(hidden)
        return (
            self.bundle.sampler(logits=logits, sampling_metadata=self.bundle.sampling)
            .sampled_token_ids.flatten()
            .to(torch.int64)
        )

    def prefill(self, key):
        f = self.frames[key]
        q = f["count"]
        bank = f["bank"]
        m = construct_meta_tensors(self.actions[key], context=None)
        seq, slot, gen, start, hit, promptlen, limit = f["command"].unbind()
        index = slot.view(1)
        old = self.progress.index_select(0, index)[0]
        cursor = torch.where(start.bool(), hit, old[0])
        valid = f["device_q_lengths"][0]
        offsets = torch.arange(q, device=self.bundle.device)
        live = offsets < valid
        f["padding_mask"] = ~live
        positions = torch.where(live, cursor + offsets, 0)
        # New-generation publication is compute-ordered behind old readers.
        table = torch.where(
            start.bool(), f["block_row"], self.tables.index_select(0, index)[0]
        )
        self.tables.index_copy_(0, index, table.view(1, -1))
        m.block_tables.copy_(table.view(1, -1))
        physical = table.gather(0, positions // self.bundle.block_size)
        m.slot_mapping.copy_(
            torch.where(
                live,
                physical * self.bundle.block_size + positions % self.bundle.block_size,
                -1,
            ).to(torch.int32)
        )
        if self.static_attention is not None:
            f["device_kv_lengths"].copy_((cursor + valid).view(1))
        sampled = self.run_model(f["ids"], positions, m, key, last_only=True)[0]
        end = cursor + valid
        final = end == promptlen
        generated = final.to(torch.int64)
        done = final & (limit == 1)
        phase = torch.where(final, torch.where(done, 3, 2), 1)
        state = torch.stack(
            (end, sampled, limit - generated, phase, gen, promptlen, generated)
        )
        self.progress.index_copy_(0, index, state.view(1, -1))
        out = self.egress[bank]
        out.zero_()
        out[:, 1].fill_(-1)
        row = torch.stack(
            (
                seq,
                slot,
                gen,
                end,
                torch.where(final, sampled, -1),
                generated,
                done.to(torch.int64),
                torch.zeros_like(seq),
            )
        )
        out.index_copy_(0, index, row.view(1, -1))
        return out

    def decode(self, key):
        f = self.frames[key]
        bank = f["bank"]
        m = construct_meta_tensors(self.actions[key], context=None)
        old = self.progress.clone()
        cursor, anchor, remaining, phase, gen, promptlen, generated = old.unbind(1)
        authorized = f["generations"] > 0
        qualified = f["generations"] == gen
        active = authorized & qualified & (phase == 2) & (remaining > 0)
        m.block_tables.copy_(torch.where(active[:, None], self.tables, 0))
        blocks = self.tables.gather(
            1,
            (cursor // self.bundle.block_size).clamp(max=self.tables.shape[1] - 1)[
                :, None
            ],
        ).flatten()
        slots = blocks * self.bundle.block_size + cursor % self.bundle.block_size
        m.slot_mapping.copy_(torch.where(active, slots, -1).to(torch.int32))
        if self.static_attention is not None:
            f["device_kv_lengths"].copy_(torch.where(active, cursor + 1, 1))
        sampled = self.run_model(anchor, cursor, m, key, last_only=False)
        count = active.to(torch.int64)
        left = remaining - count
        end = cursor + count
        done = (left == 0) & authorized & qualified
        state = torch.stack(
            (
                end,
                torch.where(active, sampled, anchor),
                left,
                torch.where(done, 3, phase),
                gen,
                promptlen,
                generated + count,
            ),
            dim=1,
        )
        self.progress.copy_(state)
        ids = torch.arange(self.residents, device=self.bundle.device, dtype=torch.int64)
        seq = f["command"][0].expand_as(ids)
        out = torch.stack(
            (
                seq,
                torch.where(authorized, ids, -1),
                gen,
                end,
                torch.where(active, sampled, -1),
                count,
                done.to(torch.int64),
                (authorized & ~qualified).to(torch.int64),
            ),
            dim=1,
        )
        self.egress[bank].copy_(out)
        return self.egress[bank]

    def publish(self, key, compute):
        if self.static_attention is not None:
            return
        f = self.frames[key]
        ready = torch.npu.Event()
        ready.record(compute)
        self.update_stream.wait_event(ready)
        with self.registry(key), self.context(f["metadata"], f["count"]) as ctx:
            acl_graph.update_full_graph_params(
                self.bundle.attention_backend,
                self.update_stream,
                ctx,
                f["count"],
                self.bundle.config,
                None,
            )
