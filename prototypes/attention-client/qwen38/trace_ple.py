"""Prototype-scoped PLE host worker; same native mailbox byte ABI.

Adapted from the owned LiveInfer Qwen38 session/worker (Apache-2.0). Bulk-copy
BF16 bytes instead of constructing one Python scalar per byte. Empty masked
waves still receive a generation acknowledgment for an idle native EP source.
"""

import struct
import time

import torch
from livemodule.serve.qwen38 import Qwen38ServingSession
from livemodule.serve.qwen38.ple_mailbox import (
    Qwen38PLEHostWorker,
    Qwen38PLEMailboxCodec,
)


class BulkCodec(Qwen38PLEMailboxCodec):
    def encode_responses(self, rows, embeddings):
        if embeddings.device.type != "cpu" or embeddings.dtype != torch.bfloat16:
            raise ValueError("PLE requires host BF16")
        if embeddings.shape != (len(rows), self.local_embed_dim):
            raise ValueError("PLE embedding shape mismatch")
        payload = bytearray(self.response_bytes)
        raw = embeddings.contiguous().view(torch.uint8).numpy().tobytes()
        stride = self.local_embed_dim * 2
        for i, row in enumerate(rows):
            struct.pack_into("<5q", payload, row.lane * 40, *row.identity)
            start = self.response_identity_bytes + row.lane * stride
            payload[start : start + stride] = raw[i * stride : (i + 1) * stride]
        return bytes(payload)


class IdleSafeWorker(Qwen38PLEHostWorker):
    def _run(self):
        last_generation = 0
        try:
            while not self._stop.is_set():
                generation = int(self.request_region.generation)
                if not generation or generation == last_generation:
                    time.sleep(self.poll_sleep_seconds)
                    continue
                rows = self.codec.decode_requests(self.request_region.read())
                if any(row.wave_generation != generation for row in rows):
                    raise RuntimeError("PLE wave generation mismatch")
                if rows:
                    tokens = torch.tensor(
                        [r.token_ids for r in rows], dtype=torch.long, device="cpu"
                    )
                    embeddings = self.table.lookup(tokens)[:, -1].to(torch.bfloat16)
                else:
                    embeddings = torch.empty(
                        0,
                        self.codec.local_embed_dim,
                        dtype=torch.bfloat16,
                        device="cpu",
                    )
                self.response_region.publish(
                    self.codec.encode_responses(rows, embeddings), generation
                )
                last_generation = generation
        except BaseException as exc:
            self._error = exc


class TraceSession(Qwen38ServingSession):
    def start_ple(self, *, token_lanes, max_polls):
        from livemodule.arch.ascend.llm.qwen38.ple_mailbox import AscendQwen38PLEMailbox

        if self._mailboxes or self._workers:
            raise RuntimeError("PLE already started")
        try:
            for index, table in self.model.host_ple_tables.items():
                codec = BulkCodec(
                    lanes=token_lanes,
                    local_embed_dim=self.model.plan.ple_heads.count
                    * self.model.contract.ple_head_dim,
                )
                mailbox = AscendQwen38PLEMailbox(codec, max_polls=max_polls)
                self._mailboxes[index] = mailbox
                worker = IdleSafeWorker(
                    codec=codec,
                    table=table,
                    request_region=mailbox.request,
                    response_region=mailbox.response,
                )
                self._workers[index] = worker
                worker.start()
        except BaseException:
            self.close()
            raise
        return self.ple_mailboxes
