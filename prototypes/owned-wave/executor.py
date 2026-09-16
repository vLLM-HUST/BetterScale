# SPDX-License-Identifier: Apache-2.0
"""LiveInference-style ingress/compute/egress reactor for the owned Qwen root.

Adapted protocol from LiveInference AscendDSV4WaveExecutor; its DSV4-specific
packet, resource and schema classes are intentionally not imported into Qwen.
"""
from collections import deque
from dataclasses import dataclass
import torch


@dataclass
class Pending:
    wave: object
    invocation: object
    inputs: tuple
    host: object
    length_carrier: object
    graph_done: object
    copy_done: object


class WaveExecutor:
    def __init__(self, root):
        self.root = root
        self.ingress = torch.npu.Stream()
        self.compute = torch.npu.Stream()
        self.egress = torch.npu.Stream()
        self.pending = deque()
        self.previous = [None, None]
        self.next_sequence = 0
        self.failed = False
        ready = torch.npu.Event()
        ready.record()
        for stream in (self.ingress, self.compute, self.egress):
            stream.wait_event(ready)

    def submit(self, wave, prompt):
        if self.failed or len(self.pending) >= 2 or wave.sequence != self.next_sequence:
            raise ValueError('executor sequence/window/liveness violation')
        root, bank = self.root, wave.bank
        if not 1 <= wave.length <= root.bundle.grant_end:
            raise ValueError('attention length exceeds resource grant')
        if wave.kind == 'prefill' and (
                len(prompt) != root.prefill_width or
                len(prompt) + wave.max_tokens - 1 > root.bundle.grant_end):
            raise ValueError('prefill exceeds frozen shape or grant')
        if wave.kind not in ('prefill', 'decode'):
            raise ValueError('unregistered wave kind')
        previous = self.previous[bank]
        command = torch.tensor([wave.sequence, wave.generation, wave.max_tokens],
                               dtype=torch.int64, pin_memory=True)
        ids = torch.tensor(prompt, dtype=torch.int64, pin_memory=True) if wave.kind == 'prefill' else None
        host = torch.empty((7,), dtype=torch.int64, pin_memory=True)
        invocation = root.invoke(f'{wave.kind}{bank}')
        try:
            with torch.npu.stream(self.ingress):
                if previous is not None:
                    self.ingress.wait_event(previous.graph_done)
                root.authorization[bank].copy_(command, non_blocking=True)
                if ids is not None:
                    root.prompt_ids[bank].copy_(ids, non_blocking=True)
                root.host_lengths[bank] = wave.length
                before = root.forward_calls
                invocation.shadow_replay(stream=self.ingress)
                assert root.forward_calls == before, 'shadow ran numerical forward'
                ready = torch.npu.Event()
                ready.record(self.ingress)
            self.compute.wait_event(ready)
            if previous is not None:
                self.compute.wait_event(previous.copy_done)
            root.publish_attention(bank, self.compute, kind=wave.kind)
            invocation.replay(stream=self.compute)
            done = torch.npu.Event()
            done.record(self.compute)
            with torch.npu.stream(self.egress):
                self.egress.wait_event(done)
                host.copy_(root.egress[bank], non_blocking=True)
                copied = torch.npu.Event()
                copied.record(self.egress)
            metadata = root.metadata if wave.kind == 'decode' else root.prefill_metadata
            ticket = Pending(wave, invocation, (command, ids), host,
                             metadata[bank].seq_lens, done, copied)
            self.pending.append(ticket)
            self.previous[bank] = ticket
            self.next_sequence += 1
            return ticket
        except BaseException:
            self.failed = True
            raise

    def receive_oldest(self):
        if self.failed or not self.pending:
            raise ValueError('no live receipt')
        ticket = self.pending[0]
        try:
            ticket.copy_done.synchronize()
            row = ticket.host.tolist()
            if row[0] != ticket.wave.sequence or row[1] != ticket.wave.generation or row[6]:
                raise ValueError(f'stale or failed device receipt: {row}')
            ticket.invocation.retire()
            self.pending.popleft()
            return row
        except BaseException:
            self.failed = True
            raise
