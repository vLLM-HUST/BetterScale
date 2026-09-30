"""Finite-wave host loop. No resident device scheduler or per-wave client RPC.

The device binding must implement the Backend contract below. This module owns
policy and enqueue dependencies, not a fake NPU implementation. All launches are
finite; events stay live until completion. tick() is nonblocking on device work.
"""
from typing import Protocol
from .wave_protocol import ReturnMode, Scheduler


class Backend(Protocol):
    """Binding requirements (the CPU oracle implements these for contract tests).

    snapshot(): coherently read immutable READY descriptors, without waiting for
      absent clients. Payloads/routes stay on device. Never return a half-written
      header; producer payload completion precedes READY publication.
    acknowledgements(): exact Output tokens after client consumption; for pull,
      after the asynchronous payload pull finishes. No ack on metadata receipt.
    launch(op, wave, stream, waits): enqueue a finite op on FIFO 'cube'/'vector',
      wait on supplied device events, and return a queryable completion event.
      pull also builds local expert counts/permutation from device routing data.
      down writes route-valued BF16 output, never owner-local weighted sums.
      push scatters to disjoint client token/top-k slots, then fences visibility.
      No operation may write outside this wave's slot or route destinations.
    done(event): nonblocking completion query; errors must raise, not mean idle.
    notify(outputs, mode): publish per-request generation-tagged completions.
      Pull descriptors refer to an owner-local slot; push means payload is already
      at the client. Keys/pointers are private bootstrap state, not log fields.
    idle(): bounded host wait/yield when no scheduling progress occurred.
    """
    def snapshot(self): ...
    def acknowledgements(self): ...
    def launch(self, op, wave, stream, waits): ...
    def done(self, event): ...
    def notify(self, outputs, mode): ...
    def idle(self): ...


class ServerLoop:
    def __init__(self, scheduler: Scheduler, backend: Backend):
        self.scheduler, self.backend = scheduler, backend
        self.prepared = None  # (wave, completed-input event)
        self.previous_down = None
        self.pending = []  # (wave, down event, return-complete event)

    def tick(self):
        """Make bounded host progress; at most one UP/GATE/DOWN wave per tick."""
        s, b = self.scheduler, self.backend
        progress = False
        for ack in b.acknowledgements():
            s.retire(ack)
            progress = True
        # Publish independently of future traffic, including the final wave.
        for item in tuple(self.pending):
            wave, down, returned = item
            if b.done(returned):
                if not b.done(down):
                    raise RuntimeError('return event completed before DOWN')
                s.down_complete(wave)
                outputs = s.publish_output(wave)
                b.notify(outputs, s.return_mode)
                self.pending.remove(item)
                progress = True
        for request in b.snapshot():
            s.publish(request)
        if self.prepared is None:
            wave = s.admit()
            if wave is not None:
                pull = b.launch('pull', wave, 'vector', ())
                self.prepared = (wave, pull)
        if self.prepared is not None:
            wave, pull = self.prepared
            waits = (pull,) if self.previous_down is None else (pull, self.previous_down)
            up = b.launch('up', wave, 'cube', waits)
            gate = b.launch('gate', wave, 'vector', (up,))
            down = b.launch('down', wave, 'cube', (gate,))
            self.previous_down = down
            # Snapshot already frozen this tick: don't wait for another source.
            # Enqueue next PULL only after current GATE, and before current PUSH.
            next_wave = s.admit()
            self.prepared = None
            if next_wave is not None:
                next_pull = b.launch('pull', next_wave, 'vector', (gate,))
                self.prepared = (next_wave, next_pull)
            returned = (b.launch('push', wave, 'vector', (down,))
                        if s.return_mode is ReturnMode.PUSH else down)
            self.pending.append((wave, down, returned))
            progress = True
        return progress

    def run(self, stop, *, max_ticks=None):
        """Host-owned loop; cancellation stops new scheduling, not peer cleanup.

        Caller must quiesce outstanding device work and coordinate session abort
        before freeing IPC. A stop request is not an implicit successful drain.
        """
        ticks = 0
        while not stop() and not self.scheduler.drained:
            if max_ticks is not None and ticks >= max_ticks:
                raise TimeoutError('finite host-loop tick budget exceeded')
            if not self.tick():
                self.backend.idle()
            ticks += 1
        return self.scheduler.drained
