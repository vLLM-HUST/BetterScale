"""Optional native EngineCore wakeup seam; leaves its batch queue untouched."""

import json
import queue
import socket
import tempfile
from threading import Thread


class CompletionInbox:
    def __init__(self, wake):
        self.directory = tempfile.TemporaryDirectory(prefix="bs-cache-")
        self.endpoint = self.directory.name + "/done.sock"
        self.messages = queue.Queue()
        self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        self.socket.bind(self.endpoint)
        self.wake = wake
        self.thread = Thread(
            target=self._receive, name="state-cache-receipts", daemon=True
        )
        self.thread.start()

    def _receive(self):
        while True:
            payload = self.socket.recv(8192)
            if payload == b"close":
                return
            try:
                message = json.loads(payload)
            except BaseException as error:
                message = error
            self.messages.put(message)
            self.wake()

    def drain(self, cache):
        while not self.messages.empty():
            message = self.messages.get_nowait()
            if isinstance(message, BaseException):
                raise message
            cache.receive(message)

    def close(self):
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as channel:
            channel.sendto(b"close", self.endpoint)
        self.thread.join(timeout=5)
        self.socket.close()
        self.directory.cleanup()


def install():
    from vllm.v1.engine import EngineCoreRequestType
    from vllm.v1.engine.core import EngineCoreProc

    if getattr(EngineCoreProc, "_betterscale_cache_wakeup", False):
        return
    original_step = EngineCoreProc._process_engine_step
    original_work = EngineCoreProc.has_work
    original_input = EngineCoreProc._handle_client_request
    original_shutdown = EngineCoreProc.shutdown

    def cache_for(core):
        cache = getattr(core.scheduler, "cache_actions", None)
        if cache is not None and cache.endpoint is None:
            inbox = core._state_cache_inbox = CompletionInbox(
                lambda: core.input_queue.put_nowait(
                    (EngineCoreRequestType.WAKEUP, None)
                )
            )
            cache.endpoint = inbox.endpoint
            core._state_cache_blocked = False
        return cache

    def step(core):
        cache = cache_for(core)
        if cache is not None:
            core._state_cache_inbox.drain(cache)
        executed = original_step(core)
        core._state_cache_model_steps = getattr(
            core, "_state_cache_model_steps", 0
        ) + bool(executed)
        if cache is not None:
            # Pending DMA is not runnable compute. Enter native input_queue wait
            # only after an empty scheduling turn and after all queued frames drain.
            core._state_cache_blocked = bool(
                not executed
                and not core.batch_queue
                and cache.pending
                and not cache.outbox
            )
        return executed

    def has_work(core):
        cache = cache_for(core)
        if cache is not None:
            if cache.outbox or not core._state_cache_inbox.messages.empty():
                return True
            if core._state_cache_blocked and not core.batch_queue:
                return False
        return original_work(core)

    def handle_input(core, *args, **kwargs):
        # New requests/cancellation and completion WAKEUP all get a fresh turn.
        core._state_cache_blocked = False
        return original_input(core, *args, **kwargs)

    def state_cache(core, command):
        cache = cache_for(core)
        if cache is None:
            raise ValueError("state cache maintenance is not enabled")
        core._state_cache_inbox.drain(cache)
        kind = command["kind"]
        if kind == "snapshot":
            return dict(
                cache.snapshot(),
                native_async=core.async_scheduling,
                step_fn=core.step_fn.__name__,
                batch_queue_size=core.batch_queue_size,
                model_steps=getattr(core, "_state_cache_model_steps", 0),
            )
        if kind == "wait":
            return cache.result(command["operation"])
        if kind == "store":
            return cache.store(command["seat"], command["key"])
        if kind == "load":
            return cache.load(command["key"], command["seat"])
        if kind == "cancel":
            cache.cancel(command["operation"])
            return None
        if kind == "drop":
            return cache.drop(command["key"])
        raise ValueError("unknown State cache command")

    def shutdown(core):
        # Native shutdown terminates rank workers before closing their receipt path.
        try:
            return original_shutdown(core)
        finally:
            inbox = getattr(core, "_state_cache_inbox", None)
            if inbox is not None:
                inbox.close()

    EngineCoreProc._process_engine_step = step
    EngineCoreProc.has_work = has_work
    EngineCoreProc._handle_client_request = handle_input
    EngineCoreProc.state_cache = state_cache
    EngineCoreProc.shutdown = shutdown
    EngineCoreProc._betterscale_cache_wakeup = True
