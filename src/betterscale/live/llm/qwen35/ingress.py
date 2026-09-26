"""Async ingress around one synchronous, distributed completed-wave owner."""

import asyncio
from contextlib import asynccontextmanager

from .scheduler import Scheduler


class Ingress:
    def __init__(self, root, broadcast):
        self.scheduler = Scheduler(root)
        self.broadcast = broadcast
        self.commands = []
        self.futures = {}
        self.wakeup = asyncio.Event()
        self.failure = None
        self.sequence = 0
        root.scheduler = self.scheduler
        root.serving_error = None

    async def execute(self, tokens, count, stops):
        if self.failure is not None:
            raise RuntimeError("live execution owner failed") from self.failure
        self.sequence += 1
        key = str(self.sequence)
        # Event-loop single writer: validation/host admission cannot race a wave.
        self.scheduler.submit(key, tokens, count, stops)
        self.commands.append(("submit", key, tokens, count, stops))
        future = asyncio.get_running_loop().create_future()
        self.futures[key] = future
        self.wakeup.set()
        try:
            return await future
        except asyncio.CancelledError:
            self.commands.append(("cancel", key))
            self.wakeup.set()
            raise
        finally:
            self.futures.pop(key, None)

    async def pump(self):
        try:
            while True:
                await self.wakeup.wait()
                self.wakeup.clear()
                while self.scheduler.busy or self.commands:
                    commands, self.commands = self.commands, []
                    self.broadcast(commands)
                    for command in commands:
                        if command[0] == "cancel":
                            self.scheduler.cancel(command[1])
                    results = self.scheduler.tick()
                    for key, result in results.items():
                        future = self.futures.get(key)
                        if future is not None and not future.done():
                            future.set_result(result)
                    # Receive arrivals/cancellations only after the wave drains.
                    await asyncio.sleep(0)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            self.failure = error
            self.scheduler.root.serving_error = type(error).__name__
            for future in self.futures.values():
                if not future.done():
                    future.set_exception(RuntimeError("live execution owner failed"))
            raise

    @asynccontextmanager
    async def lifespan(self, app):
        task = asyncio.create_task(self.pump(), name="qwen-live-wave-owner")
        try:
            yield
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            finally:
                self.scheduler.close()
