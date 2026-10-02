"""Bounded nonblocking delivery: a slow/disconnected reader never stalls State work."""
import asyncio
from collections import deque
import json
from naive_pool_node import pack


class Progress:
    def __init__(self,limit=256):
        self.limit=limit;self.items=deque();self.changed=asyncio.Event()
        self.error=None;self.closed=False;self.detached=False

    def push(self,value):
        if self.detached or self.error or self.closed:return
        if len(self.items)>=self.limit:
            self.fail(RuntimeError("Stream reader exceeded bounded output queue"))
            return
        self.items.append(value);self.changed.set()

    def fail(self,error):
        self.error=error;self.closed=True;self.changed.set()

    def finish(self):
        self.closed=True;self.changed.set()

    def detach(self):
        self.detached=True;self.items.clear()

    async def read(self):
        while True:
            if self.error:raise self.error
            if self.items:return self.items.popleft()
            if self.closed:return None
            self.changed.clear()
            await self.changed.wait()


async def generate(peer,instance,on_tokens,**args):
    """Framed msgpack avoids a giant newline/JSON decode for long prompts."""
    from naive_pool_node import unpack
    import struct
    async with peer.client.post(peer.url+"/generate",
            data=pack(dict(instance=instance,op="generate",args=args))) as response:
        if response.status!=200:
            raise RuntimeError("Generation stream rejected: "+(await response.text())[:2048])
        while True:
            size=struct.unpack("!I",await response.content.readexactly(4))[0]
            if not 0<size<=8<<20:raise RuntimeError("Invalid generation frame size")
            value=unpack(await response.content.readexactly(size))
            if value["kind"]=="tokens":on_tokens(value["value"])
            elif value["kind"]=="done":return value["value"]
            elif value["kind"]=="error":raise RuntimeError(value["value"])
            else:raise RuntimeError("Unknown generation frame")
