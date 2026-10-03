"""Front-end host-byte admission and idle LRU, separate from device seats.

Only an actually reserved private arena is admitted here. Reservations cover
the old checkpoint plus P and D incremental publications until obsolete versions
retire. Shared pages across different sessions are deliberately not discounted.
All bookkeeping runs on the coordinator event loop; no device/global-DP waits.
"""
import asyncio
from collections import OrderedDict
from dataclasses import dataclass


class HostCapacityError(ValueError):
    """Request cannot fit even a cold turn; no numerical writer was admitted."""


def page_count(tokens):
    return (tokens-2+2048)//2048


@dataclass
class Entry:
    owners: tuple
    checkpoint: dict
    charge: int
    active: bool = False


class HostCache:
    def __init__(self,coordinator,limits,resident_bytes,page_bytes):
        if (set(limits)!={(k,i) for k in ("P","D") for i in range(4)}
                or any(type(v) is not int or v<=0 for v in limits.values())
                or any(type(v) is not int or v<=0 for v in (resident_bytes,page_bytes))):
            raise ValueError("complete positive host-cache geometry required")
        self.c=coordinator
        self.limits=limits
        self.resident=(resident_bytes+63)//64*64
        self.page=(page_bytes+63)//64*64
        self.used={owner:0 for owner in limits}
        self.entries=OrderedDict()
        self.evicting=set()
        self.evictions=0
        self.changed=asyncio.Event()

    @classmethod
    async def connect(cls,c):
        """Cold/explicitly retired pools only; reject unaccounted old fixtures."""
        limits={};shapes=set()
        for kind,peer,count in (("P",c.p,4),("D",c.d,1)):
            for instance in range(count):
                rows=await peer.rpc(instance,"memory",owner=0)
                expected={(o,t) for o in range(4 if kind=="D" else 1) for t in range(2)}
                if {(r["owner"],r["rank"]) for r in rows}!=expected or len(rows)!=len(expected):
                    raise ValueError("host admission needs a complete rank-memory receipt")
                by_group={}
                for row in rows:
                    arena=row.get("host_arena",{});pool=row.get("pool",{})
                    geometry=row.get("host_geometry",{})
                    if (arena.get("backend")!="rank-private-vmm-arena"
                            or arena.get("used_bytes")!=0 or arena.get("live_buffers")!=0
                            or any(pool.get(k)!=0 for k in ("bytes","objects","checkpoints","readers","referenced"))):
                        raise ValueError("host admission requires empty, physically reserved rank pools")
                    shape=(geometry["resident_frame_bytes"],geometry["page_frame_bytes"])
                    shapes.add(shape)
                    # Older qualified workers retain the preceding scratch
                    # until the next allocation: one resident then page frames.
                    # Cover that overlap too, independently of persistent State.
                    resident,page=((n+63)//64*64 for n in shape)
                    scratch=geometry["max_transfers"]*max(resident+page,2*page)
                    # Operational fragmentation headroom, not a mathematical
                    # guarantee for an arbitrary variable-size arena workload.
                    budget=arena["reserved_bytes"]*4//5-scratch
                    group=instance if kind=="P" else row["owner"]
                    by_group.setdefault(group,[]).append(budget)
                for group,values in by_group.items():
                    if len(values)!=2:raise ValueError("missing TP budget quorum")
                    limits[kind,group]=min(values)
        if len(shapes)!=1:raise ValueError("P/D rank State frame geometry differs")
        with c.directory.transaction() as db:
            if db.execute("SELECT 1 FROM sessions WHERE manifest IS NOT NULL LIMIT 1").fetchone():
                raise ValueError("host admission cannot attach to unaccounted manifests")
        return cls(c,limits,*shapes.pop())

    def load(self,kind,index):
        owner=(kind,index)
        return self.used[owner]/self.limits[owner]

    def footprint(self,checkpoint):
        if checkpoint["block_count"]!=page_count(len(checkpoint["tokens"])):
            raise ValueError("checkpoint frontier differs from host admission geometry")
        return self.resident+checkpoint["block_count"]*self.page

    def peak(self,entry,prompt,n):
        # A resident snapshot and writable tail are private per generation.
        # Only a prefix's sealed pages (including MTP lookahead) can be shared.
        common=0
        if entry is not None:
            cp=entry.checkpoint
            if prompt[:len(cp["tokens"])]==cp["tokens"]:
                common=max(0,(len(cp["tokens"])-2)//2048)
        p_count=page_count(len(prompt)+1)
        extra=self.resident+(p_count-common)*self.page
        if n>1:
            p_sealed=max(0,(len(prompt)-1)//2048)
            extra+=self.resident+(page_count(len(prompt)+n)-p_sealed)*self.page
        return (entry.charge if entry is not None else 0)+extra

    async def acquire(self,session,prompt,n):
        owners=(("P",self.c.p_affinity[session]),)
        if session in self.c.d_affinity:owners+=(("D",self.c.d_affinity[session]),)
        if any(self.peak(None,prompt,n)>self.limits[o] for o in owners):
            raise HostCapacityError("one turn exceeds reserved host-cache capacity")
        while True:
            self.changed.clear()
            if self.c.failure:raise RuntimeError(self.c.failure)
            entry=self.entries.get(session)
            if entry is not None and (entry.active or not set(entry.owners).issubset(owners)):
                raise RuntimeError("host-cache session ownership changed")
            peak=self.peak(entry,prompt,n)
            old={o:entry.charge if entry is not None and o in entry.owners else 0 for o in owners}
            short={o for o in owners if self.used[o]+peak-old[o]>self.limits[o]}
            if not short and session not in self.evicting:
                for o in owners:self.used[o]+=peak-old[o]
                self.entries[session]=Entry(owners,entry.checkpoint if entry else {},peak,True)
                return
            if entry is None and any(peak>self.limits[o] for o in owners):
                raise HostCapacityError("one turn exceeds reserved host-cache capacity")
            # A queued host waiter has not claimed a numerical writer or P seat.
            # It may lose an idle cache copy and then prefill its complete prompt.
            victim=next((s for s,e in self.entries.items()
                if not e.active and s not in self.evicting
                and (s not in self.c.inflight or s in self.c.host_waiters)
                and short.intersection(e.owners)),None)
            if victim is None:
                await self.changed.wait()
                continue
            self.evicting.add(victim)
            try:
                if not await self.c.evict_idle(victim):
                    raise RuntimeError("LRU entry lost its published checkpoint")
            finally:
                self.evicting.remove(victim)
                self.changed.set()

    def forgot(self,session):
        entry=self.entries[session]
        if entry.active:raise RuntimeError("active host reservation cannot be evicted")
        self.entries.pop(session)
        self.evictions+=1
        for o in entry.owners:self.used[o]-=entry.charge
        self.changed.set()

    def finish(self,session,checkpoint):
        entry=self.entries[session]
        cost=self.footprint(checkpoint)
        if not entry.active or cost>entry.charge:
            raise RuntimeError("committed State exceeded its host reservation")
        for o in entry.owners:self.used[o]+=cost-entry.charge
        self.entries[session]=Entry(entry.owners,checkpoint,cost)
        self.entries.move_to_end(session)
        self.changed.set()

    def snapshot(self):
        return dict(used={f"{k}{i}":n for (k,i),n in self.used.items()},
                    limits={f"{k}{i}":n for (k,i),n in self.limits.items()},
                    active=sum(e.active for e in self.entries.values()),
                    cached=sum(not e.active for e in self.entries.values()),
                    evicting=len(self.evicting),evictions=self.evictions)
