"""Shared Mooncake object service and explicitly addressed two-host replica sink.

Immutable logical page versions map to chunk manifests. HEAD confirms all chunks
are present; publication requires complete PUT acknowledgements from both hosts.
Eviction is a cache miss, never permission to invent a restored frontier.
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor, wait
import hashlib
import json
import urllib.error
import urllib.request
from aiohttp import web

LIMIT = (129 << 20) + 4
CHUNK = 8 << 20


def identity(key):
    if not isinstance(key,str) or len(key)!=64 or any(c not in "0123456789abcdef" for c in key):
        raise ValueError("Invalid immutable object identity")
    return key


class Objects:
    def __init__(self, store):
        self.store=store
        self.lock=asyncio.Lock()
        # Bound received bodies independently of Store serialization. Slow
        # upload sockets must not lock out unrelated completed-page GET/HEAD.
        self.uploads=asyncio.Semaphore(4)
        self.downloads=asyncio.Semaphore(4)

    def inspect(self,key):
        raw=self.store.get("online-manifest:"+identity(key))
        if not isinstance(raw,bytes) or not raw: raise KeyError(key)
        manifest=json.loads(raw)
        if not 0<manifest["size"]<=LIMIT: raise ValueError("Invalid object size")
        if sum(n for _,n in manifest["chunks"]) != manifest["size"]: raise ValueError("Invalid chunks")
        for name,n in manifest["chunks"]:
            if self.store.get_size(name)!=n: raise KeyError(key)
        return manifest

    def read_parts(self,key):
        # Caller serializes Store access. Returned bytes are detached from Store
        # lifetime, so assembly/integrity work needs no Store lock.
        manifest=self.inspect(key);chunks=[]
        for name,n in manifest["chunks"]:
            data=self.store.get(name)
            if not isinstance(data,bytes) or len(data)!=n:raise KeyError(key)
            chunks.append(data)
        return manifest,chunks

    @staticmethod
    def assemble(parts):
        manifest,chunks=parts
        data=b"".join(chunks)
        # One full-object digest covers every byte and its order. Rehashing
        # every chunk here duplicates this check without stronger integrity.
        if hashlib.sha256(data).hexdigest()!=manifest["digest"]:raise ValueError("Object corrupted")
        return data

    def read(self,key):
        return self.assemble(self.read_parts(key))

    @staticmethod
    def prepare_write(key,data):
        identity(key)
        if not 0<len(data)<=LIMIT:raise ValueError("Object too large")
        digest=hashlib.sha256(data).hexdigest();chunks=[]
        for begin in range(0,len(data),CHUNK):
            chunk=data[begin:begin+CHUNK]
            chunks.append(("online-chunk:"+hashlib.sha256(chunk).hexdigest(),chunk))
        manifest=json.dumps(dict(size=len(data),digest=digest,
            chunks=[(name,len(chunk)) for name,chunk in chunks])).encode()
        return chunks,manifest,dict(bytes=len(data),digest=digest)

    def commit_write(self,key,prepared):
        chunks,manifest,receipt=prepared
        # Identity check and all Store mutations remain one serialized commit.
        old=self.store.get("online-manifest:"+key)
        if isinstance(old,bytes) and old and json.loads(old)["digest"]!=receipt["digest"]:
            raise ValueError("Immutable State version collision")
        for name,chunk in chunks:
            if self.store.get_size(name)!=len(chunk):
                rc=self.store.put(name,chunk)
                if rc!=0:raise RuntimeError(f"Object put failed: {rc}")
        rc=self.store.put("online-manifest:"+key,manifest)
        if rc!=0:raise RuntimeError(f"Manifest put failed: {rc}")
        return receipt

    def write(self,key,data):
        return self.commit_write(key,self.prepare_write(key,data))

    async def route(self,request):
        key=request.match_info["key"]
        try:
            identity(key)
            if request.method=="PUT":
                async with self.uploads:
                    data=await request.read()
                    prepared=await asyncio.to_thread(self.prepare_write,key,data)
                    async with self.lock:
                        receipt=await asyncio.to_thread(self.commit_write,key,prepared)
                return web.json_response(receipt)
            if request.method=="HEAD":
                async with self.lock:await asyncio.to_thread(self.inspect,key)
                return web.Response()
            async with self.downloads:
                async with self.lock:
                    parts=await asyncio.to_thread(self.read_parts,key)
                data=await asyncio.to_thread(self.assemble,parts)
                response=web.StreamResponse(headers={"Content-Type":"application/octet-stream",
                    "Content-Length":str(len(data))})
                await response.prepare(request)
                await response.write(data);await response.write_eof()
                return response
        except KeyError:
            raise web.HTTPNotFound()
        except ValueError as exc:
            raise web.HTTPBadRequest(text=str(exc))


class PeerObjectSink:
    def __init__(self, urls):
        from urllib.parse import urlsplit
        if len(urls)!=2 or len(set(urls))!=2:raise ValueError("Two distinct replica endpoints required")
        for url in urls:
            p=urlsplit(url)
            if (p.scheme!="http" or p.hostname not in ("10.244.1.16","10.244.2.32")
                    or p.port not in (55581,55586) or p.path or p.query or p.username or p.password):
                raise ValueError("Unqualified private object endpoint")
        self.urls=tuple(urls)
        self.puts=ThreadPoolExecutor(max_workers=2,thread_name_prefix="state-replica")

    def request(self,url,key,method="GET",data=None):
        request=urllib.request.Request(url+"/objects/"+identity(key),data=data,method=method)
        try:
            with urllib.request.urlopen(request,timeout=120) as response:
                payload=response.read(LIMIT+1)
                if len(payload)>LIMIT:raise ValueError("Oversized object response")
                return payload
        except urllib.error.HTTPError as exc:
            if exc.code==404:return None
            raise

    def ensure(self,key):
        present=[self.request(url,key,"HEAD") is not None for url in self.urls]
        if not any(present):return False
        if not all(present):
            data=self.request(self.urls[present.index(True)],key)
            if data is None:return False
            for url,has in zip(self.urls,present):
                if not has:self._put(url,key,data)
        return True

    def _put(self,url,key,data):
        raw=self.request(url,key,"PUT",data)
        receipt=json.loads(raw)
        if receipt!={"bytes":len(data),"digest":hashlib.sha256(data).hexdigest()}:
            raise RuntimeError("Object acknowledgement mismatch")

    def put(self,key,data):
        # Immutable copies may transfer concurrently; publication still requires
        # both validated acknowledgements. Join both even if either one fails.
        futures=[self.puts.submit(self._put,url,key,data) for url in self.urls]
        wait(futures)
        for future in futures:future.result()

    def close(self):
        self.puts.shutdown(wait=True)

    def get(self,key):
        for url in self.urls:
            data=self.request(url,key)
            if data is not None:return data
        raise KeyError("Both object replicas are missing")
