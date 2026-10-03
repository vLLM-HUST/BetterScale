"""Worker-owned Store connections; per-transfer registrations remain fenced."""


def acquire(owner, port, retained, factory):
    if type(port) is not int or port not in (*range(55421,55429),*range(55431,55439)):
        raise ValueError('Unqualified Store endpoint')
    entries=getattr(owner,'_pd_store_clients',None)
    if entries is None:entries={};owner._pd_store_clients=entries
    entry=entries.get(port)
    if entry is None:
        client=factory()
        try:
            rc=client.setup(f'127.0.0.1:{port}','http://127.0.0.1:55402/metadata',
                            0,64*1024**2,'tcp','','127.0.0.1:55401')
            if rc!=0:raise RuntimeError(f'Store setup failed: {rc}')
        except BaseException:
            client.close();raise
        entry=dict(client=client,busy=False,retained=None);entries[port]=entry
    if entry['busy']:raise RuntimeError('Store client still owns a transfer')
    entry.update(busy=True,retained=retained)
    return entry['client']


def release(owner,port):
    entry=owner._pd_store_clients[port]
    if not entry['busy']:raise RuntimeError('Store transfer already released')
    # Caller has drained DMA/Store work and unregistered every buffer.
    entry.update(busy=False,retained=None)


def close_worker(worker):
    entries=getattr(worker.model_runner,'_pd_store_clients',{})
    if any(e['busy'] for e in entries.values()):
        raise RuntimeError('Cannot close Store while a transfer owns buffers')
    for port in list(entries):
        entries[port]['client'].close();del entries[port]
    return True


def close_core(core):
    from model_checkpoint import idle
    idle(core)
    if getattr(core,'_pd_import',None) is not None or getattr(core,'_pd_async_export',None) is not None:
        raise RuntimeError('Cannot close Store with quarantined transfer')
    result=core.model_executor.collective_rpc('pd_close_store_clients')
    if result!=[True,True]:raise RuntimeError('Missing Store close acknowledgements')
    return True
