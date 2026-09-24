"""Expert-only process: bootstrap, one persistent launch pair, device EOF drain."""
import ctypes as C
import json,os,threading,time
from pathlib import Path
import torch
import torch_npu
from .checkpoint import Checkpoint
from .control import Channel,listen
from .ipc_acl import PrefixCopyACL
from .runtime import LIB
from .persistent_engine import Engine,geometry,extents,return_mode


def serve(directory,placement,owner,sources,layers=None):
    assert placement.mode in ('layer','expert') and 1<=sources<=7
    root,abi=geometry();source_bytes,output_bytes=extents(abi)
    lifetime=abi.get('service_lifetime_seconds',abi['persistent_launch_timeout_us']//1000000)
    if abi.get('external_watchdog'):
        assert os.environ.get('BETTERSCALE_EXPERT_EXTERNAL_WATCHDOG')=='1','disabled kernel timer requires an admitted bounded supervisor'
    torch_npu.npu.config.allow_internal_format=True
    acl=C.CDLL(LIB);acl.aclrtSetOpExecuteTimeOut.argtypes=[C.c_uint32]
    weights={};cp=Checkpoint()
    for layer in (layers if layers is not None else placement.layers(owner)):
        assert layer in placement.layers(owner)
        gate,up,down=cp.experts(layer,bounds=placement.experts(owner))
        w13=torch.cat((gate,up),dim=1).transpose(1,2).contiguous()
        w2=down.transpose(1,2).contiguous()
        weights[layer]=(torch_npu.npu_format_cast(w13,29),torch_npu.npu_format_cast(w2,29))
        del gate,up,down,w13,w2
    torch.npu.synchronize();torch.npu.empty_cache()
    # torch-npu lazily installs its default execution timeout on first tensor
    # allocation. Set these AFTER that initialization and BEFORE graph capture:
    # graph notify-waits are distinct from the kernel launch timeout attribute.
    acl.aclrtSetOpWaitTimeout.argtypes=[C.c_uint32]
    assert acl.aclrtSetOpExecuteTimeOut(lifetime)==0
    assert acl.aclrtSetOpWaitTimeout(lifetime)==0
    print('persistent deadlines',dict(execute_seconds=lifetime,wait_seconds=lifetime),flush=True)
    print('persistent weights ready',list(weights),flush=True)
    api=PrefixCopyACL(LIB);listener=listen(directory/f'e{owner}.sock');listener.listen(sources);listener.settimeout(lifetime)
    mode=return_mode();peers={};contract=dict(abi,owners=placement.owners,return_mode=mode)
    zero=torch.zeros(64,dtype=torch.int32,device='npu')
    for _ in range(sources):
        ch=Channel(listener.accept()[0]);ch.sock.settimeout(lifetime);hello=ch.expect('hello');source=hello['source']
        assert hello['contract']==contract and type(source) is int and 0<=source<sources and source not in peers
        output=key=None
        if mode=='pull':
            output=api.allocate_staging(output_bytes)
            api.copy(torch.npu.current_stream().npu_stream,output,zero.data_ptr(),256);torch.npu.synchronize()
            key=api.export(output,output_bytes,(hello['pid'],))
        ch.send(dict(op='window',key=key.decode() if key else None,pid=api.pid(),contract=contract,owner=owner))
        source=hello['source'];message=ch.expect('source')
        source_key=message['key'].encode();pointer=api.import_memory(source_key)
        if mode=='push':
            key=message['output_key'].encode();output=api.import_memory(key)
        peers[source]=dict(ch=ch,output=output,key=key,source_key=source_key,pointer=pointer)
        ch.send(dict(op='registered'))
    listener.close()
    closed=torch.zeros(64,dtype=torch.int32,device='npu');closed[0]=-1
    unused=torch.zeros(64,dtype=torch.int32,device='npu')
    engine=Engine([peers[i]['pointer'] if i in peers else closed.data_ptr() for i in range(7)],
                  [peers[i]['output'] if i in peers else unused.data_ptr() for i in range(7)],weights,owner=owner if placement.mode=='expert' else 0)
    engine.start()
    for peer in peers.values():peer['ch'].send(dict(op='ready'))
    print('persistent serving, host waits only for drain',flush=True)
    stop=threading.Event();observer=None
    if os.environ.get('BETTERSCALE_EXPERT_PERSISTENT_DIAGNOSTICS')=='1':
        def observe():
            torch.npu.set_device(0)
            with (Path(os.environ['BETTERSCALE_EXPERT_PERSISTENT_DIAGNOSTICS_DIR'])/f'persistent-owner{owner}-control.jsonl').open('w') as log:
                while not stop.wait(1):
                    state=engine.control.cpu().tolist()
                    log.write(json.dumps(dict(time=time.time(),control=state,trace=engine.trace.cpu().tolist()))+'\n');log.flush()
                    print('persistent control', {i:state[i][:12] for i in (0,1,2,43)},flush=True)
        observer=threading.Thread(target=observe);observer.start()
    try:
        expected={i:p['ch'].expect('drain')['generation'] for i,p in peers.items()}
        receipt=engine.finish()
    finally:
        stop.set()
        if observer is not None:observer.join()
    assert receipt['completed_counts']==[expected.get(i,0) for i in range(7)],receipt['completed_counts']
    for peer in peers.values():peer['ch'].send(dict(op='drained'))
    for peer in peers.values():
        peer['ch'].expect('unmapped');api.close_mapping(peer['source_key']);api.close_mapping(peer['key'])
        if mode=='pull':api.free_staging(peer['output'])
        peer['ch'].send(dict(op='released'));peer['ch'].close()
    engine.close()
    return dict(status='PASS',return_mode=mode,owner=owner,completed=expected,layers=list(weights),
                placement=placement.mode, expert_bounds=placement.experts(owner), weight_bytes=sum(t.numel()*t.element_size() for pair in weights.values() for t in pair),**receipt)
