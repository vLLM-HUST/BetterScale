"""BF16 pack → native shared MLP → collect, with fixed-order weighted reduction.

Every frame, including eager prefill, is device-published. No host RPC or stream
synchronization in forward. Internal expert frames do not split attention/KV.
"""
import os
import torch
from .checkpoint import H,K
from .control import connect
from .ipc_acl import PrefixCopyACL
from .runtime import LIB
from .persistent_engine import geometry,extents,Kernels,return_mode

class Bank:
    def __init__(self,remote,owner,rows):
        peer=remote.peers[owner]
        self.x=torch.empty(rows,H,dtype=torch.bfloat16,device='npu')
        # Collect reads aligned metadata lines, including the final token's tail.
        self.id_storage=torch.zeros((rows*K+31)//8*8,dtype=torch.int32,device='npu')
        self.probs=torch.zeros((rows*K+47)//16*16,dtype=torch.bfloat16,device='npu')
        self.output=torch.empty_like(self.x)
        self.config=torch.tensor([peer['local'],peer['output'],0,0,0,0,rows,
            peer['counter'].data_ptr(),1,20000000,0,self.probs.data_ptr(),self.output.data_ptr(),0,250,0],dtype=torch.int64,device='npu')
        if remote.placement.mode=='expert' and owner==0:
            for other in range(remote.placement.owners):self.config[1+other]=remote.peers[other]['output']

class PersistentRemote:
    def __init__(self,directory,placement,source):
        assert placement.mode in ('layer','expert')
        self.root,self.abi=geometry();self.rows=self.abi['rows'];self.source=source
        self.placement=placement;self.api=PrefixCopyACL(LIB);self.peers={};self.banks={}
        self.calls=[];self.phase='persistent';self.python_submissions=0;self.shared_callback=None
        self.return_mode=return_mode()
        source_bytes,output_bytes=extents(self.abi);contract=dict(self.abi,owners=placement.owners,return_mode=self.return_mode)
        zero=torch.zeros(64,dtype=torch.int32,device='npu')
        for owner in range(placement.owners):
            ch=connect(directory/f'e{owner}.sock');ch.sock.settimeout(self.abi.get('service_lifetime_seconds',self.abi['persistent_launch_timeout_us']//1000000))
            ch.send(dict(op='hello',source=source,pid=self.api.pid(),contract=contract))
            hello=ch.expect('window');assert hello['contract']==contract and hello['owner']==owner
            if self.return_mode=='pull':
                output_key=hello['key'].encode();output=self.api.import_memory(output_key)
            else:
                assert hello['key'] is None
                output=self.api.allocate_staging(output_bytes)
                self.api.copy(torch.npu.current_stream().npu_stream,output,zero.data_ptr(),256);torch.npu.synchronize()
                output_key=self.api.export(output,output_bytes,(hello['pid'],))
            local=self.api.allocate_staging(source_bytes)
            self.api.copy(torch.npu.current_stream().npu_stream,local,zero.data_ptr(),256);torch.npu.synchronize()
            key=self.api.export(local,source_bytes,(hello['pid'],))
            ch.send(dict(op='source',key=key.decode(),output_key=output_key.decode() if self.return_mode=='push' else None));ch.expect('registered')
            self.peers[owner]=dict(ch=ch,local=local,key=key,output=output,output_key=output_key,
                                   counter=torch.zeros(8,dtype=torch.int32,device='npu'),generation=0)
        self.kernels=Kernels()
        self.pack=self.kernels.load('neural_pack');self.publish=self.kernels.load('neural_publish')
        self.collect=self.kernels.load('neural_collect_reduced' if self.abi.get('combined_return') else 'neural_collect_pipelined');self.retire=self.kernels.load('neural_retire')
        self.promote=self.kernels.load('neural_promote')
        for owner in self.peers:
            for rows in sorted({self.rows,*range(1,int(os.environ.get('BETTERSCALE_EXPERT_GRAPH_BATCH','4'))+1)}):
                self.banks[owner,rows]=Bank(self,owner,rows)
        torch.npu.synchronize()
        for peer in self.peers.values():peer['ch'].expect('ready')

    def prepare_graph_rows(self,rows):
        """Explicit graph-owned storage; ordinary eager shapes never enter this cache."""
        assert not torch.npu.is_current_stream_capturing()
        for owner in self.peers:
            for offset in range(0,rows,self.rows):
                n=min(self.rows,rows-offset)
                if (owner,n) not in self.banks:self.banks[owner,n]=Bank(self,owner,n)

    def bank(self,owner,n):
        if (owner,n) in self.banks:
            b=self.banks[owner,n]
        else:
            assert not torch.npu.is_current_stream_capturing(),'prepare graph-owned row banks before capture'
            b=self.banks[owner,self.rows]
        if b is self.banks[owner,self.rows]:b.config[6]=n
        return b

    def __call__(self,layer,x,ids,probs):
        assert x.ndim==2 and x.shape[1]==H and ids.shape==probs.shape==(x.shape[0],K)
        owners=self.placement.targets(layer);results=[];priority=0
        try:
            from vllm.forward_context import get_forward_context,is_forward_context_available
            if is_forward_context_available():
                metadata=get_forward_context().attn_metadata
                if isinstance(metadata,dict):priority=int(any(getattr(m,'num_prefills',0)>0 for m in metadata.values()))
        except ImportError:pass
        for offset in range(0,x.shape[0],self.rows):
            n=min(self.rows,x.shape[0]-offset)
            banks=[self.bank(owner,n) for owner in owners]
            for b in banks:
                self.python_submissions+=1
                b.config[5]=layer;b.config[15]=priority
                b.x[:n].copy_(x[offset:offset+n]);b.id_storage[:n*K].copy_(ids[offset:offset+n].flatten())
                b.probs[:n*K].copy_(probs[offset:offset+n].flatten())
                self.kernels.call(self.pack,b.config,b.x,b.id_storage,16)
                self.kernels.call(self.publish,b.config,b.x,b.id_storage)
            if offset==0 and self.shared_callback is not None:self.shared_callback(layer,x)
            for b in banks:
                if priority:self.kernels.call(self.promote,b.config,b.x,b.id_storage)
            first=banks[0]
            self.kernels.call(self.collect,first.config,first.x,first.id_storage,16)
            for b in banks:self.kernels.call(self.retire,b.config,b.x,b.id_storage)
            results.append(first.output[:n].clone()) # consumers outlive bank reuse
        return results[0] if len(results)==1 else torch.cat(results)

    def receipt(self):
        torch.npu.synchronize()
        counts={o:int(p['counter'].cpu()[0]) for o,p in self.peers.items()}
        assert all(n>=0 for n in counts.values()),counts
        bank_bytes=sum(t.numel()*t.element_size() for b in self.banks.values()
                       for t in (b.x,b.id_storage,b.probs,b.output,b.config))
        return dict(placement=self.placement.mode,owners=self.placement.owners,peer_generations=counts,device_generations=counts,python_submissions=self.python_submissions,
                    bank_count=len(self.banks),bank_tensor_bytes=bank_bytes,
                    peak_allocated_bytes=torch.npu.max_memory_allocated(),
                    transport='persistent BF16 AIV/AIC; parallel pack; '+('server fixed-order combine; copy collect' if self.abi.get('combined_return') else 'pipelined fixed-order collect'),
                    persistent_build=self.kernels.identity,fine_pack=os.environ.get('BETTERSCALE_EXPERT_FINE_PACK','1')=='1',
                    host_forward_requests=0,expert_frame_rows=self.rows,return_mode=self.return_mode,combined_return=self.abi.get('combined_return',False))
    def close(self):
        receipt=self.receipt();stream=torch.npu.current_stream()
        eos=[]
        for owner,peer in self.peers.items():
            count=receipt['peer_generations'][owner];peer['generation']=count
            end=torch.zeros(8,dtype=torch.int32,device='npu');end[0]=-(count+1);eos.append(end)
            self.api.copy(stream.npu_stream,peer['local'],end.data_ptr(),32)
        stream.synchronize()
        for owner,peer in self.peers.items():peer['ch'].send(dict(op='drain',generation=peer['generation']))
        for peer in self.peers.values():peer['ch'].expect('drained')
        for peer in self.peers.values():
            if self.return_mode=='pull':self.api.close_mapping(peer['output_key'])
            peer['ch'].send(dict(op='unmapped'))
        for peer in self.peers.values():
            peer['ch'].expect('released')
            if self.return_mode=='push':
                self.api.close_mapping(peer['output_key']);self.api.free_staging(peer['output'])
            self.api.close_mapping(peer['key']);self.api.free_staging(peer['local']);peer['ch'].close()
        self.kernels.close()
