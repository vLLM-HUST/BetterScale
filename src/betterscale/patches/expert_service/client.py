"""Public Worker lifecycle adapter for target and draft routed-expert service."""
import json,os,re
from pathlib import Path
import torch
from .model_geometry import GEOMETRY as G


def model_domains(worker):
    yield 0, worker.model_runner.get_model()
    if G.total_layers == 41:
        draft = worker.model_runner.drafter.model
        yield 40, draft


def load_model(self, native):
    role_only = G.name == 'qwen35'
    if role_only:
        assert os.environ.get('BETTERSCALE_EXPERT_PERSISTENT')=='1', 'Qwen35 requires the persistent routed-only boundary'
        from .attention_role_weights import attention_only_weights
        with attention_only_weights() as role_receipt:
            result = native()
    else:
        result = native()
    assert self.vllm_config.parallel_config.tensor_parallel_size==1
    assert not self.vllm_config.parallel_config.enable_expert_parallel
    graph=not self.vllm_config.model_config.enforce_eager
    persistent=os.environ.get('BETTERSCALE_EXPERT_PERSISTENT')=='1'
    assert not graph or persistent or os.environ.get('BETTERSCALE_EXPERT_DEVICE_CLIENT')=='1','FULL requires device client'
    from .persistent_remote import PersistentRemote as MatrixRemote
    from .placement import Placement
    from vllm_ascend.ops.fused_moe.moe_comm_method import MoECommMethod,FusedExpertsResult
    self.native_remote=MatrixRemote(Path(os.environ['BETTERSCALE_EXPERT_NATIVE_CONTROL']),
        Placement(os.environ['BETTERSCALE_EXPERT_NATIVE_PLACEMENT'],int(os.environ['BETTERSCALE_EXPERT_NATIVE_OWNERS'])),
        int(os.environ['BETTERSCALE_EXPERT_NATIVE_SOURCE']))
    self.native_remote.phase='native-prefill' if graph else 'native-eager';self.native_shadows={}
    models=list(model_domains(self));weights={}
    if graph:self.model_runner.use_aclgraph=True
    self.native_retain=os.environ.get('BETTERSCALE_EXPERT_NATIVE_RETAIN_WEIGHTS','1')=='1'
    self.native_release={}
    if role_only:
        self.native_release = role_receipt
        placeholders = [v for _,model in models for n,v in model.named_parameters() if '.mlp.experts.' in n]
        assert len(placeholders)==2*len(G.layers)
        assert all(v.device.type=='cpu' and v.untyped_storage().nbytes()==2 for v in placeholders)
        self.native_release['expert_parameter_storage_bytes']=sum(v.untyped_storage().nbytes() for v in placeholders)
        print('expert role-only weight receipt',json.dumps(self.native_release,sort_keys=True),flush=True)
    if not self.native_retain:
        qualified=Path(os.environ['BETTERSCALE_EXPERT_NATIVE_QUALIFICATION'])
        receipt=json.loads(qualified.read_text())
        if persistent:
            assert receipt.get('host_forward_requests')==0 and receipt.get('expert_frame_rows')==self.native_remote.rows
            assert receipt.get('persistent_build')==self.native_remote.kernels.identity,'qualify the exact resident/client binary closure before releasing native weights'
            assert receipt.get('return_mode')==self.native_remote.return_mode
            assert receipt.get('fine_pack')==(os.environ.get('BETTERSCALE_EXPERT_FINE_PACK','1')=='1')
            assert receipt.get('persistent_shared_overlap') is True
        assert set(receipt['native_shadows'])=={str(i) for i in G.layers}
        assert max(r['relative_l2'] for r in receipt['native_shadows'].values())<=.02
    for offset,model in models:
        for name,value in model.named_parameters():
            match=re.search(r'(?:^|\.)layers\.(\d+)\.mlp\.experts(?:\.routed_experts)?\.w13_weight$',name)
            if match:weights[id(value)]=offset+int(match.group(1))
    assert set(weights.values())==set(G.layers),weights
    if persistent:
        # Preserve native shared MLP arithmetic, moving only its scheduling
        # between device publication and collect (the donor client seam).
        import types
        self.native_shared_out={};self.native_shared_modules={};self.native_shared_originals=[]
        for offset,model in models:
            for name,module in model.named_modules():
                match=re.search(r'(?:^|\.)layers\.(\d+)\.mlp\.',name)
                if match and hasattr(module,'_forward_shared_experts'):
                    layer=offset+int(match.group(1))
                    assert layer not in self.native_shared_modules
                    assert module._shared_experts is not None and not module.multistream_overlap_shared_expert
                    self.native_shared_modules[layer]=module
                    self.native_shared_originals.append((module,module._forward_shared_experts))
                    def take_shared(module_self,hidden,events,layer=layer):
                        return self.native_shared_out.pop(layer)
                    module._forward_shared_experts=types.MethodType(take_shared,module)
        assert set(self.native_shared_modules)==set(G.layers),self.native_shared_modules.keys()
        def shared_between_submit_collect(layer,hidden):
            module=self.native_shared_modules[layer]
            self.native_shared_out[layer]=module._shared_experts_part2(hidden,module._shared_experts_part1(hidden))
        self.native_remote.shared_callback=shared_between_submit_collect
    original=MoECommMethod.fused_experts
    self.native_original=original
    owner=self
    def client(method,fused_experts_input):
        frame=fused_experts_input
        assert type(method).__name__=='AllGatherCommImpl',type(method).__name__
        assert frame.hidden_states.dtype==torch.bfloat16 and frame.weights.w1.dtype==torch.bfloat16
        layer=weights[id(frame.weights.w1)]
        # Native routing already produced and rounded these exact probabilities.
        before=torch.npu.current_stream().record_event()
        output=owner.native_remote(layer,frame.hidden_states,frame.topk_ids,frame.topk_weights.float())
        complete=torch.npu.current_stream().record_event()
        if owner.native_retain and layer not in owner.native_shadows:
            if role_only:
                from .attention_role_weights import native_layer_reference
                expected=native_layer_reference(original,method,frame,layer)
            else:
                expected=original(method,frame).routed_out
            actual=output.float();expected=expected.float()
            assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
            relative=float(torch.linalg.vector_norm(actual-expected)/torch.linalg.vector_norm(expected).clamp_min(1e-12))
            assert relative<=.02,('native routed shadow',layer,relative)
            owner.native_shadows[layer]=dict(rows=output.shape[0],relative_l2=relative)
        # Conservative completion, not a claim of preserved GMM2/shared overlap.
        return FusedExpertsResult(routed_out=output,before_dispatch_evt=before,
            before_gmm2_evt=complete,before_combine_evt=complete,swiglu_limit=frame.swiglu_limit)
    MoECommMethod.fused_experts=client
    print('native routed-only boundary installed',sorted(weights.values()),flush=True)
    return result

def expert_receipt(self):
    if self.native_retain:assert set(self.native_shadows)==set(G.layers),self.native_shadows.keys()
    return dict(peer_generations=(self.native_remote.receipt()['peer_generations'] if hasattr(self.native_remote,'receipt') else {o:p['generation'] for o,p in self.native_remote.peers.items()}),
                native_shadows=self.native_shadows,native_expert_weights_retained=self.native_retain and G.name!='qwen35',
                native_layer_shadow_enabled=self.native_retain,
                native_weight_release=self.native_release,
                calls=self.native_remote.calls,scope='native routed-only boundary',
                persistent_shared_overlap=hasattr(self,'native_shared_modules'),
                **{k:v for k,v in (self.native_remote.receipt() if hasattr(self.native_remote,'receipt') else {}).items() if k!='peer_generations'})

def close_expert_service(self):
    """Drain IPC while the native engine is alive, not in its short exit grace."""
    if getattr(self,'native_remote',None) is not None:
        counts=(self.native_remote.receipt()['peer_generations'] if hasattr(self.native_remote,'receipt')
                else {o:p['generation'] for o,p in self.native_remote.peers.items()})
        self.native_remote.close();self.native_remote=None
        from vllm_ascend.ops.fused_moe.moe_comm_method import MoECommMethod
        MoECommMethod.fused_experts=self.native_original
        for module,original in getattr(self,'native_shared_originals',[]):module._forward_shared_experts=original
        return dict(drained=True,peer_generations=counts)
    return dict(drained=True,already_closed=True)
