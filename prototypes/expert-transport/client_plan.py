"""Task-local native routing producer; paired only with the plan prototype ABI."""

def install():
    import torch
    import torch_npu
    from betterscale.patches.expert_service import persistent_remote as module
    BaseBank,BaseRemote=module.Bank,module.PersistentRemote
    class PlanBank(BaseBank):
        def __init__(self,remote,owner,rows):
            super().__init__(remote,owner,rows)
            self.config=torch.cat((self.config,torch.zeros(2,dtype=torch.int64,device='npu')))
            self.plan_input=torch.zeros(rows,32,dtype=torch.bfloat16,device='npu')
    class PlanRemote(BaseRemote):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            assert self.abi.get('client_route_plan')=='native-v2-cap1'
            assert self.placement.mode=='layer' and self.abi['sources_per_wave']==1
        def __call__(self,layer,x,ids,probs):
            n=x.shape[0];assert 0<n<=self.rows,'prototype requires one explicit frame'
            if n<self.abi.get('client_route_plan_min_rows',1):return super().__call__(layer,x,ids,probs)
            owner=self.placement.targets(layer)[0];bank=self.bank(owner,n)
            expanded,inverse,counts,scale=torch_npu.npu_moe_init_routing_v2(
                bank.plan_input[:n],ids.int(),active_num=n*8,expert_num=256,
                expert_tokens_num_type=1,expert_tokens_num_flag=True,quant_mode=-1,
                active_expert_range=[0,256],row_idx_type=0)
            bank.config[16]=inverse.data_ptr();bank.config[17]=counts.data_ptr()
            result=super().__call__(layer,x,ids,probs)
            # References span every native enqueue; same-stream allocator lifetime.
            del expanded,inverse,counts,scale
            return result
    module.Bank=PlanBank;module.PersistentRemote=PlanRemote
