"""Disposable all-rank long-KV dummy observation, never installed in serving."""
import ast
import copy
import math
import time
from pathlib import Path
from pool_state_entry import Worker as BaseWorker

def native_dummy(module):
    # The pinned donor erroneously numpy.repeat()s the whole live Q partition
    # when DP pads a non-key token count. In this disposable probe preserve
    # that partition; graph capacity remains padded. No donor file is edited.
    tree=ast.parse(Path(module.__file__).read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=="NPUModelRunner")
    fn=copy.deepcopy(next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=="_dummy_run"))
    expected="num_scheduled_tokens = num_scheduled_tokens.repeat(num_reqs_padded)"
    padded="num_reqs_padded = batch_desc.num_reqs if batch_desc.num_reqs is not None else num_reqs"
    class Fix(ast.NodeTransformer):
        count=0
        def visit_Assign(self,node):
            if ast.unparse(node)==expected:
                self.count+=1
                return ast.copy_location(ast.Pass(),node)
            if ast.unparse(node)==padded:
                self.count+=1
                return ast.copy_location(ast.parse("num_reqs_padded = num_reqs").body[0],node)
            return node
    fix=Fix();fn=fix.visit(fn)
    if fix.count!=2:raise RuntimeError("Unknown pinned dummy padding source")
    fn.decorator_list=[];fn.name="_theory_dummy"
    ns=dict(vars(module));exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),
                                    module.__file__+"#theory-dummy","exec"),ns)
    return ns[fn.name]

class Worker(BaseWorker):
    def __init__(self,vllm_config,*args,**kwargs):
        from decode_graph_policy import qualify
        qualify(vllm_config)
        super().__init__(vllm_config,*args,**kwargs)
        from decode_graph_policy import install
        install()

    def theory_decode(self,context,fraction,steps,warmup):
        import torch
        from vllm.config import CUDAGraphMode
        from vllm.distributed import get_ep_group
        from vllm_ascend.worker import model_runner_v1 as module
        from betterscale.live.llm.qwen35.state import AttentionState
        runner=self.model_runner;root=runner._live_state_root
        if runner.input_batch.num_reqs:raise RuntimeError("Dummy requires empty disposable model")
        block=runner.kv_cache_config.kv_cache_groups[0].kv_cache_spec.block_size
        if block!=2048:raise ValueError("Unexpected logical FA page")
        geometry=root.geometry
        per_token=(geometry.layer_types.count("full_attention")+geometry.draft_layers)*geometry.kv_heads*geometry.attention_head_dim*4
        allocated=sum(s.tensor.numel()*s.tensor.element_size() for _,s in root.named_states())
        dense=root.capacity.token_pages*root.capacity.page_tokens*per_token
        fixed=allocated-dense
        per_request=math.ceil((context+3)/block)
        batch=int((fraction*allocated-fixed)//(per_request*block*per_token))
        if not 1<=batch<=runner.scheduler_config.max_num_seqs:raise ValueError("Dummy batch outside execution capacity")
        if batch*per_request>=runner.kv_cache_config.num_blocks:raise ValueError("Dummy includes null/overflow page")
        # Unique physical pages per request, including draft attention. No null
        # page aliasing that would make a nominal 100K run a cache-hot toy.
        for leaf in [*root.target.values(),root.draft]:
            if isinstance(leaf,AttentionState):
                for t in leaf.numerical_tensors():t.zero_()
        for row in range(batch):
            ids=list(range(1+row*per_request,1+(row+1)*per_request))
            runner.input_batch.block_table[0].add_row(ids,row)
            gdn=runner.input_batch.block_table[runner._live_gdn_group]
            gdn.get_cpu_tensor()[row].fill_(row)
        runner.input_batch.block_table.commit_block_table(batch)
        runner.input_batch.req_ids[:]=[f"theory-{i}" for i in range(batch)]
        runner.input_batch.req_id_to_index.update({rid:i for i,rid in enumerate(runner.input_batch.req_ids)})
        if runner.input_batch.num_reqs!=batch:raise RuntimeError("Dummy live-row count mismatch")
        runner.input_batch.num_computed_tokens_cpu_tensor[:batch].fill_(context-3)
        runner.input_batch.num_prompt_tokens_cpu_tensor[:batch].fill_(context-3)
        root.remaining_outputs.tensor.fill_(1000000)
        root.continuation.selection.tensor.fill_(1);root.conv_selection.tensor.fill_(1)
        runner.input_ids.gpu.random_(0,10000)
        runner.positions[:batch*3].copy_(torch.arange(context-3,context,device=runner.device).repeat(batch))
        runner.drafter.input_ids.random_(0,10000)
        runner._pd_target_only_ready=False
        call=native_dummy(module)
        def step(i):
            runner.gdn_query_start_loc.cpu.fill_(batch*3)
            runner.gdn_query_start_loc.cpu[0]=0
            runner.query_start_loc.cpu.zero_()
            runner._owned_next_bank=i%2
            runner._elastic_dummy=True
            try:
                return call(runner,batch*3,cudagraph_runtime_mode=CUDAGraphMode.FULL,
                            force_attention=True,uniform_decode=True,profile_seq_lens=context)
            finally:runner._elastic_dummy=False
        with torch.inference_mode():
            for i in range(warmup):step(i)
            torch.npu.synchronize()
            # Receipts prove actual consumer lengths and nonaliasing CPU table;
            # both target and draft consume this device length plane.
            lengths=runner.seq_lens[:batch].cpu().tolist()
            if lengths!=[context]*batch:raise RuntimeError("Dummy did not publish long device lengths")
            events=[torch.npu.Event(enable_timing=True) for _ in range(steps+1)]
            host=[]
            for i in range(steps):
                host.append(time.perf_counter_ns());events[i].record();step(i)
            events[-1].record();torch.npu.synchronize()
            gaps=[events[i].elapsed_time(events[i+1]) for i in range(steps)]
        return dict(rank=get_ep_group().rank_in_group,batch=batch,context=context,
            steps=steps,warmup=warmup,step_ms=gaps,host_begin_ns=host,
            allocated_state_bytes=allocated,fixed_state_bytes=fixed,
            dense_pool_bytes=dense,logical_blocks=runner.kv_cache_config.num_blocks,
            active_dense_bytes=batch*per_request*block*per_token,
            occupied_fraction=(fixed+batch*per_request*block*per_token)/allocated,
            per_token_rank_bytes=per_token,logical_blocks_per_request=per_request,
            device_lengths=lengths,graph_capacity=runner._owned_frame.metas[next(iter(runner._owned_frame.metas))].tokens,
            scope="Synthetic native target+MTP dummy loop, not sampled output or E2E goodput; no live scheduler/admission/State I/O.")
