"""Diagnostic State guard, imported only after native plugin initialization."""
from online_entry import Worker as BaseWorker

class Worker(BaseWorker):
    def idle_guard_arm(self,active_owner):
        from betterscale.live.llm.qwen35.state import AttentionState
        from vllm.distributed import get_ep_group
        self._guard_rank=get_ep_group().rank_in_group
        self._guard=[]
        if self.vllm_config.parallel_config.data_parallel_rank!=active_owner:
            root=self.model_runner._live_state_root
            tensors=[]
            for leaf in root.target.values():
                for tensor in leaf.numerical_tensors():
                    tensors.append(tensor[:64] if isinstance(leaf,AttentionState) else tensor)
            tensors.extend((root.continuation.selection.tensor,root.conv_selection.tensor,
                            root.remaining_outputs.tensor,root.continuation.resident_epoch.tensor))
            self._guard=[(t,t.cpu()) for t in tensors]
        return dict(rank=self._guard_rank,tensors=len(self._guard),
                    bytes=sum(v.numel()*v.element_size() for _,v in self._guard))

    def idle_guard_check(self):
        import torch
        for tensor,before in self._guard:
            after=tensor.cpu()
            if not torch.equal(before.contiguous().view(torch.uint8),after.contiguous().view(torch.uint8)):
                raise RuntimeError(f"Idle rank {self._guard_rank} State changed")
        count=len(self._guard);self._guard=[]
        return dict(rank=self._guard_rank,checked=count,exact=True)

