"""Startup observation only: retain native allocator/profile/capture behavior."""
import json
import torch
from betterscale.worker import Worker

class MemoryWorker(Worker):
    def snapshot(self, phase, **extra):
        torch.npu.synchronize()
        free, total = torch.npu.mem_get_info()
        record = dict(phase=phase, rank=self.rank, free=free, total=total,
                      allocated=torch.npu.memory_allocated(), reserved=torch.npu.memory_reserved(),
                      peak_allocated=torch.npu.max_memory_allocated(),
                      peak_reserved=torch.npu.max_memory_reserved())
        record.update(extra)
        print('CAPACITY_OBSERVATION '+json.dumps(record), flush=True)

    def determine_available_memory(self):
        self.snapshot('before_profile')
        result = super().determine_available_memory()
        self.snapshot('after_profile', kv_budget=result,
                      activation=getattr(self,'peak_activation_memory',None),
                      non_torch=getattr(self,'non_torch_memory',None))
        return result

    def compile_or_warm_up_model(self):
        self.snapshot('before_capture_after_kv')
        result = super().compile_or_warm_up_model()
        self.snapshot('after_capture_and_patch_warmup', graph=getattr(self,'npugraph_memory_bytes',None))
        return result
