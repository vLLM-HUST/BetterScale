"""Manual-headroom observation control for constructing the TP draft envelope."""

from memory_worker import MemoryWorker
from draft_warmup import prepare


class DraftProbeWorker(MemoryWorker):
    def compile_or_warm_up_model(self):
        result = super().compile_or_warm_up_model()
        prepare(self)
        return result
