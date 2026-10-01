"""Diagnostic only: retain MTP buffers/compute, but never schedule its proposals.

This is not a production P-only path: draft computation still runs. It isolates
speculative verification/accepted-candidate transitions from target numerics.
"""
def install(scheduler_class):
    original=scheduler_class._update_after_schedule
    def update(self,output):
        original(self,output)
        for rid in output.num_scheduled_tokens:
            # Native accounting above used the actually scheduled proposal count.
            # Suppress only the next grant's placeholders; keep lookahead capacity.
            self.requests[rid].spec_token_ids=[]
    scheduler_class._update_after_schedule=update
