"""Match native draft task-update resources to the actual ping-pong graph key.

The donor keys these mutable lists only by token capacity. Our shared dispatcher
adds a bank to draft graph descriptors too: pooling both banks' handles would
update the other, potentially still executing graph. Keep graph and update
ownership identical, for both native decode and prefill parameter planes.
"""
from contextlib import contextmanager
from functools import wraps


@contextmanager
def graph_resources(acl, proposer, bank):
    assert bank in (0, 1)
    names = ('_draft_graph_params', '_draft_graph_prefill_params')
    saved = {name: getattr(acl, name) for name in names}
    if not hasattr(proposer, '_mtp_graph_banks'):
        proposer._mtp_graph_banks = {}
    try:
        for name, original in saved.items():
            if original is None:
                continue
            key = (name, bank)
            if key not in proposer._mtp_graph_banks:
                keys = original.handles.keys()
                proposer._mtp_graph_banks[key] = acl.GraphParams(
                    events={k: [] for k in keys}, workspaces={k: None for k in keys},
                    handles={k: [] for k in keys}, attn_params={k: [] for k in keys})
            setattr(acl, name, proposer._mtp_graph_banks[key])
        yield
    finally:
        for name, original in saved.items():
            setattr(acl, name, original)


def install():
    import vllm_ascend.compilation.acl_graph as acl
    from vllm_ascend.spec_decode.llm_base_proposer import AscendSpecDecodeBaseProposer as Proposer
    if getattr(Proposer, '_betterscale_mtp_banks', False):
        return

    def wrap(original):
        @wraps(original)
        def call(self, *args, **kwargs):
            if not self.use_cuda_graph:
                return original(self, *args, **kwargs)
            # The same runner dispatcher supplies target AND draft descriptors.
            bank = self.runner._owned_bank
            with graph_resources(acl, self, bank):
                return original(self, *args, **kwargs)
        return call

    Proposer.dummy_run = wrap(Proposer.dummy_run)
    Proposer._propose = wrap(Proposer._propose)
    Proposer._betterscale_mtp_banks = True
