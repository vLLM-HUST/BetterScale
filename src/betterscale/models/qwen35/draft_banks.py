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
    names = ("_draft_graph_params", "_draft_graph_prefill_params")
    saved = {name: getattr(acl, name) for name in names}
    if not hasattr(proposer, "_mtp_graph_banks"):
        proposer._mtp_graph_banks = {}
    try:
        for name, original in saved.items():
            if original is None:
                continue
            key = (name, bank)
            if key not in proposer._mtp_graph_banks:
                keys = original.handles.keys()
                proposer._mtp_graph_banks[key] = acl.GraphParams(
                    events={k: [] for k in keys},
                    workspaces={k: None for k in keys},
                    handles={k: [] for k in keys},
                    attn_params={k: [] for k in keys},
                )
            setattr(acl, name, proposer._mtp_graph_banks[key])
        yield
    finally:
        for name, original in saved.items():
            setattr(acl, name, original)



@contextmanager
def target_capacity(dispatcher, tokens):
    """Draft step0 consumes the target rows; keep its graph envelope.

    Re-dispatching only the unpadded count can turn a17-token warm prefill
    (target mixed32) into verification24, whose captured attention admits Q1..3.
    This is a capacity floor, not a change to live queries or numerical State.
    The proposer is serial/non-reentrant, as required by graph_resources.
    """
    if type(tokens) is not int or tokens<1:
        raise ValueError("Invalid target graph capacity")
    original=dispatcher.dispatch
    def dispatch(num_tokens,*args,**kwargs):
        return original(max(num_tokens,tokens),*args,**kwargs)
    dispatcher.dispatch=dispatch
    try:
        yield
    finally:
        dispatcher.dispatch=original


def install():
    import vllm_ascend.compilation.acl_graph as acl
    from vllm_ascend.spec_decode.llm_base_proposer import (
        AscendSpecDecodeBaseProposer as Proposer,
    )

    if getattr(Proposer, "_betterscale_mtp_banks", False):
        return

    def wrap(original, *, proposing=False):
        @wraps(original)
        def call(self, *args, **kwargs):
            if not self.use_cuda_graph:
                return original(self, *args, **kwargs)
            # The same runner dispatcher supplies target AND draft descriptors.
            bank = self.runner._owned_bank
            with graph_resources(acl, self, bank):
                if not proposing:
                    return original(self, *args, **kwargs)
                # Pinned donor _propose argument7 is the actual target
                # descriptor, already classified and padded across DP ranks.
                target=kwargs.get("target_model_batch_desc", args[6] if len(args)>6 else None)
                if target is None:
                    raise ValueError("Missing target graph descriptor for MTP")
                with target_capacity(self.runner.cudagraph_dispatcher,target.num_tokens):
                    return original(self, *args, **kwargs)

        return call

    Proposer.dummy_run = wrap(Proposer.dummy_run)
    Proposer._propose = wrap(Proposer._propose, proposing=True)
    from .draft_output import install as install_live_draft_rows

    install_live_draft_rows()
    from .draft_sampling import install as install_request_sampling

    install_request_sampling()
    Proposer._betterscale_mtp_banks = True
