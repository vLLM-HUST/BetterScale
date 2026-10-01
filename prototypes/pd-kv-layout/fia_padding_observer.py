"""Opt-in, one-shot proof of the native/owned target padding translation."""
import json
from pathlib import Path

def install(directory):
    from vllm.distributed import get_ep_group
    from betterscale.patches.qwen_fia.context_parallel import adapter, plan
    rank=get_ep_group().rank_in_group
    original=adapter.target_metadata
    recorded=False
    def observe(metadata, live, tokens):
        nonlocal recorded
        result=original(metadata,live,tokens)
        if not recorded and len(metadata.seq_lens_list)>live and any(metadata.seq_lens_list[live:]):
            recorded=True
            ends=list(metadata.actual_seq_lengths_q)
            queries=[b-a for a,b in zip([0]+ends,ends)]
            try:
                plan.schedule(list(metadata.seq_lens_list),queries)
                old_failure=None
            except ValueError as error:
                old_failure=str(error)
            report=dict(rank=rank,live_requests=live,graph_tokens=tokens,
                        num_actual_tokens=metadata.num_actual_tokens,
                        query_ends=ends,kv_lengths=list(metadata.seq_lens_list),
                        device_length_shape=list(metadata._mtp_device_seq_lens.shape),
                        corrected_query_ends=result.actual_seq_lengths_q,
                        corrected_kv_lengths=result.seq_lens_list,
                        unchanged_device_lengths=result._mtp_device_seq_lens is metadata._mtp_device_seq_lens,
                        old_planner_failure=old_failure)
            root=Path(directory);root.mkdir(parents=True,exist_ok=True)
            (root/f"rank{rank}.json").write_text(json.dumps(report,indent=2))
        return result
    adapter.target_metadata=observe
