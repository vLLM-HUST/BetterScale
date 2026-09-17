"""Static TP2 State payload estimate from the owned Qwen38 declarations.

No weights or State storage are allocated. Includes existing MTP history lanes
and fixed-seat GDN scratch, even for target-only serving. Excludes allocator
alignment and graph/parameter/runtime overhead; not a hardware capacity gate.
"""

import json
import torch
from pathlib import Path
import argparse

p = argparse.ArgumentParser()
p.add_argument(
    "--model",
    type=Path,
    default=Path("/data/shared_models/Qwen3.8-Flash-Next-w8a8-mtp"),
)
a = p.parse_args()
from livemodule.core.state_tensor import (
    StateDomain,
    ExactStateCapacity,
    ElasticStateCapacity,
)
from livemodule.llm.qwen38.contract import Qwen38TextContract
from livemodule.llm.qwen38.parallel import Qwen38ParallelPlan
from livemodule.llm.qwen38.state import Qwen38RequestStateBank

c = Qwen38TextContract.from_model_config(
    json.loads((a.model / "config.json").read_text())
)
p = Qwen38ParallelPlan(c, 0, 2)
h = StateDomain(ElasticStateCapacity())
r = StateDomain(ExactStateCapacity(1))
from livemodule import LiveRuntime, live_runtime

context = live_runtime(LiveRuntime(device="meta"))
context.__enter__()
bank = Qwen38RequestStateBank(
    contract=c,
    plan=p,
    block_size=64,
    history_domain=h,
    request_domain=r,
    model_dtype=torch.bfloat16,
    max_requests=1,
    num_speculative_tokens=0,
)
states = list(bank.named_states())
history = [(n, s) for n, s in states if s.domain is h]
fixed = [(n, s) for n, s in states if s.domain is r]
bpt = sum(s.logical_block_bytes for _, s in history) // 64
conv = (
    (
        2 * p.gdn_key_heads.count * c.linear_key_head_dim
        + p.gdn_value_heads.count * c.linear_value_head_dim
    )
    * (c.linear_conv_kernel_dim - 1)
    * 2
)
recurrent = (
    p.gdn_value_heads.count * c.linear_key_head_dim * c.linear_value_head_dim * 4
)
gdn = (conv + recurrent) * c.layer_types.count("linear_attention") * 2
per_seat = (
    gdn
    + sum(s.logical_block_bytes + s.fixed_bytes for _, s in fixed)
    + sum(s.fixed_bytes for _, s in history)
)
print(
    json.dumps(
        dict(
            history_lanes=len(history),
            history_bytes_per_token_per_rank=bpt,
            gdn_bytes_per_request_per_rank_including_scratch=gdn,
            all_fixed_bytes_per_request_per_rank=per_seat,
            history_owners=[n for n, _ in history if n.endswith("_key")],
            estimates=[
                dict(
                    state_gib=gib,
                    seats=n,
                    history_tokens_per_TP2_group=max(
                        0, (int(gib * 2**30) - n * per_seat) // (64 * bpt)
                    )
                    * 64,
                )
                for gib in [4, 8, 40, 48]
                for n in [1, 8, 16, 32, 64]
            ],
        ),
        indent=2,
    )
)

context.__exit__(None, None, None)
