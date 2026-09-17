"""CPU contract checks requiring the selected model-runtime Python overlay."""

from types import SimpleNamespace
import torch
from trace_commit import RetainedCommit
from trace_ple import BulkCodec
from livemodule.serve.qwen38.ple_mailbox import (
    Qwen38PLEMailboxCodec,
    Qwen38PLEMailboxRow,
)


class Model:
    num_speculative_tokens = 1
    contract = SimpleNamespace(hc_count=1, hidden_size=4)

    def compute_top_tokens(self, hidden):
        return hidden[..., 0].to(torch.int64)

    def publish_speculative_acceptance(self, accepted, active):
        self.accepted = torch.where(active, accepted + 1, 0)


model = Model()
commit = RetainedCommit(model)
# Both proposals accepted; request has room for only the FIRST committed token.
result = commit.forward_bounded(
    torch.tensor([[5], [7]]),
    torch.tensor([[[5.0], [6.0]], [[7.0], [8.0]]]),
    torch.arange(16).reshape(2, 2, 4).float(),
    torch.tensor([True, True]),
    torch.tensor([1, 2]),
    torch.tensor([-1, -1]),
)
assert result[3].tolist() == [1, 2]
assert result[6].tolist() == [5, 8]
assert result[7].tolist() == [[0.0, 1.0, 2.0, 3.0], [12.0, 13.0, 14.0, 15.0]]
assert model.accepted.tolist() == [1, 2]
codec = BulkCodec(lanes=32, local_embed_dim=16)
rows = tuple(Qwen38PLEMailboxRow(i, i // 4, 2, 3, 4, (1, 2, 3)) for i in [0, 2, 7, 31])
x = torch.randn(4, 16, dtype=torch.bfloat16)
assert codec.encode_responses(rows, x) == Qwen38PLEMailboxCodec.encode_responses(
    codec, rows, x
)
assert codec.encode_responses((), torch.empty(0, 16, dtype=torch.bfloat16)) == bytes(
    codec.response_bytes
)
print(
    "PASS: bounded retained endpoint, sparse byte-exact PLE, empty wave acknowledgment"
)
