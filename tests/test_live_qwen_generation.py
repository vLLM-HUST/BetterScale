"""Host protocol tests with controlled proposals, not model-quality evidence."""

from types import SimpleNamespace
from unittest.mock import patch
import torch
import pytest
from betterscale.live.llm.qwen35 import Capacity
from betterscale.live.llm.qwen35.generation import generate
from betterscale.live.llm.qwen35.residents import ResidentTable


class ProtocolRoot:
    live_device = "cpu"

    def __init__(self, bad_draft=False):
        self.calls = []
        self.bad_draft = bad_draft
        self.continuation = SimpleNamespace(
            **{
                name: SimpleNamespace(tensor=torch.zeros(shape, dtype=dtype))
                for name, shape, dtype in [
                    ("anchor_hidden", (2, 2), torch.float32),
                    ("anchor_token", (2,), torch.long),
                    ("selection", (2,), torch.long),
                    ("target_cursor", (2,), torch.long),
                    ("draft_cursor", (2,), torch.long),
                    ("proposal", (2, 2), torch.long),
                ]
            }
        )
        self.residents_table = ResidentTable(
            Capacity(1, 2, page_tokens=4, token_pages=8), clear_seat=self.clear_seat
        )

    def clear_seat(self, seat):
        self.continuation.selection.tensor[seat] = 1

    def target_step(self, tokens, **kwargs):
        self.calls.append(("target", list(tokens), kwargs))
        return self.result(tokens, draft=False)

    def draft_step(self, tokens, hidden_seed, **kwargs):
        self.calls.append(("draft", list(tokens), hidden_seed.clone(), kwargs))
        return self.result(tokens, draft=True)

    def result(self, tokens, draft):
        hidden = torch.tensor(
            [[t + (100 if draft else 0)] * 2 for t in tokens], dtype=torch.float32
        )
        logits = torch.full((len(tokens), 64), -10.0)
        for i, token in enumerate(tokens):
            logits[i, token + (7 if draft and self.bad_draft else 1)] = 10
        return hidden, logits


@pytest.mark.parametrize("bad_draft", [False, True])
def test_verification_matches_greedy_and_retains_hot_prefix(bad_draft):
    root = ProtocolRoot(bad_draft)
    with patch.object(
        torch, "npu", SimpleNamespace(synchronize=lambda _: None), create=True
    ):
        a = generate(root, [1, 2, 3], 6)
        assert a["token_ids"] == [4, 5, 6, 7, 8, 9]
        assert a["accepted_drafts"] == (0 if bad_draft else 4)
        b = generate(root, [20, 21], 1)
        assert b["seat"] == 1
        before = len(root.calls)
        c = generate(root, list(range(1, 10)), 2)
        assert c["cached_tokens"] == 9 and c["seat"] == 0
        assert c["token_ids"] == [10, 11]
        # Hot hit starts at retained hidden boundary; no target prefix reforward.
        assert root.calls[before][0] == "draft"
        assert root.calls[before][3]["position"] == 8


def test_recursive_draft_rows_are_replaced_by_verified_target_hidden():
    root = ProtocolRoot()
    with patch.object(
        torch, "npu", SimpleNamespace(synchronize=lambda _: None), create=True
    ):
        result = generate(root, [1, 2, 3], 3)
    assert result["token_ids"] == [4, 5, 6]
    name, tokens, hidden, metadata = root.calls[-1]
    assert (name, tokens, metadata["position"]) == ("draft", [5, 6], 3)
    torch.testing.assert_close(hidden, torch.tensor([[4.0, 4.0], [5.0, 5.0]]))
    assert root.continuation.selection.tensor[0] == 3
    assert root.continuation.target_cursor.tensor[0] == 6
    assert root.continuation.draft_cursor.tensor[0] == 5


def test_eos_truncates_commit_and_next_candidate_selection():
    root = ProtocolRoot()
    with patch.object(
        torch, "npu", SimpleNamespace(synchronize=lambda _: None), create=True
    ):
        result = generate(root, [1, 2, 3], 6, eos_token_ids=(5,))
    assert result["token_ids"] == [4, 5]
    assert root.residents_table.seats[0].tokens == [1, 2, 3, 4, 5]
    assert root.continuation.selection.tensor[0] == 2
