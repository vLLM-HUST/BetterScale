"""CPU execution of the actual captured prefill body with a numerical model double.

Run with owned-wave, joint-wave and the retained LiveInference on PYTHONPATH.
Checks addresses, cursor and final-token selection, not neural equivalence.
"""

from contextlib import nullcontext
from types import SimpleNamespace, MethodType
from unittest.mock import patch
import torch
from vllm_ascend.worker.worker import NPUWorker  # native import order
from root import ServingRoot


def check_prefill():
    for valid, capacity, hit in [
        (70, 128, 0),
        (13, 16, 1000),
        (41, 64, 980),
        (30, 32, 994),
        (1, 1, 1023),
    ]:
        m = SimpleNamespace(
            block_tables=torch.zeros((1, 8), dtype=torch.int32),
            slot_mapping=torch.zeros(capacity, dtype=torch.int32),
        )
        cache = torch.full((1152,), -99, dtype=torch.int64)
        table = torch.arange(1, 9, dtype=torch.int32)
        frame = dict(
            count=capacity,
            bank=0,
            command=torch.tensor([0, 0, 1, 1, hit, hit + valid, 2]),
            block_row=table,
            device_q_lengths=torch.tensor([valid]),
            device_kv_lengths=torch.ones(1, dtype=torch.int64),
            ids=torch.arange(capacity) + 10,
        )

        class Model:
            def __call__(self, **kw):
                positions = kw["positions"]
                ids = kw["input_ids"]
                assert torch.equal(positions[:valid], torch.arange(hit, hit + valid))
                assert torch.all(positions[valid:] == 0)
                live = m.slot_mapping >= 0
                cache[m.slot_mapping[live].long()] = ids[live]
                return ids[:, None]

            def compute_logits(self, hidden):
                assert hidden.shape == (1, 1)
                assert hidden.item() == valid - 1 + 10
                return hidden

        owner = SimpleNamespace(
            frames={"p": frame},
            bundle=SimpleNamespace(
                device="cpu",
                block_size=128,
                model=Model(),
                sampling=None,
                sampler=lambda **kw: SimpleNamespace(sampled_token_ids=kw["logits"]),
            ),
            progress=torch.zeros((1, 7), dtype=torch.int64),
            tables=torch.zeros((1, 8), dtype=torch.int32),
            egress=torch.zeros((2, 1, 8), dtype=torch.int64),
            forward_calls=0,
            registry=lambda key: nullcontext(),
            context=lambda *a, **kw: nullcontext(),
            static_attention=SimpleNamespace(scope=lambda key: nullcontext()),
        )
        owner.actions = {"p": None}
        owner.run_model = MethodType(ServingRoot.run_model, owner)
        with patch("root.construct_meta_tensors", return_value=m):
            out = ServingRoot.prefill(owner, "p")
        assert owner.progress[0].tolist() == [
            hit + valid,
            valid - 1 + 10,
            1,
            2,
            1,
            hit + valid,
            1,
        ]
        assert out[0].tolist() == [0, 0, 1, hit + valid, valid - 1 + 10, 1, 0, 0]
        assert frame["device_kv_lengths"].item() == hit + valid
        assert torch.all(m.slot_mapping[valid:] == -1)
        expected = torch.full_like(cache, -99)
        expected[128 + hit : 128 + hit + valid] = frame["ids"][:valid]
        assert torch.equal(cache, expected)
    print(
        "PASS: actual prefill body, ceiling padding, last-valid sampler, KV sentinels and max-context bounds"
    )


if __name__ == "__main__":
    check_prefill()
