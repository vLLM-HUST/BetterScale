"""CPU/NPU-independent row-map oracle for MC2's prefix-valid mask adapter."""

import torch
from colocated_ep import compact_prefix

torch.manual_seed(17)
for n in (1, 5, 32, 256):
    h = torch.randn(n, 16, dtype=torch.bfloat16)
    ids = torch.randint(512, (n, 10), dtype=torch.int32)
    probabilities = torch.rand(n, 10)
    for active in (
        torch.zeros(n, dtype=torch.bool),
        torch.ones(n, dtype=torch.bool),
        torch.arange(n) % 3 == 1,
        torch.rand(n) > 0.5,
    ):
        ph, pi, pp, pa, destination = compact_prefix(h, ids, probabilities, active)
        order = torch.cat((active.nonzero().flatten(), (~active).nonzero().flatten()))
        assert torch.equal(pa, active[order])
        for old, packed in ((h, ph), (ids, pi), (probabilities, pp)):
            assert torch.equal(packed, old[order])
            assert torch.equal(packed.index_select(0, destination), old)
print("PASS:16 masks, exact stable compaction and inverse; no host count in adapter")
