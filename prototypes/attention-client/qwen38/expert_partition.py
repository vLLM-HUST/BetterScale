"""Contiguous ownership with padded NZ catalogs for512 experts on E3/E4/E8.

The final E3 owner has170 live experts in171 storage slots. Padding never owns
an expert ID and must always receive zero routed rows. Binary route division
must use the same slot count; this helper alone does not qualify the E3 ABI.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ExpertPartition:
    owners: int
    experts: int = 512

    def __post_init__(self):
        if self.owners not in (3, 4, 8) or self.experts != 512:
            raise ValueError("Qwen38 catalog supports512 experts on E3/E4/E8")

    @property
    def slots(self):
        return (self.experts + self.owners - 1) // self.owners

    def bounds(self, owner):
        if not 0 <= owner < self.owners:
            raise ValueError("expert owner outside topology")
        start = owner * self.slots
        return start, min(start + self.slots, self.experts)

    def locate(self, expert):
        if not 0 <= expert < self.experts:
            raise ValueError("routed expert ID outside checkpoint")
        return divmod(expert, self.slots)
