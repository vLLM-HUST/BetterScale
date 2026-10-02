"""Bounded count sweep. Capacity keys never encode request partitions."""

from .execution_capacity import EXECUTION

import os

MTP_TOKENS = int(os.environ.get("MTP_TOKENS", "2"))
assert 0 <= MTP_TOKENS <= 4
WIDTH = MTP_TOKENS + 1
# Disjoint from prefill capacity keys. Padding is reported, not hidden in rates.
BINS = ((3, 6, 12, 24, 40, 48, 80) if EXECUTION == 16 else
        (3, 6, 12, 24, 40, 48, 72, 80, 96, 144, 192, 240, 320, 400))
SPEC_CAPACITIES = tuple(n for n in BINS if n < EXECUTION * WIDTH) + (
    next(n for n in BINS if n >= EXECUTION * WIDTH),
)


def capture_lengths(lengths, width=WIDTH):
    """Only GDN capture input: unused capacity stays padding, not extra queries."""
    assert 2 <= width <= 5 and 0 < len(lengths) <= EXECUTION
    assert all(n > 0 for n in lengths)
    return tuple(min(n, width) for n in lengths)
