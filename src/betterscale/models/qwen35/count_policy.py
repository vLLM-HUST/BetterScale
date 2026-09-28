"""Bounded count sweep. Capacity keys never encode request partitions."""

import os

MTP_TOKENS = int(os.environ.get("MTP_TOKENS", "2"))
assert 0 <= MTP_TOKENS <= 4
WIDTH = MTP_TOKENS + 1
# Disjoint from prefill capacity keys. Padding is reported, not hidden in rates.
BINS = (3, 6, 12, 24, 40, 48, 80, 96, 108, 144, 180)


def spec_capacities(requests=16, width=WIDTH):
    if type(requests) is not int or not 1 <= requests <= 36:
        raise ValueError("Qwen35 execution seats must be in 1..36")
    return tuple(n for n in BINS if n < requests * width) + (
        next(n for n in BINS if n >= requests * width),
    )


SPEC_CAPACITIES = spec_capacities()


def capture_lengths(lengths, width=WIDTH, requests=16):
    """Only GDN capture input: unused capacity stays padding, not extra queries."""
    assert 2 <= width <= 5 and 0 < len(lengths) <= requests
    assert all(n > 0 for n in lengths)
    return tuple(min(n, width) for n in lengths)
