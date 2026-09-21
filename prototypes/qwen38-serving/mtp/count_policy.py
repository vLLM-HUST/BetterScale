"""Bounded count sweep. Capacity keys never encode request partitions."""
import os

MTP_TOKENS = int(os.environ.get('MTP_TOKENS', '2'))
assert 0 <= MTP_TOKENS <= 4
WIDTH = MTP_TOKENS + 1
# Disjoint from prefill capacity keys. Padding is reported, not hidden in rates.
BINS = (3, 6, 12, 24, 40)
SPEC_CAPACITIES = tuple(n for n in BINS if n < 8*WIDTH) + (next(n for n in BINS if n >= 8*WIDTH),)


def capture_lengths(lengths, width=WIDTH):
    """Only GDN capture input: unused capacity stays padding, not extra queries."""
    assert 2 <= width <= 5 and 0 < len(lengths) <= 8
    assert all(n > 0 for n in lengths)
    return tuple(min(n,width) for n in lengths)
