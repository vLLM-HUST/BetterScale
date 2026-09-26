"""Paged-wave host metadata; actual FIA arithmetic has a separate NPU gate."""

from types import SimpleNamespace

import pytest
import torch

from betterscale.live.arch.ascend.attention import PagedAttentionWave


def wave_fixture():
    wave = object.__new__(PagedAttentionWave)
    wave.closed = False
    wave.tokens, wave.requests, wave.context_tokens = 6, 2, 262144
    wave.mask = object()
    wave.planner = SimpleNamespace(fixtures=(None, SimpleNamespace(shape=(4096,))))
    observed = []
    wave.frame = SimpleNamespace(
        columns=2048,
        prepare=lambda planner, metadata, table, n: observed.append(
            (metadata, table, n)
        ),
    )
    return wave, observed


def test_paged_metadata_carries_pages_not_a_token_sized_gather():
    wave, observed = wave_fixture()
    rows = [list(reversed(range(2048))), [3001, 2000]]
    wave.prepare([3, 3], [262144, 129], rows)
    metadata, table, n = observed[0]
    assert n == 2 and table.shape == (2, 2048) and table.dtype == torch.int32
    assert metadata.actual_seq_lengths_q == [3, 6]
    assert metadata.seq_lens_list == [262144, 129]
    assert table[0].tolist() == rows[0]
    assert table[1, :2].tolist() == rows[1]
    assert table[1, 2:].count_nonzero() == 0


@pytest.mark.parametrize(
    "queries,lengths,rows",
    [
        ([3], [5], [[0]]),
        ([1, 3], [5, 5], [[0], [1]]),
        ([3, 3], [2, 5], [[0], [1]]),
        ([3, 3], [129, 5], [[0], [1]]),
        ([3, 3], [129, 5], [[0, 0], [1]]),
        ([3, 3], [129, 5], [[0, 4096], [1]]),
        ([3, 3], [262145, 5], [list(range(2049)), [1]]),
    ],
)
def test_invalid_page_metadata_never_reaches_native_planning(queries, lengths, rows):
    wave, observed = wave_fixture()
    with pytest.raises(ValueError):
        wave.prepare(queries, lengths, rows)
    assert not observed
