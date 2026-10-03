import numpy as np
import pytest

from probe import PAGE, DIM, SENTINEL, allocate, export_head, restore_head, logical

@pytest.mark.parametrize("layout", ["tp2", "tp1_native", "tp1_head_major"])
@pytest.mark.parametrize("tokens", [1, 127, 128, 129, 389])
def test_incremental_roundtrip_and_guard_pages(layout, tokens):
    rng = np.random.default_rng(91)
    pages = (tokens + PAGE - 1) // PAGE
    pmap = np.arange(pages)[::-1] + 1
    dmap = np.roll(np.arange(pages), 1) + 1
    truth = rng.integers(0, SENTINEL, (tokens, 2, DIM), dtype=np.uint16)
    source = np.full((2, pages + 2, PAGE, DIM), SENTINEL, dtype=np.uint16)
    for token in range(tokens):
        source[:, pmap[token // PAGE], token % PAGE] = truth[token]
    target = allocate(layout, pages + 2)
    boundaries = sorted(set([0, tokens] + [v for v in [1, 17, 127, 128, 129, 257] if v < tokens]))
    previous = 0
    for start, stop in zip(boundaries, boundaries[1:]):
        for head in (1, 0):  # completion order need not follow head order
            payload = export_head(source[head], pmap, start, stop)
            restore_head(payload, target, dmap, start, head, layout)
        np.testing.assert_array_equal(logical(target, dmap, stop, layout), truth[:stop])
        previous = stop
    assert previous == tokens
    # Unowned physical pages and unused tail stay untouched.
    canonical = target.transpose(1, 2, 0, 3) if layout == "tp2" else (
        target.transpose(0, 2, 1, 3) if layout == "tp1_head_major" else target)
    assert np.all(canonical[0] == SENTINEL)
    assert np.all(canonical[-1] == SENTINEL)
    if tokens % PAGE:
        assert np.all(canonical[dmap[-1], tokens % PAGE:] == SENTINEL)
    # Reverse D->wire preserves exactly the two original global heads.
    for head in (0, 1):
        dsource = canonical[:, :, head, :]
        np.testing.assert_array_equal(export_head(dsource, dmap, 0, tokens), truth[:, head])

@pytest.mark.parametrize("table", [[0, 0], [-1, 1], [0, 8], [0], [0.0, 1.0]])
def test_invalid_page_maps_rejected(table):
    with pytest.raises(ValueError):
        export_head(np.zeros((3, PAGE, DIM), dtype=np.uint16), table, 0, 129)

def test_wire_is_snapshot_not_mutable_source_alias():
    source = np.ones((1, PAGE, DIM), dtype=np.uint16)
    chunk = export_head(source, [0], 0, 17)
    source[:] = 2
    assert np.all(chunk == 1)

def test_wrong_wire_rejected():
    with pytest.raises(ValueError):
        restore_head(np.zeros((3, DIM), dtype=np.float32), allocate("tp2", 1), [0], 0, 0, "tp2")
