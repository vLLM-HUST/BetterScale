import numpy as np
import pytest

from gdn_checkpoint import Geometry, capture, combine, split, install

G = Geometry(2, 4, 8, 8)


def layers(seed, geometry=G):
    rng = np.random.default_rng(seed)
    return [(rng.integers(0, 65535, (3, 5, geometry.channels), dtype=np.uint16),
             rng.standard_normal((9, geometry.value_heads, 8, 8)).astype(np.float32))
            for _ in range(3)]


@pytest.mark.parametrize("selected", [1, 2, 3])
@pytest.mark.parametrize("conv_selected", [1, 2, 3])
def test_selected_checkpoint_roundtrip(selected, conv_selected):
    source = layers(17)
    snap = capture(source, 1, selected, conv_selected, 129, G, writer_retired=True)
    target = layers(27)
    before = [(a.copy(), b.copy()) for a, b in target]
    result = install(snap, target, 0, destination_quiescent=True)
    assert result == dict(selected=1, conv_selected=1, target_cursor=129, draft_valid=False)
    for index, ((window, matrix), (old_window, old_matrix)) in enumerate(zip(target, before)):
        np.testing.assert_array_equal(matrix[0], source[index][1][3+selected-1])
        np.testing.assert_array_equal(window[0, :3], source[index][0][1, conv_selected-1:conv_selected+2])
        np.testing.assert_array_equal(window[1:], old_window[1:])
        np.testing.assert_array_equal(matrix[3:], old_matrix[3:])
        assert not matrix[1:3].any()
        assert not window[0, 3:].any()


def test_tp2_tp1_roundtrip_preserves_qkv_sections():
    shards = [capture(layers(seed), 1, 3, 2, 257, G, writer_retired=True) for seed in (17, 29)]
    merged = combine(shards)
    q = G.key_heads * G.key_dim
    np.testing.assert_array_equal(merged.conv[..., :2*q],
                                  np.concatenate([s.conv[..., :q] for s in shards], axis=-1))
    recovered = split(merged, 2)
    for a, b in zip(shards, recovered):
        assert a.geometry == b.geometry and a.cursor == b.cursor
        np.testing.assert_array_equal(a.conv, b.conv)
        np.testing.assert_array_equal(a.recurrent, b.recurrent)


def test_incompatible_frontier_rejected():
    shards = [capture(layers(17), 1, 1, 1, cursor, G, writer_retired=True) for cursor in (128, 129)]
    with pytest.raises(ValueError):
        combine(shards)


def test_writer_and_destination_fences():
    with pytest.raises(ValueError):
        capture(layers(17), 1, 1, 1, 129, G, writer_retired=False)
    snap = capture(layers(17), 1, 1, 1, 129, G, writer_retired=True)
    with pytest.raises(ValueError):
        install(snap, layers(19), 1, destination_quiescent=False)


def test_invalid_late_leaf_does_not_partially_restore():
    snap = capture(layers(17), 1, 1, 1, 129, G, writer_retired=True)
    target = layers(23)
    before = target[0][0].copy()
    target[-1] = (target[-1][0][..., :-1], target[-1][1])
    with pytest.raises(ValueError):
        install(snap, target, 1, destination_quiescent=True)
    np.testing.assert_array_equal(before, target[0][0])
