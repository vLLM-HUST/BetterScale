"""CPU oracle for target GDN checkpoint selection and fixed TP resharding.

Mirrors owned State selection (one-based candidate and conv selector).
Not a device-export hook; caller must provide quiescent CPU snapshots.
No MTP draft KV/proposals or historical checkpoint reconstruction.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Geometry:
    key_heads: int
    value_heads: int
    key_dim: int
    value_dim: int
    history: int = 3

    def __post_init__(self):
        if min(self.key_heads, self.value_heads, self.key_dim, self.value_dim, self.history) <= 0:
            raise ValueError("invalid geometry")

    @property
    def channels(self):
        return 2 * self.key_heads * self.key_dim + self.value_heads * self.value_dim


@dataclass
class Snapshot:
    cursor: int
    geometry: Geometry
    recurrent: np.ndarray  # [layer, local value head, key dim, value dim]
    conv: np.ndarray  # [layer, history, Q channels + K channels + V channels]


def capture(layers, seat, selected, conv_selected, cursor, geometry, *, writer_retired):
    if not writer_retired:
        raise ValueError("final device writer not retired")
    if not layers or seat < 0 or cursor < 0 or selected not in (1, 2, 3) or conv_selected not in (1, 2, 3):
        raise ValueError("invalid checkpoint descriptor")
    recurrent, conv = [], []
    for window, candidates in layers:
        if (window.dtype != np.uint16 or candidates.dtype != np.float32
                or window.ndim != 3 or candidates.ndim != 4
                or window.shape[1:] != (geometry.history + 2, geometry.channels)
                or candidates.shape != (window.shape[0] * 3, geometry.value_heads,
                                        geometry.key_dim, geometry.value_dim)
                or seat >= window.shape[0]):
            raise ValueError("unqualified State geometry")
        recurrent.append(candidates[seat * 3 + selected - 1].copy())
        conv.append(window[seat, conv_selected - 1:conv_selected - 1 + geometry.history].copy())
    return Snapshot(cursor, geometry, np.stack(recurrent), np.stack(conv))


def validate(snapshot):
    g = snapshot.geometry
    if (snapshot.cursor < 0 or snapshot.recurrent.dtype != np.float32
            or snapshot.conv.dtype != np.uint16 or snapshot.recurrent.ndim != 4
            or snapshot.conv.ndim != 3
            or snapshot.recurrent.shape[1:] != (g.value_heads, g.key_dim, g.value_dim)
            or snapshot.conv.shape != (len(snapshot.recurrent), g.history, g.channels)):
        raise ValueError("invalid canonical target snapshot")


def combine(shards):
    if not shards:
        raise ValueError("no shards")
    for shard in shards:
        validate(shard)
    first = shards[0]
    if any(s.cursor != first.cursor or s.geometry != first.geometry
           or len(s.recurrent) != len(first.recurrent) for s in shards):
        raise ValueError("mixed frontier/geometry")
    g = first.geometry
    q_end = g.key_heads * g.key_dim
    v_start = 2 * q_end
    # Each TP shard stores Q,K,V; concatenate corresponding sections, NOT
    # [Q0,K0,V0,Q1,K1,V1], which would corrupt TP1 convolution history.
    sections = [(0, q_end), (q_end, v_start), (v_start, g.channels)]
    conv = np.concatenate([np.concatenate([s.conv[..., a:b] for s in shards], axis=-1)
                           for a, b in sections], axis=-1)
    return Snapshot(first.cursor,
                    Geometry(g.key_heads * len(shards), g.value_heads * len(shards),
                             g.key_dim, g.value_dim, g.history),
                    np.concatenate([s.recurrent for s in shards], axis=1), conv)


def split(snapshot, ranks):
    validate(snapshot)
    g = snapshot.geometry
    if ranks <= 0 or g.key_heads % ranks or g.value_heads % ranks:
        raise ValueError("heads must divide TP")
    local = Geometry(g.key_heads // ranks, g.value_heads // ranks,
                     g.key_dim, g.value_dim, g.history)
    q_end = g.key_heads * g.key_dim
    sections = [(0, q_end), (q_end, 2 * q_end), (2 * q_end, g.channels)]
    result = []
    for rank in range(ranks):
        conv = np.concatenate([snapshot.conv[..., a + (b-a)//ranks*rank:
                                                 a + (b-a)//ranks*(rank+1)]
                               for a, b in sections], axis=-1)
        recurrent = snapshot.recurrent[:, rank*local.value_heads:(rank+1)*local.value_heads].copy()
        result.append(Snapshot(snapshot.cursor, local, recurrent, conv))
    return result


def install(snapshot, layers, seat, *, destination_quiescent):
    """Normalize to candidate0/history0; caller must publish selectors=1.

    Stage/validate all leaves first. No live native root is mutated by this probe.
    """
    validate(snapshot)
    if not destination_quiescent:
        raise ValueError("destination still executing")
    g = snapshot.geometry
    if len(layers) != len(snapshot.recurrent):
        raise ValueError("layer census mismatch")
    for window, candidates in layers:
        if (seat < 0 or seat >= len(window) or window.dtype != np.uint16
                or candidates.dtype != np.float32
                or window.shape[1:] != (g.history + 2, g.channels)
                or candidates.shape != (len(window)*3, g.value_heads, g.key_dim, g.value_dim)):
            raise ValueError("destination geometry mismatch")
    for index, (window, candidates) in enumerate(layers):
        # Speculative slots aren't valid restored target State.
        candidates[seat*3:seat*3+3] = 0
        candidates[seat*3] = snapshot.recurrent[index]
        window[seat] = 0
        window[seat, :g.history] = snapshot.conv[index]
    return {"selected": 1, "conv_selected": 1, "target_cursor": snapshot.cursor,
            "draft_valid": False}
