"""Whole-layer ownership for the persistent routed-expert service."""
from dataclasses import dataclass
from .model_geometry import GEOMETRY as G


@dataclass(frozen=True)
class Placement:
    mode: str
    owners: int

    def __post_init__(self):
        if self.mode != 'layer' or self.owners not in (1, 2, 4):
            raise ValueError('persistent service supports layer placement and E1/E2/E4')

    def layers(self, owner):
        if not 0 <= owner < self.owners:
            raise ValueError('invalid owner')
        return tuple(layer for layer in G.layers
                     if layer % self.owners == owner)

    def experts(self, owner):
        self.layers(owner)
        return (0, G.experts)

    def targets(self, layer):
        if layer not in G.layers:
            raise ValueError('layer is outside the configured routed layer set')
        return (layer % self.owners,)
