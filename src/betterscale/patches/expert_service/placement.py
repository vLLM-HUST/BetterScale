"""Whole-layer ownership for the persistent routed-expert service."""
from dataclasses import dataclass
from .model_geometry import GEOMETRY as G


@dataclass(frozen=True)
class Placement:
    mode: str
    owners: int

    def __post_init__(self):
        if self.mode not in ('layer', 'expert') or self.owners not in (1, 2, 4):
            raise ValueError('persistent service supports layer/expert placement and E1/E2/E4')

    def layers(self, owner):
        if not 0 <= owner < self.owners:
            raise ValueError('invalid owner')
        return tuple(layer for layer in G.layers
                     if self.mode == 'expert' or layer % self.owners == owner)

    def experts(self, owner):
        self.layers(owner)
        return ((owner*G.experts//self.owners, (owner+1)*G.experts//self.owners)
                if self.mode == 'expert' else (0, G.experts))

    def targets(self, layer):
        if layer not in G.layers:
            raise ValueError('layer is outside the configured routed layer set')
        return tuple(range(self.owners)) if self.mode == 'expert' else (layer % self.owners,)
