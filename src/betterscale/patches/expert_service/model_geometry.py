"""Qwen35 geometry; one physical MTP layer may run multiple draft steps."""
import os
from dataclasses import dataclass
@dataclass(frozen=True)
class Geometry:
    name: str = 'qwen35'
    hidden: int = 2048
    inner: int = 512
    experts: int = 256
    topk: int = 8
    total_layers: int = 40
    @property
    def layers(self):return tuple(range(self.total_layers))
_draft = int(os.environ.get('BETTERSCALE_EXPERT_DRAFT_LAYERS', '0'))
if _draft not in (0,1):raise ValueError('Qwen35 has zero or one physical draft layer')
GEOMETRY = Geometry(total_layers=40+_draft)
