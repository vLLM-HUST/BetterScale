"""CPU-only admission and process binding for the experimental expert route."""
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
from .route_plan import threshold as route_plan_threshold


@dataclass(frozen=True)
class ServiceConfig:
    control: str
    build: str
    owners: int
    sources: int
    source: int
    draft_layers: int
    qualification: str | None = None
    placement: str = 'layer'

    @classmethod
    def from_vllm(cls, config):
        values = dict((getattr(config, 'additional_config', None) or {}).get('betterscale_experts', {}))
        if values.pop('experimental', False) is not True:
            raise ValueError('Expert service is experimental; explicit experimental=true required')
        allowed = {'control', 'build', 'owners', 'sources', 'source', 'qualification', 'placement'}
        if values.keys() - allowed:
            raise ValueError(f'Unknown expert options: {sorted(values.keys() - allowed)}')
        spec = config.speculative_config
        if spec is not None and (spec.method != 'mtp' or spec.num_speculative_tokens != 2):
            raise ValueError('Expert service supports no speculation or native MTP2')
        result = cls(**values, draft_layers=int(spec is not None))
        result.validate()
        return result

    def validate(self):
        if self.placement not in ('layer', 'expert'):
            raise ValueError('Unsupported expert placement')
        if any(type(x) is not int for x in (self.owners, self.sources, self.source, self.draft_layers)):
            raise ValueError('Expert topology fields must be integers')
        if self.owners not in (2, 4) or not 1 <= self.sources <= 7 or self.owners + self.sources > 8:
            raise ValueError('Expert topology requires E2/E4 and <=8 total devices')
        if not 0 <= self.source < self.sources or self.draft_layers not in (0, 1):
            raise ValueError('Invalid attention source or physical draft layer count')
        for name in ('control', 'build'):
            if not Path(getattr(self, name)).is_absolute():
                raise ValueError(f'{name} must be an absolute task-owned path')

    def check_build(self):
        abi = json.loads((Path(self.build) / 'abi.json').read_text())
        route_plan_threshold(abi)
        cap = abi.get('sources_per_wave')
        if type(cap) is not int or not 1 <= cap <= 7:
            raise ValueError('Invalid expert sources_per_wave ABI')
        expected = dict(version=1, model='qwen35-bf16-persistent', hidden=2048,
                        inner=512, topk=8, experts=256, rows=4096, sources=7,
                        layer_count=40+self.draft_layers, persistent_launch_timeout_us=0,
                        external_watchdog=True, combined_return=self.placement=='layer', fine_pack=False)
        expected['placement']=self.placement
        if self.placement=='expert':
            expected.update(expert_owners=self.owners, local_experts=256//self.owners)
        if any(abi.get(k) != v for k, v in expected.items()):
            raise ValueError('Expert binary ABI does not match target/draft service')
        for name in ('launch.so', 'persistent_vector.o', 'persistent_cube.o', 'queue_service.o'):
            if not (Path(self.build) / name).is_file():
                raise ValueError(f'Missing expert binary: {name}')
        if self.qualification is not None and not Path(self.qualification).is_file():
            raise ValueError('Missing exact-build expert qualification receipt')

    def bind(self, model, graph_rows):
        # Geometry is frozen on first import. Reject stale imports rather than
        # mutate modules that may already own device buffers or weight tables.
        module = sys.modules.get('betterscale.patches.expert_service.model_geometry')
        if module is not None and module.GEOMETRY.total_layers != 40+self.draft_layers:
            raise RuntimeError('Expert geometry already bound; start a fresh process')
        env = dict(DRAFT_LAYERS=str(self.draft_layers), MODEL=str(model), PERSISTENT='1',
                   PERSISTENT_BUILD=self.build, PERSISTENT_SERVER_GRAPH='0', NATIVE_CONTROL=self.control,
                   NATIVE_PLACEMENT=self.placement, NATIVE_OWNERS=str(self.owners),
                   NATIVE_SOURCE=str(self.source), GRAPH_BATCH=str(graph_rows),
                   NATIVE_RETAIN_WEIGHTS='1' if self.qualification is None else '0',
                   FINE_PACK='0', RETURN_MODE='pull')
        if self.qualification is not None:
            env['NATIVE_QUALIFICATION'] = self.qualification
        for key, value in env.items():
            name = 'BETTERSCALE_EXPERT_' + key
            if name in os.environ and os.environ[name] != value:
                raise RuntimeError(f'Conflicting expert process configuration: {name}')
        os.environ.update({'BETTERSCALE_EXPERT_'+key: value for key, value in env.items()})
