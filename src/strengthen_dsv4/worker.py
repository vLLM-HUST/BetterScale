"""Native worker lifecycle integration, selected explicitly with --worker-cls."""
import json
import logging
import os
from pathlib import Path

from .compat import check_runtime
from .config import PATCH_IDS, validate_worker_config
from vllm_ascend.worker.worker import NPUWorker

log=logging.getLogger(__name__)


class Worker(NPUWorker):
    def __init__(self, vllm_config, *args, **kwargs):
        self.strengthen_compat=check_runtime()
        policy=validate_worker_config(vllm_config)
        self.strengthen_profile=policy['profile']
        self.strengthen_artifacts=Path(policy['artifacts'])
        self.strengthen_artifacts.mkdir(parents=True,exist_ok=True)
        os.environ['STRENGTHEN_DSV4_ARTIFACTS']=str(self.strengthen_artifacts)
        from .patches.target import install
        install(self.strengthen_profile=='optimized')
        super().__init__(vllm_config,*args,**kwargs)

    def compile_or_warm_up_model(self):
        result=super().compile_or_warm_up_model()
        if self.strengthen_profile=='optimized':
            from .patches.split_draft import install as split
            from .patches.cross_step import install as cut
            from .patches.ordered_replay import configure as order
            from .patches.qli_cpu import configure as qli
            split(self); cut(self); order(self,True); qli(self,True,False)
        receipt=self.strengthen_status()
        if receipt['max_length_concurrency'] < 4:
            raise RuntimeError('KV budget cannot admit four full-length requests; increase --kv-gib')
        self._write_strengthen_status('ready',receipt)
        log.info('strengthen-dsv4 rank=%s profile=%s READY patches=%s',
                 self.rank,self.strengthen_profile,receipt['patches'])
        return result

    def strengthen_status(self):
        import torch
        from vllm.v1.core.kv_cache_utils import get_kv_cache_capacity
        r=self.model_runner
        tokens,concurrency=get_kv_cache_capacity(r.vllm_config,r.kv_cache_config)
        result=dict(rank=self.rank,profile=self.strengthen_profile,
            patches=list(PATCH_IDS if self.strengthen_profile=='optimized' else ('compat-lcm',)),
            compatibility=self.strengthen_compat,kv_capacity_tokens=tokens,
            max_length_concurrency=concurrency,allocated=torch.npu.memory_allocated(),
            reserved=torch.npu.memory_reserved(),peak=torch.npu.max_memory_allocated())
        if hasattr(self,'_exact_draft_graph'):
            result['draft']=self._exact_draft_graph.receipt()
        if hasattr(r,'_cross_step_bounds'):
            result['cross_step']=r._cross_step_bounds.receipt()
        return result

    def _write_strengthen_status(self,phase,result):
        path=self.strengthen_artifacts/f'{phase}-rank{self.rank}.json'
        temporary=path.with_suffix('.tmp')
        temporary.write_text(json.dumps(result,indent=2))
        temporary.replace(path)

    def shutdown(self):
        try:
            if getattr(self,'model_runner',None) is not None and hasattr(self.model_runner,'kv_cache_config'):
                self._write_strengthen_status('shutdown',self.strengthen_status())
        except Exception:
            log.exception('Unable to save strengthen-dsv4 shutdown receipt')
        finally:
            super().shutdown()
