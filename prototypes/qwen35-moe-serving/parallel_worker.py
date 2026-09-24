"""Native matrix qualification: shared MTP correctness bridge and load receipts.

Receipts describe actual loaded tensors, not a throughput or correctness seal.
Keep this observer out of the frozen earlier benchmark capsules.
"""
import json
import os
from pathlib import Path

from native_worker import Worker as Native


def describe_model(model):
    rows = []
    for name, module in model.named_modules():
        weights = {key: list(value.shape)
                   for key, value in module.named_parameters(recurse=False)}
        if not weights:
            continue
        if not any(part in name for part in ('experts', 'qkv', 'in_proj')):
            continue
        row = {'name': name, 'class': type(module).__name__, 'weights': weights}
        config = getattr(module, 'moe_config', None)
        if config is not None:
            parallel = config.moe_parallel_config
            row['expert_parallel'] = {
                key: getattr(parallel, key)
                for key in ('tp_size', 'tp_rank', 'dp_size', 'ep_size', 'ep_rank', 'use_ep')}
            row['intermediate_size'] = config.intermediate_size
            mapping = getattr(module, 'expert_map', None)
            row['expert_map'] = None if mapping is None else mapping.cpu().tolist()
        rows.append(row)
    return rows


class PartitionObserver:
    def load_model(self, *args, **kwargs):
        result = super().load_model(*args, **kwargs)
        from betterscale.patches.qwen_mtp_feedback import install
        install(self.model_runner)
        parallel = self.vllm_config.parallel_config
        receipt = {
            'rank': self.rank,
            'attention_tp': parallel.tensor_parallel_size,
            'attention_dp': parallel.data_parallel_size,
            'data_parallel_rank': parallel.data_parallel_rank,
            'enable_expert_parallel': parallel.enable_expert_parallel,
            'correctness_bridge': 'qwen_mtp_feedback/private-host-mailbox/9f58da1',
            'target': describe_model(self.model_runner.model),
        }
        drafter = getattr(self.model_runner, 'drafter', None)
        draft_model = getattr(drafter, 'model', None)
        receipt['draft'] = [] if draft_model is None else describe_model(draft_model)
        root = Path(os.environ['MATRIX_RECEIPTS'])
        root.mkdir(parents=True, exist_ok=True)
        path = root / f'partition-dp{parallel.data_parallel_rank}-rank{self.rank}.json'
        with path.open('x') as stream:
            json.dump(receipt, stream, indent=2)
        return result


class Worker(PartitionObserver, Native):
    pass
