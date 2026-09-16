"""Compare logical cold-prefill state across padding policies, not physical slots."""

import json
import os
from pathlib import Path
import re
import torch
from bucket_full_worker import Worker as BaseWorker


def install():
    from vllm.forward_context import get_forward_context
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

    original = NPUModelRunner._model_forward

    def forward(self, *args, **kwargs):
        output = original(self, *args, **kwargs)
        label = getattr(self, "_logical_label", None)
        if label is None:
            return output
        self._logical_label = None
        torch.npu.synchronize()
        ctx = get_forward_context()
        actual = self._logical_tokens
        values = {"hidden": output[:actual].detach().cpu()}
        for name, metadata in ctx.attn_metadata.items():
            match = re.search(r"language_model\.model\.layers\.(\d+)\.", name)
            if match is None or int(match.group(1)) >= 64:
                continue
            cache = self.compilation_config.static_forward_context[name].kv_cache
            if type(metadata).__name__ == "GDNAttentionMetadata":
                slot = int(metadata.non_spec_state_indices_tensor[0].item())
                values[name + ".conv"] = cache[0][slot].detach().cpu()
                values[name + ".ssm"] = cache[1][slot].detach().cpu()
            elif type(metadata).__name__ == "AscendMetadata":
                block_size = cache[0].shape[1]
                count = (actual + block_size - 1) // block_size
                blocks = metadata.block_tables[0, :count].long()
                for kind, tensor in zip(["key", "value"], cache):
                    selected = tensor.index_select(0, blocks).flatten(0, 1)[:actual]
                    values[name + "." + kind] = selected.detach().cpu()
        assert len(values) == 129, (len(values), list(values))
        if label == "native":
            self._logical_reference = values
        checks = []
        for name, value in values.items():
            reference = self._logical_reference[name]
            delta = value.float() - reference.float()
            checks.append(
                dict(
                    name=name,
                    dtype=str(value.dtype),
                    shape=list(value.shape),
                    max_abs=float(delta.abs().max()),
                    rms=float(delta.square().mean().sqrt()),
                    bf16_reference_max_abs=float(
                        (value.float() - reference.bfloat16().float()).abs().max()
                    ),
                    finite=bool(
                        torch.isfinite(value).all() and torch.isfinite(reference).all()
                    ),
                    exact=torch.equal(value, reference),
                )
            )
        path = (
            Path(os.environ["CAPSULE"])
            / f"logical-{label}-rank{self._logical_rank}.json"
        )
        path.write_text(json.dumps(dict(label=label, checks=checks), indent=2))
        return output

    NPUModelRunner._model_forward = forward


class Worker(BaseWorker):
    def __init__(self, *args, **kwargs):
        install()
        super().__init__(*args, **kwargs)

    def arm_logical(self, label, tokens):
        self.model_runner._logical_label = label
        self.model_runner._logical_tokens = tokens
        self.model_runner._logical_rank = self.rank
        return dict(rank=self.rank, label=label)
