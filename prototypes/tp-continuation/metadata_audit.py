"""One-shot diagnostic: captured TP metadata versus its native construction."""

import dataclasses
import json
import os
from pathlib import Path
import torch


def fields(value, path=(), seen=None):
    if seen is None:
        seen = set()
    if id(value) in seen:
        return
    seen.add(id(value))
    if isinstance(value, torch.Tensor):
        yield str(path), value
    elif dataclasses.is_dataclass(value):
        for f in dataclasses.fields(value):
            yield from fields(getattr(value, f.name), path + (f.name,), seen)
    elif type(value).__name__ == "RopeDataProxy":
        yield from fields(value._data, path + ("rope",), seen)
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from fields(item, path + (key,), seen)
    elif isinstance(value, (tuple, list)):
        for key, item in enumerate(value):
            yield from fields(item, path + (key,), seen)


def install(worker):
    r = worker.model_runner
    producer, metadata = r._async_decode
    original = r._build_attention_metadata
    done = False

    def build(*args, **kwargs):
        nonlocal done
        output = original(*args, **kwargs)
        if done or producer.active is None or not metadata.replays:
            return output
        done = True
        torch.npu.synchronize()
        actual = dict(fields(output[0]))
        saved = {name: t.clone() for name, t in actual.items()}
        cpu = r.optimistic_seq_lens_cpu.clone()
        records = []
        try:
            for mode in ("same_cpu", "exact_cpu"):
                if mode == "exact_cpu":
                    n = r.input_batch.num_reqs
                    r.optimistic_seq_lens_cpu[:n].copy_(r.seq_lens[:n].cpu())
                reference = metadata.native(*args, **kwargs)
                torch.npu.synchronize()
                expected = dict(fields(reference[0]))
                diffs = []
                for name, value in saved.items():
                    other = expected[name]
                    if value.shape != other.shape:
                        diffs.append(
                            dict(
                                path=name,
                                shape=list(value.shape),
                                other_shape=list(other.shape),
                            )
                        )
                    elif not torch.equal(value, other):
                        diffs.append(
                            dict(
                                path=name,
                                shape=list(value.shape),
                                dtype=str(value.dtype),
                                count=int((value != other).sum().cpu()),
                                max_diff=float(
                                    (value.float() - other.float()).abs().max().cpu()
                                ),
                            )
                        )
                records.append(dict(mode=mode, fields=len(actual), diffs=diffs))
        finally:
            r.optimistic_seq_lens_cpu.copy_(cpu)
            for name, value in actual.items():
                value.copy_(saved[name])
        (
            Path(os.environ["DONOR_DP_OUTPUT"])
            / f"metadata-audit-rank{worker.rank}.json"
        ).write_text(json.dumps(records, indent=2))
        return output

    r._build_attention_metadata = build
