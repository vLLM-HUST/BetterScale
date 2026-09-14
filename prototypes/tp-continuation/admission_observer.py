"""Experiment-only first-capture receipts; no extra device synchronization."""

import json
import os
from pathlib import Path
import time
from strengthen_dsv4.worker import Worker


class AdmissionWorker(Worker):
    def compile_or_warm_up_model(self):
        result = super().compile_or_warm_up_model()
        r = self.model_runner
        producer, metadata = r._async_decode
        original = r._build_attention_metadata
        p = r.vllm_config.parallel_config
        rank = p.data_parallel_rank * p.tensor_parallel_size + self.rank
        path = (
            Path(os.environ["CONTINUATION_ADMISSION_OUTPUT"])
            / f"admission-rank{rank}.jsonl"
        )
        seen, carriers, skipped = set(), {}, set()

        def build(*args, **kwargs):
            before = len(metadata.entries)
            started = time.monotonic()
            output = original(*args, **kwargs)
            n = kwargs.get("num_reqs", 0)
            nt = kwargs.get("num_tokens_padded") or kwargs.get("num_tokens", 0)
            skip_key = (n, kwargs.get("num_reqs_padded"), nt)
            if (
                producer.active is not None
                and r._cross_step_bounds.admitted
                and nt > r.max_num_reqs * 6
                and skip_key not in skipped
            ):
                skipped.add(skip_key)
                with path.open("a") as stream:
                    stream.write(
                        json.dumps(
                            dict(
                                timestamp=time.time(),
                                native_padding_fallback=True,
                                shape=skip_key,
                                host_seconds=time.monotonic() - started,
                            )
                        )
                        + "\n"
                    )
            if len(metadata.entries) > before:
                for key in metadata.entries.keys() - seen:
                    n, nr, nt, carrier = key
                    carrier = carriers.setdefault(carrier, len(carriers))
                    with path.open("a") as stream:
                        stream.write(
                            json.dumps(
                                dict(
                                    timestamp=time.time(),
                                    local_requests=n,
                                    padded_requests=nr,
                                    padded_tokens=nt,
                                    carrier=carrier,
                                    host_seconds=time.monotonic() - started,
                                )
                            )
                            + "\n"
                        )
                    seen.add(key)
            return output

        r._build_attention_metadata = build
        return result
