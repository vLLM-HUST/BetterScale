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
        initial_slots = len(producer.slots)
        initial_entries = len(metadata.entries)
        draft = getattr(self, "_exact_draft_graph", None)
        path.write_text(
            json.dumps(
                dict(
                    phase="before_READY_return",
                    producer_banks=initial_slots,
                    metadata_entries=initial_entries,
                    shapes=sorted({key[:3] for key in metadata.entries}),
                    draft_decode=len(draft.decode_graphs) if draft else 0,
                    draft_query=len(draft.query_graphs) if draft else 0,
                )
            )
            + "\n"
        )

        def build(*args, **kwargs):
            before = len(metadata.entries)
            started = time.monotonic()
            output = original(*args, **kwargs)
            assert len(metadata.entries) == initial_entries, "Online metadata capture"
            assert len(producer.slots) == initial_slots, "Online producer admission"
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
        # Qualification only: a missed online graph must fail visibly instead
        # of hiding behind benchmark warmup. No additional device fence.
        if draft is not None:
            import torch

            def online_graph_forbidden(*args, **kwargs):
                raise AssertionError("NPUGraph constructed after worker READY")

            torch.npu.NPUGraph = online_graph_forbidden
        return result
