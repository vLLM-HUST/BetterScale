"""Read-only observation RPCs for the actual packaged Worker; no runtime hooks at import."""

import json
import os
from pathlib import Path
from types import SimpleNamespace


class PackagedObserver:

    def start_window(self, label, profile=False):
        from diagnostics import start_decode_observation

        p = self.model_runner.vllm_config.parallel_config
        rank = p.data_parallel_rank * p.tensor_parallel_size + self.rank
        root = Path(os.environ["DONOR_DP_OUTPUT"]) / label
        root.mkdir(parents=True, exist_ok=True)
        os.environ["FULL_MIXED_OUTPUT"] = str(root)
        self._donor_observer = SimpleNamespace(
            rank=rank, model_runner=self.model_runner
        )
        receipt = start_decode_observation(
            self._donor_observer,
            profile,
            label,
            profile_steps=int(os.environ.get("DONOR_DP_PROFILE_STEPS", "0")),
        )
        runner = self.model_runner
        original = runner._determine_batch_execution_and_padding
        self._donor_modes = []
        self._donor_dummy = False
        dummy = runner._dummy_run

        def observed_dummy(*args, **kwargs):
            previous = self._donor_dummy
            self._donor_dummy = True
            try:
                return dummy(*args, **kwargs)
            finally:
                self._donor_dummy = previous

        self._donor_observer._decode_observation_originals.append(
            (runner, "_dummy_run", dummy)
        )
        runner._dummy_run = observed_dummy

        def observed(*args, **kwargs):
            result = original(*args, **kwargs)
            self._donor_modes.append(
                dict(
                    mode=str(result[0]),
                    padded=result[1].num_tokens,
                    dummy=self._donor_dummy,
                    across_dp=None if result[3] is None else result[3].tolist(),
                    actual_tokens=int(
                        kwargs.get("num_tokens", args[0] if args else -1)
                    ),
                    actual_requests=int(
                        kwargs.get("num_reqs", args[1] if len(args) > 1 else -1)
                    ),
                )
            )
            return result

        self._donor_observer._decode_observation_originals.append(
            (runner, "_determine_batch_execution_and_padding", original)
        )
        runner._determine_batch_execution_and_padding = observed
        return receipt

    def stop_window(self):
        from diagnostics import stop_decode_observation

        result = stop_decode_observation(self._donor_observer)
        root = Path(os.environ["FULL_MIXED_OUTPUT"])
        label = self._donor_observer._decode_observation_label
        (root / f"{label}-modes-rank{self._donor_observer.rank}.json").write_text(
            json.dumps(self._donor_modes, indent=2)
        )
        return result

    def donor_receipt(self):
        import torch
        from vllm.v1.core.kv_cache_utils import get_kv_cache_capacity

        r = self.model_runner
        producer, metadata = r._async_decode
        tokens, concurrency = get_kv_cache_capacity(r.vllm_config, r.kv_cache_config)
        return dict(
            rank=r.vllm_config.parallel_config.data_parallel_rank,
            kv_capacity_tokens=tokens,
            max_length_concurrency=concurrency,
            allocated=torch.npu.memory_allocated(),
            reserved=torch.npu.memory_reserved(),
            peak=torch.npu.max_memory_allocated(),
            target_replays=r.model._decode_pair.replays,
            preparation_replays=producer.sequence,
            preparation_banks=len(producer.slots),
            metadata_entries=len(metadata.entries),
            metadata_replays=metadata.replays,
        )
