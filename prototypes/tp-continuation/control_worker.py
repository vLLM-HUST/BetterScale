"""Experiment-only ablation: kept FULL target / split-draft without new banks.

DP uses the existing native-DSA FULL patch plus cross-step receipt cut. TP is
main@7f7bda3's unchanged composition. This is NOT the untouched stock donor.
No selector is added to the delivered Worker.
"""

from strengthen_dsv4.compat import check_runtime
from strengthen_dsv4.config import validate_worker_config
from vllm_ascend.worker.worker import NPUWorker


class ControlWorker(NPUWorker):
    def __init__(self, config, *args, **kwargs):
        check_runtime()
        validate_worker_config(config)
        from strengthen_dsv4.patches import compat_lcm, target_full

        self.native_dp = config.parallel_config.tensor_parallel_size == 1
        compat_lcm.install()
        target_full.install(native_dsa=self.native_dp)
        super().__init__(config, *args, **kwargs)

    def compile_or_warm_up_model(self):
        result = super().compile_or_warm_up_model()
        from strengthen_dsv4.patches import cross_step

        if self.native_dp:
            cross_step.install(self, native_dsa=True, max_requests=2)
        else:
            from strengthen_dsv4.patches import split_draft, ordered_replay, qli_cpu

            split_draft.install(self)
            cross_step.install(self)
            ordered_replay.install(self)
            qli_cpu.install(self)
        return result
