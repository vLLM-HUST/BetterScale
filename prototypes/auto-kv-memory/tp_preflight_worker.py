"""Dummy-only entry; keep fixture layout repair out of the real-weight path."""

from dummy_weights import install as install_dummy_layout
from tp_physical_worker import TPPhysicalWorker


class TPPreflightWorker(TPPhysicalWorker):
    def __init__(self, vllm_config, *args, **kwargs):
        # Fail before installing the process-wide DummyModelLoader hook, not
        # after native model initialization or communication warmup.
        assert vllm_config.load_config.load_format == "dummy"
        install_dummy_layout()
        super().__init__(vllm_config, *args, **kwargs)
