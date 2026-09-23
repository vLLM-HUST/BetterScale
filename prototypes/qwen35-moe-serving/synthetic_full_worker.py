"""Explicit benchmark-only entry; synthetic responses are not model quality."""
from betterscale.worker import Worker as Parent
from synthetic_acceptance import install


class Worker(Parent):
    def __init__(self, vllm_config, *args, **kwargs):
        install(vllm_config)
        super().__init__(vllm_config, *args, **kwargs)
