"""Native entry: vllm serve ... --worker-cls betterscale.worker.Worker.

This is the same qualified Worker, not another subclass or execution path.
The legacy strengthen_dsv4.worker.Worker import remains supported.
"""

from strengthen_dsv4.worker import Worker

__all__ = ["Worker"]
