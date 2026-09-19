"""Legacy import aliases. New commands use betterscale.worker.Worker.

No-MTP configurations now select owned GDN, including APC-off configurations.
The old native-layout single-prefill route is no longer selected by this entry.
"""

from .worker import Worker
from .models.qwen import check_runtime, validate_config

MixedWorker = Worker
