"""Single public entry; native configuration selects locally owned overrides."""

from threading import RLock

from vllm_ascend.worker.worker import NPUWorker
from .models import select

# Native hooks are process-global. Reject incompatible composition and poison
# partial bootstrap rather than pretending monkey patches can be rolled back.
_lock = RLock()
_route = None
_failed = False


class Worker(NPUWorker):
    def __init__(self, vllm_config, *args, **kwargs):
        global _route, _failed
        implementation, route = select(vllm_config)
        with _lock:
            if _failed or (_route is not None and _route != route):
                raise RuntimeError(
                    "BetterScale composition conflict; start a fresh process"
                )
            implementation.check(vllm_config)
            self._patches = implementation
            _route = route
            try:
                implementation.before_init(self, vllm_config)
                super().__init__(vllm_config, *args, **kwargs)
                hook = getattr(implementation, "after_init", None)
                if hook is not None:
                    hook(self)
            except BaseException:
                _failed = True
                raise

    def _init_device(self):
        hook = getattr(self._patches, "init_device", None)
        native = super()._init_device
        return native() if hook is None else hook(self, native)

    def determine_available_memory(self):
        hook = getattr(self._patches, "determine_available_memory", None)
        native = super().determine_available_memory
        return native() if hook is None else hook(self, native)

    def load_model(self, *args, **kwargs):
        result = super().load_model(*args, **kwargs)
        hook = getattr(self._patches, "model_loaded", None)
        if hook is not None:
            hook(self)
        return result

    def compile_or_warm_up_model(self):
        result = super().compile_or_warm_up_model()
        hook = getattr(self._patches, "warmed", None)
        if hook is not None:
            hook(self)
        return result
