"""An execution handoff is capability removal, not another runner wrapper."""
from contextlib import contextmanager

EXECUTION_METHODS = (
    'execute_model', 'sample_tokens', '_model_forward', '_sample',
    '_build_attention_metadata', '_prepare_inputs', '_update_states',
    '_update_full_graph_params_if_needed', '_dummy_run', 'capture_model',
)


@contextmanager
def forbid_runner_execution(runner):
    original = {}
    calls = []
    for name in EXECUTION_METHODS:
        if hasattr(runner, name):
            original[name] = getattr(runner, name)
            def forbidden(*args, _name=name, **kwargs):
                calls.append(_name)
                raise RuntimeError(f'runner execution after ownership handoff: {_name}')
            setattr(runner, name, forbidden)
    try:
        yield calls
    finally:
        for name, value in original.items():
            setattr(runner, name, value)
