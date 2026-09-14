"""DP8 stable-K5 input producer and two-bank target continuation.

The worker installs target packets before native graph capture, then supplies
its cross-step admission object after warmup. No scheduler, launch environment,
post-start RPC, or experiment artifact directory belongs in this patch.
"""


def install_capture():
    from . import _target

    _target.install()


def install(worker):
    import torch
    from . import _host
    from ._producer import DecodeProducer
    from ._metadata import DecodeMetadata
    from ._warmup import prepare

    r = worker.model_runner
    if hasattr(r, "_async_decode"):
        return
    assert r.vllm_config.parallel_config.tensor_parallel_size == 1
    assert hasattr(r, "_cross_step_bounds")
    pair = r.model._decode_pair
    pair.stream = torch.npu.current_stream().npu_stream
    _host.install(worker)
    producer = DecodeProducer(worker)
    metadata = DecodeMetadata(producer)
    prepare(producer, metadata)
    r._async_decode = (producer, metadata)
