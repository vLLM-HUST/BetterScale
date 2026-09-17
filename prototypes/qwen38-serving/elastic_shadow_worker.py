"""Diagnostic FULL/NONE whole-model state oracle; never imported by the MOD."""

from betterscale.qwen_worker import MixedWorker


class Worker(MixedWorker):
    def __init__(self, *args, **kwargs):
        from shadow_worker import install_shadow

        install_shadow()
        from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
        from vllm.forward_context import get_forward_context

        original = NPUModelRunner._model_forward

        def observed(runner, *a, **kw):
            if getattr(runner, "_shadow_remaining", 0):
                ctx = get_forward_context()
                owned = next(
                    m.owned for m in ctx.attn_metadata.values() if hasattr(m, "owned")
                )
                # Independent publication witness, including every GDN group.
                for meta in {
                    id(m.owned): m.owned
                    for m in ctx.attn_metadata.values()
                    if hasattr(m, "owned")
                }.values():
                    if hasattr(meta, "host"):
                        import numpy as np

                        for key, expected in meta.host.items():
                            actual = (
                                meta.indices[key]
                                if isinstance(key, int)
                                else getattr(meta, key)
                            )
                            assert np.array_equal(actual.cpu().numpy(), expected), key
                ends = owned.cu.cpu().tolist()
                lengths = [b - a for a, b in zip(ends, ends[1:]) if b > a]
                slots = owned.slots[: len(lengths)].cpu().tolist()
                assert len(set(slots)) == len(slots) and all(n >= 0 for n in slots)
                runner._owned_partitions.append(
                    dict(
                        lengths=lengths,
                        slots=slots,
                        bank=getattr(ctx.batch_descriptor, "bank", None),
                        capacity=owned.tokens,
                        decode=owned.decode,
                        mode=str(ctx.cudagraph_runtime_mode),
                    )
                )
            return original(runner, *a, **kw)

        NPUModelRunner._model_forward = observed
        super().__init__(*args, **kwargs)

    def load_model(self, *args, **kwargs):
        result = super().load_model(*args, **kwargs)
        import os

        if os.environ.get("ELASTIC_LAYER_DEBUG") == "1":
            from elastic_layer_debug import install

            install()
        return result

    def arm_shadow(self, steps=1):
        r = self.model_runner
        r._shadow_rank = self.rank
        r._shadow_remaining = steps
        r._shadow_results = []
        r._owned_partitions = []
        r._shadow_serial = getattr(r, "_shadow_serial", 0) + 1
        r._shadow_label = f"arm{r._shadow_serial}-"
        return {"rank": self.rank, "armed": steps}

    def shadow_result(self):
        r = self.model_runner
        return dict(
            rank=self.rank,
            remaining=r._shadow_remaining,
            passed=r._shadow_remaining == 0
            and all(x["passed"] for x in r._shadow_results),
            steps=r._shadow_results,
            partitions=r._owned_partitions,
        )
