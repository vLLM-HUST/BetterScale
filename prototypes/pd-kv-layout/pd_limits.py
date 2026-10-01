"""Explicit experiment envelopes; long-context configuration needs hardware gates."""
import os


def context_limit():
    value=int(os.environ.get("BETTERSCALE_PD_CONTEXT","8192"))
    if value not in (8192,262144):
        raise ValueError("PD context must be wiring8192 or model262144")
    return value


def state_budget():
    # Keep the short wiring experiment unchanged. The long gate uses the
    # previously measured TP2 logical State budget, not R*context page capping.
    return (8<<30) if context_limit()==8192 else int(24.25*(1<<30))


def checkpoint_limit():
    # TP2 target dense:20,480 bytes/token + about64MiB GDN/conv; 256K <6GiB.
    return (512<<20) if context_limit()==8192 else (6<<30)
