"""Explicit experiment envelopes; long-context configuration needs hardware gates."""
import os


def context_limit():
    value=int(os.environ.get("BETTERSCALE_PD_CONTEXT","8192"))
    if value not in (8192,262144):
        raise ValueError("PD context must be wiring8192 or model262144")
    return value


def state_budget(kind=None):
    # Keep the short wiring experiment unchanged. The long gate uses the
    # previously measured TP2 logical State budget, not R*context page capping.
    default = (8<<30) if context_limit()==8192 else int(24.25*(1<<30))
    raw = os.environ.get("BETTERSCALE_PD_D_STATE_GIB")
    if kind != "D" or raw is None:
        return default
    from decimal import Decimal, InvalidOperation
    try:
        value = Decimal(raw)
        if not value.is_finite() or not 8 <= value <= 48:
            raise ValueError("D State budget must be finite and in8..48GiB")
    except InvalidOperation as error:
        raise ValueError("Invalid D State budget") from error
    return int(value * (1<<30))


def checkpoint_limit():
    # TP2 per-rank MTP dense:11,264 bytes/token plus resident State; 256K <6GiB.
    return (512<<20) if context_limit()==8192 else (6<<30)
