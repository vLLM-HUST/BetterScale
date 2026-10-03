"""Process-local D execution envelope; P and the released path remain C16."""
import os

def execution_capacity():
    value=int(os.environ.get("BETTERSCALE_QWEN35_DECODE_CAPACITY","16"))
    if value not in (16,32,48,64,80):
        raise ValueError("D execution capacity must be16/32/48/64/80")
    if value!=16 and os.environ.get("BETTERSCALE_PD_DECODE_ONLY")!="1":
        raise ValueError("Wider execution requires the explicit D-only path")
    return value

EXECUTION=execution_capacity()
