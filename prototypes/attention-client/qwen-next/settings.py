"""Fixed A2/E4 experimental contract; not an automatic production placement."""

import os

LAYERS = int(os.environ.get("NEXT_LAYERS", "4"))
REAL = os.environ.get("NEXT_REAL") == "1"
MODEL = "/data/shared_models/Qwen3-Next-80B-A3B-Instruct"
assert LAYERS in (4, 48)
CONTRACT = dict(
    version=2,
    layers=LAYERS,
    hidden=2048,
    intermediate=512,
    experts=512,
    topk=10,
    sources=2,
    owners=4,
    rows=32,
    dtype="bf16",
    real=REAL,
)
