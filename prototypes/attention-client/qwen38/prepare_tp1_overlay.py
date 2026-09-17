"""Add TP1 to a disconnected research runtime, preserving the TP2 closure.

TP1 owns both KV heads and all24 Q heads. Indexer selection stays shared across
those heads. Gather/publish move512 contiguous elements per KV row; FIA receives
24 Q heads and2 KV heads. No whole-cache transpose or duplicate indexer call.
This does not qualify TP1 numerics; run the leaf and full-model gates separately.
"""

import argparse
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("source", type=Path)
p.add_argument("output", type=Path)
a = p.parse_args()
source, out = a.source.resolve(), a.output.resolve()
if out.exists():
    raise FileExistsError(out)
edits = {}


def patch(relative, changes):
    text = edits.get(relative, (source / relative).read_text())
    for old, new in changes:
        if text.count(old) != 1:
            raise ValueError(f"Unexpected source at {relative}: {old!r}")
        text = text.replace(old, new)
    edits[relative] = text


prefix = "livemodule/llm/qwen38/"
patch(
    prefix + "contract.py",
    [
        ("SUPPORTED_TP_SIZES = (2, 4, 8)", "SUPPORTED_TP_SIZES = (1, 2, 4, 8)"),
        (
            "return tuple((first, first + 1) for first in range(0, tp_size, 2))",
            "return ((0,),) if tp_size == 1 else tuple((first, first + 1) for first in range(0, tp_size, 2))",
        ),
    ],
)
patch(
    prefix + "parallel.py",
    [
        (
            "    @property\n    def island_index",
            "    @property\n    def island_size(self) -> int:\n        return min(self.world_size, QSA_ISLAND_SIZE)\n\n    @property\n    def island_index",
        ),
        ("return self.rank // QSA_ISLAND_SIZE", "return self.rank // self.island_size"),
        ("return self.rank % QSA_ISLAND_SIZE", "return self.rank % self.island_size"),
        (
            "def island_ranks(self) -> tuple[int, int]:\n        first = self.island_index * QSA_ISLAND_SIZE\n        return first, first + 1",
            "def island_ranks(self) -> tuple[int, ...]:\n        first = self.island_index * self.island_size\n        return tuple(range(first, first + self.island_size))",
        ),
        (
            "island * QSA_ISLAND_SIZE + self.island_rank\n            for island in range(self.world_size // QSA_ISLAND_SIZE)",
            "island * self.island_size + self.island_rank\n            for island in range(self.world_size // self.island_size)",
        ),
        (
            "width = self.contract.num_attention_heads // QSA_ISLAND_SIZE",
            "width = self.contract.num_attention_heads // self.island_size",
        ),
        (
            "return HeadSlice(self.island_rank, self.island_rank + 1)",
            "width = self.contract.num_key_value_heads // self.island_size\n        return HeadSlice(self.island_rank * width, (self.island_rank + 1) * width)",
        ),
        (
            "    selected: GroupCoordinator | None = None",
            "    if plan.world_size == 1:\n        _QSA_GROUP_CACHE[key] = tp\n        return tp\n\n    selected: GroupCoordinator | None = None",
        ),
    ],
)
prefix = "livemodule/arch/ascend/llm/qwen38/"
patch(
    prefix + "qsa_gather.py",
    [
        (
            "    keys = torch.empty((rows, selected, 1, dim), device=query.device, dtype=key_cache.dtype)",
            '    heads = key_cache.shape[2]\n    if heads not in (1, 2) or key_cache.stride(2) != dim or value_cache.stride(2) != dim:\n        raise ValueError("QSA requires contiguous one/two-head KV rows")\n    keys = torch.empty((rows, selected, heads, dim), device=query.device, dtype=key_cache.dtype)',
        ),
        (
            "rows, selected, dim, key_cache.shape[1], table.shape[1],",
            "rows, selected, heads * dim, key_cache.shape[1], table.shape[1],",
        ),
    ],
)
patch(
    prefix + "qsa_publish.py",
    [
        (
            "MS: tl.constexpr, RS: tl.constexpr):\n    d = tl.arange(0, 256)",
            "MS: tl.constexpr, RS: tl.constexpr, WIDTH: tl.constexpr):\n    d = tl.arange(0, WIDTH)",
        ),
        (
            "valid.stride(0), rows.stride(0))",
            "valid.stride(0), rows.stride(0), key_cache.shape[2] * 256)",
        ),
        (
            "key_cache.shape[2:] != (1, 256)",
            "(key_cache.shape[2] not in (1, 2) or key_cache.shape[3] != 256 or key_cache.stride(2) != 256 or value_cache.stride(2) != 256)",
        ),
    ],
)
patch(
    prefix + "qsa_attention.py",
    [
        (
            "key_cache.shape[2] != 1 or key_cache.shape[3] != query.shape[3]",
            "key_cache.shape[2] not in (1, 2) or key_cache.shape[3] != query.shape[3]",
        ),
        (
            "query.shape[2:] != (12, 256)",
            "query.shape[2:] not in ((12, 256), (24, 256))",
        ),
        (
            'raise ValueError("fused Qwen QSA requires BF16 [rows,1,12,256]")',
            'raise ValueError("fused Qwen QSA requires BF16 Q heads12 or24, D256")\n        if query.shape[2] != keys.shape[2] * 12:\n            raise ValueError("QSA Q/KV head ownership disagrees")',
        ),
        (
            'keys.squeeze(2)[:, None].contiguous(),\n            values.squeeze(2)[:, None].contiguous(),\n            num_heads=12, num_key_value_heads=1, input_layout="BNSD",',
            'keys.transpose(1, 2).contiguous(),\n            values.transpose(1, 2).contiguous(),\n            num_heads=query.shape[2], num_key_value_heads=keys.shape[2], input_layout="BNSD",',
        ),
    ],
)


# Produce FIA's B,H,S,D backing directly while preserving the public B,S,H,D
# view. The transpose at the FIA call is then metadata-only for either TP width.
patch(
    prefix + "qsa_gather.py",
    [
        (
            "DIM: tl.constexpr,\n            PAGE:",
            "DIM: tl.constexpr, HEADS: tl.constexpr,\n            PAGE:",
        ),
        (
            "    keys = torch.empty((rows, selected, heads, dim), device=query.device, dtype=key_cache.dtype)\n    values = torch.empty_like(keys)",
            "    keys = torch.empty((rows, heads, selected, dim), device=query.device, dtype=key_cache.dtype).transpose(1, 2)\n    values = torch.empty((rows, heads, selected, dim), device=query.device, dtype=key_cache.dtype).transpose(1, 2)",
        ),
        (
            "rows, selected, heads * dim, key_cache.shape[1], table.shape[1],",
            "rows, selected, heads * dim, heads, key_cache.shape[1], table.shape[1],",
        ),
        (
            "out = (row.to(tl.int64) * SELECTED + column[:, None]) * DIM + d[None, :]",
            "out = (row.to(tl.int64) * HEADS + d[None, :] // 256) * SELECTED * 256 + column[:, None] * 256 + d[None, :] % 256",
        ),
        (
            "            flat = (tile % tiles_per_row) * TILE * DIM + tl.arange(0, TILE * DIM)\n            tl.store(OutK + row.to(tl.int64) * SELECTED * DIM + flat, 0,\n                     flat < SELECTED * DIM)\n            tl.store(OutV + row.to(tl.int64) * SELECTED * DIM + flat, 0,\n                     flat < SELECTED * DIM)",
            "            zeros = (tile % tiles_per_row) * TILE * 256 + tl.arange(0, TILE * 256)\n            for head in tl.static_range(HEADS):\n                base = (row.to(tl.int64) * HEADS + head) * SELECTED * 256\n                tl.store(OutK + base + zeros, 0, zeros < SELECTED * 256)\n                tl.store(OutV + base + zeros, 0, zeros < SELECTED * 256)",
        ),
    ],
)
# Two-head active stores must also remain affine. A vector modulo that
# interleaves KV heads turns these writes into a scatter on Ascend.
patch(
    prefix + "qsa_gather.py",
    [
        (
            """            d = tl.arange(0, DIM)
            # Invalid selections need no cache traffic and are zeroed before FIA.
            k = tl.load(K + physical[:, None] * KS0 + offset[:, None] * KS1 + d[None, :],
                        valid[:, None], other=0)
            v = tl.load(V + physical[:, None] * VS0 + offset[:, None] * VS1 + d[None, :],
                        valid[:, None], other=0)
            out = (row.to(tl.int64) * HEADS + d[None, :] // 256) * SELECTED * 256 + column[:, None] * 256 + d[None, :] % 256
            tl.store(OutK + out, k, lane[:, None])
            tl.store(OutV + out, v, lane[:, None])""",
            """            d = tl.arange(0, 256)
            for head in tl.static_range(HEADS):
                k = tl.load(K + physical[:, None] * KS0 + offset[:, None] * KS1 + head * 256 + d[None, :],
                            valid[:, None], other=0)
                v = tl.load(V + physical[:, None] * VS0 + offset[:, None] * VS1 + head * 256 + d[None, :],
                            valid[:, None], other=0)
                out = ((row.to(tl.int64) * HEADS + head) * SELECTED + column[:, None]) * 256 + d[None, :]
                tl.store(OutK + out, k, lane[:, None])
                tl.store(OutV + out, v, lane[:, None])""",
        ),
    ],
)
patch(
    prefix + "qsa_attention.py",
    [
        (
            "    ) -> torch.Tensor:\n        from livemodule.llm.qwen38.mapped_qsa import Qwen38MappedKVAddress\n        if isinstance(key_cache, Qwen38MappedKVAddress):",
            """    ) -> torch.Tensor:
        # Indexer still scores/selects the complete wave. Only independent
        # attention query rows consume KV in bounded128-row batches, sharing
        # the allocator workspace. This applies equally to TP1 and TP2.
        if query.shape[0] > 128:
            if request_rows is None:
                if block_table.shape[0] != query.shape[0]:
                    raise ValueError("QSA requires one table row per query or explicit request rows")
                request_rows = torch.arange(query.shape[0], device=query.device)
            return torch.cat([
                self.decode(query[start:start+128], key_cache, value_cache,
                            packed_indices[start:start+128], block_table,
                            scale=scale, request_rows=request_rows[start:start+128],
                            leading_pages=leading_pages)
                for start in range(0, query.shape[0], 128)
            ], dim=0)
        from livemodule.llm.qwen38.mapped_qsa import Qwen38MappedKVAddress
        if isinstance(key_cache, Qwen38MappedKVAddress):""",
        ),
        (
            ".view(query.shape[0], indices.shape[1], 1, query.shape[-1])",
            ".view(query.shape[0], indices.shape[1], key_cache.shape[2], query.shape[-1])",
        ),
        (
            "                query.float(),\n                keys.squeeze(2).float().transpose(-1, -2)[:, None],",
            "                query.float().reshape(query.shape[0], keys.shape[2], -1, query.shape[-1]),\n                keys.float().permute(0, 2, 3, 1),",
        ),
        (
            "            values.squeeze(2).float()[:, None],\n        ).to(query.dtype)",
            "            values.float().transpose(1, 2),\n        ).reshape_as(query).to(query.dtype)",
        ),
    ],
)


def mirror(src, dst, relative=""):
    dst.mkdir()
    for child in src.iterdir():
        name = f"{relative}/{child.name}".lstrip("/")
        target = dst / child.name
        if name in edits:
            target.write_text(edits[name])
        elif any(path.startswith(name + "/") for path in edits):
            mirror(child, target, name)
        elif child.name != "__pycache__":
            target.symlink_to(child, target_is_directory=child.is_dir())


out.parent.mkdir(parents=True, exist_ok=True)
mirror(source, out)
print(out)
