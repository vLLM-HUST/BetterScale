"""Shared wire geometry, in int32 words; negotiated before exporting memory.

Capacity is fixed for a binary/channel, while each descriptor carries actual
rows. Metadata never overlaps scales or either BF16/INT8 payload. Keep the
legacy 32-row offsets so old, explicitly identified ABI2 capsules still work.
"""

from dataclasses import dataclass

ALIGN = 2 * 1024**2


def align(value, multiple):
    return (value + multiple - 1) // multiple * multiple


@dataclass(frozen=True)
class ChannelLayout:
    rows: int = 32

    def __post_init__(self):
        # Coordinator/worker routing metadata currently occupies one 64KiB UB.
        # This is a deliberate bound, not a hardware serving-capacity claim.
        if self.rows not in (32, 128, 256, 512, 1024):
            raise ValueError("channel capacity must be 32, 128, 256, 512 or 1024")

    @property
    def routes(self):
        return self.rows * 10

    @property
    def map_words(self):
        return self.routes + 8

    @property
    def scales(self):
        return max(512, align(64 + self.routes, 128))

    @property
    def payload(self):
        return max(1024, align(self.scales + self.rows * 8, 128))

    @property
    def source_bytes(self):
        return align(4 * self.payload + self.rows * 2560 * 2, ALIGN)

    @property
    def output_bytes(self):
        return align(64 * 4 + self.routes * 2560 * 2, ALIGN)

    def contract(self):
        return dict(
            version=1,
            model="qwen38",
            hidden=2560,
            inner=640,
            topk=10,
            experts=512,
            owners=4,
            rows=self.rows,
            input="target_int8_scale_mtp_bf16",
            client_words=17,
            prefix_pipeline=False,
        )

    @classmethod
    def from_abi(cls, abi):
        layout = cls(abi["rows"])
        if abi.get("version") == 2 and layout.rows == 32:
            return layout
        if (
            abi.get("version") != 3
            or abi.get("source_scale_words") != layout.scales
            or abi.get("source_payload_words") != layout.payload
        ):
            raise ValueError("binary and channel layout disagree")
        return layout
