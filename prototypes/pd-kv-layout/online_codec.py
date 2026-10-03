"""Lossless resident-wire adapter; State/layout and DMA contracts stay unchanged.

Use standard Zstandard frames only when large objects shrink substantially.
Dense pages stay raw. The transport namespace must separate this representation
from raw-only producers. No compressor/decompressor context is shared by threads.
"""
MAX_WIRE=(129<<20)+4
MIN_COMPRESS=64<<20
MAGIC=b"\x28\xb5\x2f\xfd"
PIN="0.25.0"


class ResidentWireSink:
    def __init__(self,sink):
        import zstandard
        if zstandard.__version__!=PIN:raise RuntimeError("Unqualified wire codec version")
        self.sink=sink;self.codec=zstandard

    def encode(self,data):
        if not 4<len(data)<=MAX_WIRE:raise ValueError("Invalid State wire size")
        if len(data)<MIN_COMPRESS:return data
        encoded=self.codec.ZstdCompressor(level=1,threads=0,
            write_content_size=True,write_checksum=True).compress(data)
        return encoded if len(encoded)*5<=len(data)*3 else data

    def decode(self,data):
        if not 4<len(data)<=MAX_WIRE:raise ValueError("Invalid State wire size")
        if not data.startswith(MAGIC):return data
        try:
            params=self.codec.get_frame_parameters(data)
            size=params.content_size
            # max_output_size alone is not an allocation cap for known-size
            # frames. Validate the embedded size before invoking decompression.
            if (not 4<size<=MAX_WIRE or params.window_size>MAX_WIRE
                    or params.dict_id or not params.has_checksum):
                raise ValueError("Unqualified compressed State frame")
            value=self.codec.ZstdDecompressor(max_window_size=(MAX_WIRE+1023)//1024).decompress(
                data,max_output_size=size,read_across_frames=False,allow_extra_data=False)
        except self.codec.ZstdError as error:
            raise ValueError("Invalid compressed State frame") from error
        if len(value)!=size:raise ValueError("State frame size mismatch")
        return value

    def ensure(self,key):return self.sink.ensure(key)
    def put(self,key,data):return self.sink.put(key,self.encode(data))
    def get(self,key):return self.decode(self.sink.get(key))
