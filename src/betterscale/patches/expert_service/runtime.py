"""CANN runtime library; never initializes a device at import."""
import os
from pathlib import Path
LIB = str(Path(os.environ.get("CANN_ROOT", "/usr/local/Ascend/cann-9.0.1")) / "lib64/libascendcl.so")
