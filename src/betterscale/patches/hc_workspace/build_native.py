"""Rebuild from the pinned Ascend checkout; never change that checkout/runtime.

python build_native.py --upstream /path/to/vllm-ascend --cann /path/to/cann-9.0.1
The output is written beside this script for a source-distribution build.
Requires CANN development headers, CMake, Ninja and a Linux aarch64 C++ toolchain.
"""

import argparse
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile

PIN = "9bf964cb4b87c8cd0d6852c41a55b3c29711fa95"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--cann", type=Path, required=True)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    # git archive, not the working tree: unrelated local donor changes cannot
    # enter a native release accidentally. A fresh directory also isolates CMake
    # generators that write into csrc/build despite a different binary directory.
    with tempfile.TemporaryDirectory(prefix="betterscale-native-build-") as directory:
        root = Path(directory)
        archive = root / "source.tar"
        with archive.open("wb") as output:
            subprocess.run(
                ["git", "-C", str(args.upstream), "archive", PIN, "csrc"],
                stdout=output,
                check=True,
            )
        with tarfile.open(archive) as source:
            source.extractall(root, filter="data")
        subprocess.run(
            ["git", "apply", str(here / "workspace.patch")], cwd=root, check=True
        )
        src = root / "csrc"
        build = src / "build-host-a2"
        subprocess.run(
            [
                "cmake",
                "-S",
                str(src),
                "-B",
                str(build),
                "-G",
                "Ninja",
                "-DBUILD_OPEN_PROJECT=ON",
                "-DASCEND_OP_NAME=" + (here / "a2-ops.txt").read_text().strip(),
                "-DCANN_3RD_LIB_PATH=" + str(src / "third_party"),
                "-DCUSTOM_ASCEND_CANN_PACKAGE_PATH=" + str(args.cann.resolve()),
                "-DCHECK_COMPATIBLE=true",
                "-DENABLE_CCACHE=OFF",
                "-DASCEND_COMPUTE_UNIT=ascend910b",
                "-DENABLE_BUILT_IN=OFF",
                "-DENABLE_OPS_HOST=ON",
                "-DENABLE_OPS_KERNEL=OFF",
                "-DENABLE_BUILD_PKG=OFF",
            ],
            check=True,
        )
        subprocess.run(
            ["cmake", "--build", str(build), "--target", "cust_opmaster", "-j4"],
            check=True,
        )
        shutil.copyfile(
            build / "libcust_opmaster_rt2.0.so", here / "libcust_opmaster_rt2.0.so"
        )
    print(
        "Built host tiler. Qualify it and update native.json before publishing a new artifact."
    )


if __name__ == "__main__":
    main()
