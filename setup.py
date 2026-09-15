"""The only native payload is a qualified Linux/aarch64 CANN host-tiling library."""

import hashlib
import json
from pathlib import Path
from setuptools import setup
from setuptools.command.bdist_wheel import bdist_wheel
from setuptools.command.build_py import build_py
from setuptools.command.sdist import sdist


def check_native():
    root = Path(__file__).parent / "src/betterscale/patches/hc_workspace"
    library = root / "libcust_opmaster_rt2.0.so"
    expected = json.loads((root / "native.json").read_text())[
        "candidate_library_sha256"
    ]
    if (
        not library.is_file()
        or hashlib.sha256(library.read_bytes()).hexdigest() != expected
    ):
        raise RuntimeError(
            "Missing/unqualified HC-pre binary; see patches/hc_workspace/README.md"
        )


class NativeBuild(build_py):
    def run(self):
        check_native()
        super().run()


class NativeSource(sdist):
    def run(self):
        check_native()
        super().run()


class NativeWheel(bdist_wheel):
    def finalize_options(self):
        super().finalize_options()
        self.root_is_pure = False

    def get_tag(self):
        return "py3", "none", "linux_aarch64"


setup(
    cmdclass={
        "build_py": NativeBuild,
        "sdist": NativeSource,
        "bdist_wheel": NativeWheel,
    }
)
