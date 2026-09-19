"""Qualified Linux/aarch64 native payloads; installation never compiles operators."""

import hashlib
import json
from pathlib import Path
from setuptools import setup
from setuptools.command.bdist_wheel import bdist_wheel
from setuptools.command.build_py import build_py
from setuptools.command.sdist import sdist


def check_native():
    root = Path(__file__).parent / "src/betterscale/patches"
    artifacts = (
        ("hc_workspace", "libcust_opmaster_rt2.0.so", "candidate_library_sha256"),
        ("qwen_gdn", "libbs_gdn.so", "sha256"),
        ("qwen_gdn", "libbs_gdn_host.so", "host_sha256"),
        ("qwen_fia", "libbs_fia.so", "sha256"),
    )
    for patch, filename, key in artifacts:
        library = root / patch / filename
        expected = json.loads((root / patch / "native.json").read_text())[key]
        if (
            not library.is_file()
            or hashlib.sha256(library.read_bytes()).hexdigest() != expected
        ):
            raise RuntimeError(
                f"Missing/unqualified {patch}/{filename}; see its patch README"
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
