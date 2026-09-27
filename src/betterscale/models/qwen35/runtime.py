"""Prepare an isolated pinned donor; never edit the installed/shared runtime."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile


ROOT = Path(__file__).parent


def validate(directory):
    """Check every pinned Ascend source before making this directory importable."""
    pins = json.loads((ROOT.parents[1] / "qwen35_pins.json").read_text())
    for item in pins["source_files"]:
        if item["path"].startswith("vllm_ascend/"):
            path = directory / item["path"]
            if not path.is_file() or digest(path) != item["sha256"]:
                raise ValueError(f"Unqualified Qwen35 runtime: {item['path']}")
    package = directory / "vllm_ascend"
    required = [
        package / "libvllm_ascend_kernels.so",
        package
        / "_cann_ops_custom/vendors/custom_transformer/op_api/lib/libcust_opapi.so",
    ]
    if not list(package.glob("vllm_ascend_C*.so")) or not all(
        p.is_file() for p in required
    ):
        raise ValueError(
            "Qwen35 needs a built donor with native payload, not a source-only checkout"
        )
    return directory


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(source, output):
    """Copy a pristine or already-qualified donor, then check exact patch outputs."""
    source, output = source.resolve(), output.absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing to replace runtime: {output}")
    if output.resolve().is_relative_to(source):
        raise ValueError("Runtime output must be outside the source directory")
    contract = json.loads((ROOT / "runtime.json").read_text())
    states = []
    for item in contract["files"]:
        path = source / item["path"]
        actual = digest(path)
        if actual not in (item["before"], item["after"]):
            raise ValueError(f"Unqualified donor input: {item['path']}")
        states.append(actual == item["after"])
    if any(states) and not all(states):
        raise ValueError(
            "Mixed original/adapted donor source; use one complete runtime"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".qwen35-runtime-", dir=output.parent
    ) as temp:
        staged = Path(temp) / "runtime"
        shutil.copytree(
            source / "vllm_ascend",
            staged / "vllm_ascend",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        if not all(states):
            subprocess.run(
                [
                    "patch",
                    "--batch",
                    "--forward",
                    "--fuzz=0",
                    "-p1",
                    "-i",
                    str(ROOT / "runtime.patch"),
                ],
                cwd=staged,
                check=True,
                capture_output=True,
                text=True,
            )
        validate(staged)
        # The destination was not caller-owned. Never refresh a live runtime in place.
        if output.exists() or output.is_symlink():
            raise FileExistsError(output)
        staged.rename(output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Pinned donor root containing vllm_ascend/",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New, nonexistent isolated runtime directory",
    )
    args = parser.parse_args()
    print(prepare(args.source, args.output))


if __name__ == "__main__":
    main()
