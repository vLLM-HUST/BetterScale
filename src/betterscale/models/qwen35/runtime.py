"""Prepare an isolated pinned donor; never edit the installed/shared runtime."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile


ROOT = Path(__file__).parent


def profiles():
    """Return every complete donor contract shipped with this source."""
    result = []
    for contract_path in sorted(ROOT.glob("runtime*.json")):
        contract = json.loads(contract_path.read_text())
        patch_path = contract_path.with_suffix(".patch")
        if not patch_path.is_file():
            raise ValueError(f"Missing Qwen35 runtime patch: {patch_path.name}")
        result.append(
            dict(
                contract=contract,
                contract_path=contract_path,
                patch_path=patch_path,
                pins_name=contract.get("pins", "qwen35_pins.json"),
            )
        )
    if not result:
        raise ValueError("No Qwen35 runtime contracts are packaged")
    return result


def matching_profile(directory):
    """Identify one fully patched donor profile from exact source identities."""
    matches = []
    for profile in profiles():
        pins_path = ROOT.parents[1] / profile["pins_name"]
        pins = json.loads(pins_path.read_text())
        items = [
            item
            for item in pins["source_files"]
            if item["path"].startswith("vllm_ascend/")
        ]
        if items and all(
            (directory / item["path"]).is_file()
            and digest(directory / item["path"]) == item["sha256"]
            for item in items
        ):
            matches.append(profile)
    if len(matches) != 1:
        raise ValueError(
            "Unqualified Qwen35 runtime: expected exactly one complete donor profile"
        )
    return matches[0]


def validate(directory):
    """Check one complete pinned Ascend profile before making it importable."""
    matching_profile(directory)
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
    candidates = []
    mixed = False
    for profile in profiles():
        states = []
        for item in profile["contract"]["files"]:
            path = source / item["path"]
            if not path.is_file():
                break
            actual = digest(path)
            if actual not in (item["before"], item["after"]):
                break
            states.append(actual == item["after"])
        else:
            if any(states) and not all(states):
                mixed = True
            else:
                candidates.append((profile, all(states)))
    if len(candidates) != 1:
        if mixed:
            raise ValueError(
                "Mixed original/adapted donor source; use one complete runtime"
            )
        raise ValueError("Unqualified donor input: no complete runtime profile")
    profile, already_adapted = candidates[0]
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
        if not already_adapted:
            subprocess.run(
                [
                    "patch",
                    "--batch",
                    "--forward",
                    "--fuzz=0",
                    "-p1",
                    "-i",
                    str(profile["patch_path"]),
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
