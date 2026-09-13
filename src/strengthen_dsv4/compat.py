"""Fail closed on pinned private APIs; do not infer compatibility from a version."""

from functools import lru_cache
import hashlib
from importlib import metadata, resources
import json


def pins():
    return json.loads(
        resources.files("strengthen_dsv4").joinpath("pins.json").read_text()
    )


@lru_cache(maxsize=1)
def check_runtime():
    expected = pins()
    actual = {}
    failures = []
    for name, version in expected["versions"].items():
        try:
            actual[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            failures.append(f"{name}: not installed")
            continue
        if actual[name].split("+", 1)[0] != version:
            failures.append(f"{name}: expected {version}, found {actual[name]}")
    for item in expected["source_files"]:
        try:
            path = metadata.distribution(item["distribution"]).locate_file(item["path"])
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        except (metadata.PackageNotFoundError, FileNotFoundError):
            failures.append(item["path"] + ": missing")
            continue
        if digest != item["sha256"]:
            failures.append(item["path"] + ": differs from the qualified pin")
    if failures:
        raise RuntimeError(
            "Incompatible donor environment; no patches applied:\n"
            + "\n".join(failures)
        )
    return dict(
        versions=actual,
        pins=expected["commits"],
        checked_source_files=len(expected["source_files"]),
    )
