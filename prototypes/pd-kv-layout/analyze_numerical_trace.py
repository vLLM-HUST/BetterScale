"""Compare bounded white-box prompt tensors to repeat zero (CPU only)."""
import argparse
import json
from pathlib import Path
import torch


def differences(reference, candidate):
    assert reference.shape == candidate.shape and reference.dtype == candidate.dtype
    delta = (reference.float() - candidate.float()).abs()
    return dict(equal=torch.equal(reference, candidate),
                changed=int(torch.count_nonzero(delta)), elements=delta.numel(),
                max_abs=float(delta.max()), mean_abs=float(delta.mean()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    comparisons = []
    for rank in (2, 3):
        files = sorted(args.root.glob(f"rank{rank}-prefill*.pt"),
                       key=lambda p: int(p.stem.split("prefill")[1]))
        assert len(files) >= 2, (rank, len(files))
        reference = torch.load(files[0], map_location="cpu", weights_only=True)
        for file in files[1:]:
            candidate = torch.load(file, map_location="cpu", weights_only=True)
            row = dict(rank=rank, index=candidate["index"], layers={}, moe={}, inputs={})
            for index, states in reference["layers"].items():
                row["layers"][index] = {key: differences(value, candidate["layers"][index][key])
                                       for key, value in states.items()}
            for key, value in reference.get("moe", {}).items():
                row["moe"][key] = differences(value, candidate["moe"][key])
            for key, value in reference["inputs"].items():
                # Inputs are graph-padded; only the 281 prompt positions are live.
                row["inputs"][key] = differences(value[..., :281], candidate["inputs"][key][..., :281])
            comparisons.append(row)
    (args.root/"comparison.json").write_text(json.dumps(comparisons, indent=2))
    for category in ("inputs", "moe"):
        for key in comparisons[0][category]:
            metrics = [row[category][key] for row in comparisons]
            print(category, key, "exact", sum(m["equal"] for m in metrics), "/", len(metrics),
                  "max_abs", max(m["max_abs"] for m in metrics))
    for index in comparisons[0]["layers"]:
        for key in ("hidden", "residual"):
            metrics = [row["layers"][index][key] for row in comparisons]
            print("layer", index, key, "exact", sum(m["equal"] for m in metrics), "/", len(metrics),
                  "max_abs", max(m["max_abs"] for m in metrics))


if __name__ == "__main__":
    main()

