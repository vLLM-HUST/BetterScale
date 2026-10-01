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



def compare_warm(root):
    """Compare the first continued token to the same cold-prefill position."""
    rows=[]
    for rank in (2,3):
        paths=sorted(root.glob(f"rank{rank}-prefill*.pt"),
                     key=lambda p:int(p.stem.split("prefill")[1]))
        states=[torch.load(p,map_location="cpu",weights_only=True) for p in paths]
        for index,state in enumerate(states):
            if state["actual_tokens"]!=280:
                continue
            warm=states[index+1]
            cold=next(d for d in states[index+2:] if d["actual_tokens"]==281)
            assert warm["actual_tokens"]==1 and warm["last_only"] and cold["last_only"]
            wi,ci=warm["actual_tokens"]-1,cold["actual_tokens"]-1
            assert torch.equal(warm["inputs"]["input_ids"][wi],cold["inputs"]["input_ids"][ci])
            assert torch.equal(warm["inputs"]["positions"][...,wi],cold["inputs"]["positions"][...,ci])
            row=dict(rank=rank,warm_index=warm["index"],cold_index=cold["index"],layers={},gdn={})
            for layer,values in warm["layers"].items():
                row["layers"][layer]={key:differences(value,cold["layers"][layer][key])
                                      for key,value in values.items()}
            for key,value in warm.get("gdn",{}).items():
                row["gdn"][key]=differences(value[-1:],cold["gdn"][key][-1:])
            rows.append(row)
    assert rows, "No warm/cold pairs"
    (root/"warm-cold-comparison.json").write_text(json.dumps(rows,indent=2))
    print(json.dumps(rows[0],indent=2))
    print("pairs",len(rows))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--warm-prefix",action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.warm_prefix:
        return compare_warm(args.root)
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

