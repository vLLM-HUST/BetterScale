"""Read completed case receipts; reject unmatched work or failed role gates."""

import argparse
import json
from pathlib import Path
from analyze_traces import summarize

TOPOLOGIES = {
    "tp2-e4": (2, 2),
    "tp2-ep8": (2, 4),
    "tp1-e4": (1, 4),
    "tp1-e3": (1, 5),
    "tp1-ep8": (1, 8),
}


def signature(result):
    return [
        tuple(
            s[k]
            for k in (
                "trace_id",
                "completed_turns",
                "output_tokens",
                "prefill_tokens",
                "prefix_reused_tokens",
                "final_encoded_context",
            )
        )
        for s in result["sessions"]
    ]


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "cases", type=Path, help="JSON mapping topology to completed case directory"
    )
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    cases = json.loads(a.cases.read_text())
    records = {}
    expected = None
    for name, directory in cases.items():
        tp, groups = TOPOLOGIES[name]
        directory = Path(directory)
        if (directory / "exit").read_text().strip() != "0":
            raise ValueError(f"case did not pass: {name}")
        capsule = Path((directory / "capsule").read_text().strip())
        parameters = json.loads((directory / "parameters.json").read_text())
        if parameters["layout"] != name or parameters["mode"] != "trace":
            raise ValueError(f"mismatched case metadata: {name}")
        result = summarize(capsule / "roles", groups, tp)
        current = signature(result)
        if expected is not None and current != expected:
            raise ValueError(f"unmatched workload: {name}")
        expected = current
        result.pop("sessions")
        records[name] = dict(capsule=str(capsule), parameters=parameters, **result)
    report = dict(
        scope="same repaired full48+MTP K1 checkpoint, equal-eight-card topology controls; NOT unmodified vLLM or language quality",
        time_scope="maximum source duration after common warmup rendezvous; not precise globally timestamped makespan",
        caveat="native colocated control globally votes phases rather than mixing prefill/decode; differences are not solely expert batching",
        sessions=len(expected or []),
        cases=records,
    )
    a.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
