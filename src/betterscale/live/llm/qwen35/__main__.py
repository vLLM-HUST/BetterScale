"""Experimental live-only Qwen service; launched by the BetterScale CLI."""

import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--context-tokens", type=int, default=262144)
    parser.add_argument("--execution-seats", type=int, default=16)
    parser.add_argument("--resident-seats", type=int, default=20)
    parser.add_argument("--token-pages", type=int, default=0)
    args = parser.parse_args()
    from .bootstrap import open_model
    from .http import serve

    with open_model(
        args.model,
        context_tokens=args.context_tokens,
        resident_seats=args.resident_seats,
        execution_seats=args.execution_seats,
        token_pages=args.token_pages,
        paged_attention=True,
        prefill_tokens=max((n for n in (4, 16, 64, 256, 1024) if n <= args.context_tokens), default=0),
    ) as root:
        serve(root, args.model, port=args.port)


if __name__ == "__main__":
    main()
