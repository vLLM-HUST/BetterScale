"""Experimental live-only Qwen service; launched by the BetterScale CLI."""

import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--context-tokens", type=int, default=512)
    parser.add_argument("--resident-seats", type=int, default=20)
    parser.add_argument("--token-pages", type=int, default=64)
    args = parser.parse_args()
    from .bootstrap import open_model
    from .http import serve

    with open_model(
        args.model,
        context_tokens=args.context_tokens,
        resident_seats=args.resident_seats,
        token_pages=args.token_pages,
    ) as root:
        serve(root, args.model, port=args.port)


if __name__ == "__main__":
    main()
