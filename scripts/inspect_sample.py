#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--n", type=int, default=5)
    args = parser.parse_args()

    with Path(args.input).open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx >= args.n:
                break
            print(json.dumps(json.loads(line), indent=2))


if __name__ == "__main__":
    main()
