"""End-to-end helper that extracts the screen and reads gauge numbers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from number_reader import read_gauges
from screen_detector import extract_screen


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract screen from a raw photo, then read gauge numbers."
    )
    parser.add_argument("image", type=Path, help="Path to the raw photo.")
    parser.add_argument(
        "-o",
        "--screen-output",
        type=Path,
        help="Optional path to save the normalized screen crop.",
    )
    parser.add_argument("--gpu", action="store_true", help="Use GPU for EasyOCR (if available).")
    parser.add_argument("--json", action="store_true", help="Print readings as JSON.")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    screen_path = extract_screen(args.image, args.screen_output)
    readings = read_gauges(screen_path, gpu=args.gpu)

    print(f"Screen saved to: {screen_path}")
    if args.json:
        print(json.dumps(readings, indent=2))
    else:
        for label, values in readings.items():
            joined = ", ".join(values) if values else "<no reading>"
            print(f"{label}: {joined}")


if __name__ == "__main__":
    main()
