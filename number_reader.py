"""Read numeric indicators from a processed screen image using EasyOCR."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import cv2
import easyocr
import numpy as np

# Map gauge names to their (y1, y2) and (x1, x2) coordinates in pixels.
BOXES: Dict[str, Tuple[Tuple[float, float], Tuple[float, float]]] = {
    "fuel_level": ((298.0, 348.15), (606.51, 766.06)),
    "cool_temp": ((298.0, 348.15), (20.72, 194.80)),
    "sys_volt": ((378.55, 428.28), (20.72, 194.80)),
    "oil_press": ((454.53, 511.18), (20.72, 194.80)),
    "mach_hours": ((377.17, 429.67), (518.09, 739.14)),
    "rpm": ((204.47, 256.97), (334.34, 462.82)),
}


def normalize_bounds(start: float, end: float, limit: int) -> Tuple[int, int]:
    """Convert float coordinates to safe integer bounds within the image."""
    lo = int(round(min(start, end)))
    hi = int(round(max(start, end)))
    lo = max(0, min(lo, limit))
    hi = max(0, min(hi, limit))
    return lo, hi


def preprocess_region(region: np.ndarray) -> np.ndarray:
    """Enhance contrast and resize for better OCR."""
    gray = region if region.ndim == 2 else cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    blurred = cv2.GaussianBlur(resized, (3, 3), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def read_gauges(image_path: Path, gpu: bool = False) -> Dict[str, Iterable[str]]:
    """Run EasyOCR over each predefined box and return recognized numbers."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Unable to read image at {image_path}")

    reader = easyocr.Reader(["en"], gpu=gpu)
    height, width = image.shape[:2]
    results: Dict[str, Iterable[str]] = {}

    for name, ((y1, y2), (x1, x2)) in BOXES.items():
        y_start, y_end = normalize_bounds(y1, y2, height)
        x_start, x_end = normalize_bounds(x1, x2, width)
        region = image[y_start:y_end, x_start:x_end]
        if region.size == 0:
            results[name] = []
            continue
        prepared = preprocess_region(region)
        texts = reader.readtext(prepared, detail=0, allowlist="0123456789.")
        results[name] = texts

    return results


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read gauge numbers from a screen crop.")
    parser.add_argument("image", type=Path, help="Path to the processed screen image.")
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Enable GPU inference for EasyOCR if CUDA is available.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print results as JSON instead of plain text.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    readings = read_gauges(args.image, args.gpu)

    if args.json:
        print(json.dumps(readings, indent=2))
    else:
        for name, values in readings.items():
            joined = ", ".join(values) if values else "<no reading>"
            print(f"{name}: {joined}")


if __name__ == "__main__":
    main()
