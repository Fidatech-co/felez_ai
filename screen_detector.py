"""Utility to isolate a screen/monitor from a photo.

The script searches for the largest quadrilateral contour in the image and
warps it so the result is a straightened screen crop.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


def order_points(points: np.ndarray) -> np.ndarray:
    """Return the four points arranged as top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = points.sum(axis=1)
    rect[0] = points[np.argmin(s)]
    rect[2] = points[np.argmax(s)]

    diff = np.diff(points, axis=1)
    rect[1] = points[np.argmin(diff)]
    rect[3] = points[np.argmax(diff)]
    return rect


def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a perspective transform so the four points become a rectangle."""
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = int(max(width_a, width_b))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = int(max(height_a, height_b))

    dst = np.array(
        [[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]],
        dtype="float32",
    )
    matrix = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, matrix, (max_width, max_height))


def find_screen_contour(image: np.ndarray) -> np.ndarray:
    """Locate the largest 4-point contour that likely corresponds to a screen."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) == 4:
            return approx.reshape(4, 2)

    raise ValueError("Unable to find a screen-like contour in the image.")


def enhance_screen(image: np.ndarray) -> np.ndarray:
    """Convert to high-contrast black/white and resize to 800x600."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)
    blurred = cv2.GaussianBlur(equalized, (3, 3), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.resize(binary, (800, 600), interpolation=cv2.INTER_AREA)


def extract_screen(input_path: Path, output_path: Path | None) -> Path:
    """Extract the screen from the input image, writing the result to output_path."""
    image = cv2.imread(str(input_path))
    if image is None:
        raise FileNotFoundError(f"Unable to read image at {input_path}")

    contour = find_screen_contour(image)
    warped = four_point_transform(image, contour.astype("float32"))
    processed = enhance_screen(warped)

    if output_path is None:
        output_path = input_path.with_name(f"{input_path.stem}_screen.png")
    cv2.imwrite(str(output_path), processed)
    return output_path


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract a screen region from an image.")
    parser.add_argument("image", type=Path, help="Path to the photo containing a screen.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Where to save the extracted screen (default: <image>_screen.png).",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    output_path = extract_screen(args.image, args.output)
    print(f"Screen extracted to: {output_path}")


if __name__ == "__main__":
    main()
