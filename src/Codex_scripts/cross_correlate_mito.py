"""Locate the highest raw cross-correlation between a TIFF and a template.

Run from the project folder:
    uv run src/Codex_scripts/cross_correlate_mito.py
    uv run src/Codex_scripts/cross_correlate_mito.py testImages/Mito_Cropped.tif testImages/Mito_template.tif

Keep median_filters.py beside this script for its calibrated TIFF reader.
Only placements where the entire template fits inside the image are evaluated.
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.signal import correlate
import tifffile

from median_filters import read_image


def cross_correlation(image, template):
    """Each map pixel gives the score for that template top-left position."""
    if image.ndim != 2 or template.ndim != 2:
        raise ValueError("Image and template must both be 2D grayscale arrays.")
    if not image.size or not template.size:
        raise ValueError("Image and template must not be empty.")
    if any(t > n for t, n in zip(template.shape, image.shape)):
        raise ValueError("The template must fit inside the input image.")
    if not np.isfinite(image).all() or not np.isfinite(template).all():
        raise ValueError("Image and template must contain only finite values.")
    # Multiply corresponding pixels and sum at each possible placement.
    # Float64 avoids integer overflow; direct calculation avoids FFT roundoff.
    # 'valid' excludes partial overlaps and zero padding around the image.
    return correlate(image.astype(np.float64), template.astype(np.float64),
                     mode="valid", method="direct")


def main():
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path,
                        default=repo / "testImages/Mito_Cropped.tif")
    parser.add_argument("template", nargs="?", type=Path,
                        default=repo / "testImages/Mito_template.tif")
    args = parser.parse_args()

    image, save_options, scale, unit = read_image(args.input)
    template, _, template_scale, template_unit = read_image(args.template)
    if unit != template_unit or not np.allclose(scale, template_scale):
        raise ValueError("Image and template must have matching pixel sizes and units.")
    scores = cross_correlation(image, template)
    row, col = np.unravel_index(np.argmax(scores), scores.shape)
    height, width = template.shape
    center_row = row + (height - 1) / 2
    center_col = col + (width - 1) / 2

    output = args.input.with_name(f"{args.input.stem}_xcorr_{args.template.stem}.tif")
    # A one-pixel move in this map is a one-pixel move in the source image.
    # Retain pixel sizes; each map coordinate refers to the template TOP LEFT.
    # Float32 is ImageJ compatible. Locate the maximum above in float64 first.
    tifffile.imwrite(output, scores.astype(np.float32), **save_options)
    print(f"Saved correlation map: {output}")
    print(f"Map shape: {scores.shape} (rows, columns); raw, unnormalized scores")
    print(f"Highest cross-correlation (float64): {scores[row, col]:.17g}")
    print(f"Template top-left: row={row}, column={col} (x={col}, y={row})")
    print(f"Template center: row={center_row:g}, column={center_col:g} "
          f"(x={center_col:g}, y={center_row:g})")
    print(f"Matched region: image[{row}:{row + height}, {col}:{col + width}]")
    print("All coordinates are zero-based pixels in the original image; slice stops are excluded.")
    ties = int(np.count_nonzero(scores == scores[row, col]))
    if ties > 1:
        print(f"There are {ties} equal maxima; reporting the first in row-major order.")
    print("Raw correlation favors bright regions; the highest score need not be the best shape match.")


if __name__ == "__main__":
    main()
