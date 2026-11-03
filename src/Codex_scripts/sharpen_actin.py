"""Sharpen a 2D grayscale TIFF and compare it with the original in napari.

Run from the project folder:
    uv run src/Codex_scripts/sharpen_actin.py
    uv run src/Codex_scripts/sharpen_actin.py testImages/Actin_cropped.tif

Keep median_filters.py beside this script; it supplies the metadata reader.
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.ndimage import convolve
import tifffile

from median_filters import read_image


KERNEL = np.array([
    [-1, -1, -1],
    [-1, 12, -1],
    [-1, -1, -1],
], dtype=np.float64)


def sharpen(image):
    """Multiply the center by 12 and subtract its eight neighbors."""
    # Do not normalize: the requested kernel sums to 4, so a uniform region
    # becomes four times brighter. Reflect the image at its boundaries.
    # Floating point preserves negative values and avoids unsigned overflow.
    return convolve(image.astype(np.float64), KERNEL, mode="reflect").astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path,
                        default=Path(__file__).resolve().parents[2] / "testImages/Actin_cropped.tif")
    parser.add_argument("--no-view", action="store_true", help="Save without opening napari")
    args = parser.parse_args()

    original, save_options, scale, unit = read_image(args.input)
    filtered = sharpen(original)
    output = args.input.with_name(f"{args.input.stem}_sharpen_center12.tif")
    # Pixel dimensions and spatial calibration stay unchanged.
    tifffile.imwrite(output, filtered, **save_options)
    print(f"Saved {output}")
    print("Kernel sum = 4: uniform regions become four times brighter.")

    if not args.no_view:
        import napari

        viewer = napari.Viewer(title=f"Actin sharpening comparison — scale in {unit}")
        # A shared range includes negative and enhanced values without clipping.
        limits = (float(min(original.min(), filtered.min())),
                  float(max(original.max(), filtered.max())))
        if limits[0] == limits[1]:
            limits = (limits[0], limits[0] + 1)
        for name, image in (("Original", original), ("Sharpened (center 12, neighbors -1)", filtered)):
            viewer.add_image(image, name=name, colormap="gray", scale=scale,
                             contrast_limits=limits)
        viewer.grid.enabled = True
        viewer.grid.shape = (1, 2)
        viewer.reset_view()
        napari.run()


if __name__ == "__main__":
    main()
