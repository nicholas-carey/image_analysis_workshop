"""Compare Sobel gradients with and without a 5x5 Gaussian pre-blur.

Run from the project folder:
    uv run src/sobel_gaussian_actin.py
    uv run src/sobel_gaussian_actin.py --sigma 1.0

Keep sobel_actin.py and median_filters.py beside this script. All outputs are
saved beside the input TIFF with its spatial calibration. --no-view skips GUI.
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
import tifffile

from median_filters import read_image
from sobel_actin import sobel_images


def positive_sigma(value):
    sigma = float(value)
    if not np.isfinite(sigma) or sigma <= 0:
        raise argparse.ArgumentTypeError("Sigma must be finite and greater than zero.")
    return sigma


def compare_gradients(original, sigma=1.0):
    """Blur first, then calculate Sobel responses using the previous script."""
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("Sigma must be finite and greater than zero.")
    # Radius 2 fixes the Gaussian kernel to exactly 5x5. Sigma sets its spread.
    # Reflect the boundaries, matching the Sobel script's edge handling.
    blurred = gaussian_filter(original.astype(np.float64), sigma=sigma,
                              radius=2, mode="reflect")
    after_blur = sobel_images(blurred)
    without_blur = sobel_images(original)
    return {
        "gaussian_5x5": blurred.astype(np.float32),
        "sobel_x_after_blur": after_blur["sobel_x"],
        "sobel_y_after_blur": after_blur["sobel_y"],
        "magnitude_after_blur": after_blur["sobel_magnitude"],
        "magnitude_without_blur": without_blur["sobel_magnitude"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path,
                        default=Path(__file__).resolve().parents[1] / "testImages/Actin_cropped.tif")
    parser.add_argument("--sigma", type=positive_sigma, default=1.0,
                        help="Gaussian sigma in pixels (default: 1; kernel stays 5x5)")
    parser.add_argument("--no-view", action="store_true", help="Save without opening napari")
    args = parser.parse_args()

    original, save_options, scale, unit = read_image(args.input)
    results = compare_gradients(original, args.sigma)
    for name, data in results.items():
        output = args.input.with_name(
            f"{args.input.stem}_preblur5x5_sigma_{args.sigma}_{name}.tif")
        tifffile.imwrite(output, data, **save_options)
        print(f"Saved {output}")

    if not args.no_view:
        import napari

        viewer = napari.Viewer(title=f"Sobel with/without Gaussian pre-blur — scale in {unit}")
        intensity_limits = (float(original.min()), float(original.max()))
        if intensity_limits[0] == intensity_limits[1]:
            intensity_limits = (intensity_limits[0], intensity_limits[0] + 1)
        magnitude_max = max(float(results[key].max()) for key in
                            ("magnitude_without_blur", "magnitude_after_blur")) or 1.0
        amplitude = max(float(np.abs(results[key]).max()) for key in
                        ("sobel_x_after_blur", "sobel_y_after_blur")) or 1.0
        # Both magnitudes share a fixed display range for a fair comparison.
        layers = [
            ("Original", original, intensity_limits),
            (f"Gaussian 5x5 (sigma {args.sigma:g})", results["gaussian_5x5"], intensity_limits),
            ("Magnitude WITHOUT pre-blur", results["magnitude_without_blur"], (0, magnitude_max)),
            ("Magnitude WITH pre-blur", results["magnitude_after_blur"], (0, magnitude_max)),
            ("Sobel X after pre-blur", results["sobel_x_after_blur"], (-amplitude, amplitude)),
            ("Sobel Y after pre-blur", results["sobel_y_after_blur"], (-amplitude, amplitude)),
        ]
        for name, data, limits in layers:
            viewer.add_image(data, name=name, colormap="gray", scale=scale,
                             contrast_limits=limits)
        viewer.grid.enabled = True
        viewer.grid.shape = (3, 2)
        viewer.reset_view()
        napari.run()


if __name__ == "__main__":
    main()
