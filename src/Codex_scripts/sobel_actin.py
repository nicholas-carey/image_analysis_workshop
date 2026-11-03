"""Apply Sobel X/Y kernels and display their gradient magnitude in napari.

Run from the project folder:
    uv run src/Codex_scripts/sobel_actin.py
    uv run src/Codex_scripts/sobel_actin.py testImages/Actin_cropped.tif

Use --no-view to save without opening napari. Keep median_filters.py beside
this script; it supplies the TIFF reader that preserves pixel-size metadata.
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.ndimage import correlate
import tifffile

from median_filters import read_image


# X increases toward the right; Y increases downward (image row coordinates).
# Apply these weights directly to each neighborhood using correlation.
# Convolution would flip the kernels and reverse the signs of the responses.
SOBEL_X = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1],
], dtype=np.float64)
SOBEL_Y = SOBEL_X.T.copy()


def sobel_images(image):
    """Return signed X/Y responses and their nonnegative combined strength."""
    # Float calculations avoid unsigned overflow and retain negative responses.
    floating = image.astype(np.float64)
    gx = correlate(floating, SOBEL_X, mode="reflect")
    gy = correlate(floating, SOBEL_Y, mode="reflect")
    # These are raw, unnormalized Sobel responses, not physical-unit derivatives.
    magnitude = np.hypot(gx, gy)  # sqrt(gx**2 + gy**2)
    return {
        "sobel_x": gx.astype(np.float32),
        "sobel_y": gy.astype(np.float32),
        "sobel_magnitude": magnitude.astype(np.float32),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path,
                        default=Path(__file__).resolve().parents[2] / "testImages/Actin_cropped.tif")
    parser.add_argument("--no-view", action="store_true", help="Save without opening napari")
    args = parser.parse_args()

    original, save_options, scale, unit = read_image(args.input)
    results = sobel_images(original)
    for name, data in results.items():
        output = args.input.with_name(f"{args.input.stem}_{name}.tif")
        tifffile.imwrite(output, data, **save_options)
        print(f"Saved {output}")

    if not args.no_view:
        import napari

        viewer = napari.Viewer(title=f"Sobel comparison — spatial scale in {unit}")
        original_limits = (float(original.min()), float(original.max()))
        if original_limits[0] == original_limits[1]:
            original_limits = (original_limits[0], original_limits[0] + 1)
        viewer.add_image(original, name="Original", colormap="gray", scale=scale,
                         contrast_limits=original_limits)
        # Shared symmetric limits: zero is gray, positive is bright, negative dark.
        amplitude = max(float(np.max(np.abs(results[key])))
                        for key in ("sobel_x", "sobel_y")) or 1.0
        for key, label in (("sobel_x", "Sobel X (left-to-right change)"),
                           ("sobel_y", "Sobel Y (top-to-bottom change)")):
            viewer.add_image(results[key], name=label, colormap="gray", scale=scale,
                             contrast_limits=(-amplitude, amplitude))
        viewer.add_image(results["sobel_magnitude"], name="Gradient magnitude",
                         colormap="gray", scale=scale,
                         contrast_limits=(0, float(results["sobel_magnitude"].max()) or 1.0))
        viewer.grid.enabled = True
        viewer.grid.shape = (2, 2)
        viewer.reset_view()
        napari.run()


if __name__ == "__main__":
    main()
