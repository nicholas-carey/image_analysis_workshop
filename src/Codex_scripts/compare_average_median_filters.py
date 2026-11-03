"""Compare normalized averaging kernels and median filters on a grayscale TIFF.

Run from the project folder:
    uv run src/Codex_scripts/compare_average_median_filters.py
    uv run src/Codex_scripts/compare_average_median_filters.py testImages/Nucleus_cropped_noise.tif

All six filtered images are saved beside the input (or in --output-dir).
Use --no-view to save without launching napari. Keep median_filters.py beside
this script: it supplies the median footprints and pixel-size metadata reader.
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.ndimage import convolve, median_filter
import tifffile

from median_filters import FOOTPRINTS, read_image


def make_kernels():
    """Divide each kernel by its total so that constant brightness is preserved."""
    center_weighted = np.ones((3, 3), dtype=np.float64)
    center_weighted[1, 1] = 4
    kernels = {
        "mean_3x3": np.ones((3, 3), dtype=np.float64),
        "weighted_mean_3x3_center4": center_weighted,
        "mean_7x7": np.ones((7, 7), dtype=np.float64),
    }
    return {name: kernel / kernel.sum() for name, kernel in kernels.items()}


def filter_images(image):
    """Apply every filter independently to the original image."""
    results = {}
    # Floating point keeps fractional averages instead of rounding to integers.
    # Calculate in float64, then save as ImageJ-compatible float32 TIFFs.
    floating_image = image.astype(np.float64)
    for name, kernel in make_kernels().items():
        results[name] = convolve(floating_image, kernel, mode="reflect").astype(np.float32)

    # Recreate the previous script's results, so it need not be run beforehand.
    # Both filter families reflect the image at its edges.
    for name, footprint in FOOTPRINTS.items():
        results[f"median_{name}"] = median_filter(image, footprint=footprint, mode="reflect")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path,
                        default=Path(__file__).resolve().parents[2] / "testImages/Nucleus_cropped_noise.tif")
    parser.add_argument("--output-dir", type=Path, help="Default: input image folder")
    parser.add_argument("--no-view", action="store_true", help="Save without opening napari")
    args = parser.parse_args()

    image, save_options, scale, unit = read_image(args.input)
    results = filter_images(image)
    output_dir = args.output_dir or args.input.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, filtered in results.items():
        output = output_dir / f"{args.input.stem}_{name}.tif"
        # Filtering changes intensities, but not pixel sizes or image dimensions.
        tifffile.imwrite(output, filtered, **save_options)
        print(f"Saved {output}")

    if not args.no_view:
        import napari

        viewer = napari.Viewer(title=f"Average and median filters — scale in {unit}")
        limits = (float(image.min()), float(image.max()))
        if limits[0] == limits[1]:
            limits = (limits[0], limits[0] + 1)
        for name, data in {"Original": image, **results}.items():
            viewer.add_image(data, name=name, colormap="gray", scale=scale,
                             contrast_limits=limits)
        viewer.grid.enabled = True
        viewer.grid.shape = (2, 4)
        viewer.reset_view()
        napari.run()


if __name__ == "__main__":
    main()
