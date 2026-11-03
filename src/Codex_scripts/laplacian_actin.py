"""Compare an eight-neighbor Laplacian with Gaussian unsharp detail masks.

Run from the project folder:
    uv run src/Codex_scripts/laplacian_actin.py
    uv run src/Codex_scripts/laplacian_actin.py --sigma-5 1 --sigma-21 5

Keep median_filters.py and gaussian_unsharp_masks.py beside this script.
Here, unsharp masks mean original minus blur, before adding detail back.
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.ndimage import convolve
import tifffile

from median_filters import read_image
from gaussian_unsharp_masks import filter_images, positive_sigma


LAPLACIAN = np.array([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1],
], dtype=np.float64)


def compare_filters(original, sigma_5=1.0, sigma_21=5.0):
    """Compute the exact requested kernel and the previous script's masks."""
    # The kernel sums to zero: uniform regions give a zero response.
    # Floating point retains both positive and negative detail values.
    laplacian = convolve(original.astype(np.float64), LAPLACIAN, mode="reflect")
    gaussian_results = filter_images(original, sigma_5, sigma_21)
    return {
        "laplacian_center8": laplacian.astype(np.float32),
        "unsharp_masked_5x5": gaussian_results["unsharp_masked_5x5"],
        "unsharp_masked_21x21": gaussian_results["unsharp_masked_21x21"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path,
                        default=Path(__file__).resolve().parents[2] / "testImages/Actin_cropped.tif")
    parser.add_argument("--sigma-5", type=positive_sigma, default=1.0,
                        help="5x5 Gaussian sigma in pixels (default: 1)")
    parser.add_argument("--sigma-21", type=positive_sigma, default=5.0,
                        help="21x21 Gaussian sigma in pixels (default: 5)")
    parser.add_argument("--no-view", action="store_true", help="Save without opening napari")
    args = parser.parse_args()

    original, save_options, scale, unit = read_image(args.input)
    results = compare_filters(original, args.sigma_5, args.sigma_21)
    # Recompute masks from this input, so no earlier script needs to run first.
    # Distinct filenames keep the earlier comparison's outputs intact.
    suffixes = {
        "laplacian_center8": "laplacian_center8",
        "unsharp_masked_5x5": f"laplacian_comparison_unsharp_masked_5x5_sigma_{args.sigma_5}",
        "unsharp_masked_21x21": f"laplacian_comparison_unsharp_masked_21x21_sigma_{args.sigma_21}",
    }
    for name, data in results.items():
        output = args.input.with_name(f"{args.input.stem}_{suffixes[name]}.tif")
        tifffile.imwrite(output, data, **save_options)
        print(f"Saved {output}")

    if not args.no_view:
        import napari

        viewer = napari.Viewer(title=f"Laplacian and unsharp detail masks — scale in {unit}")
        limits = (float(original.min()), float(original.max()))
        if limits[0] == limits[1]:
            limits = (limits[0], limits[0] + 1)
        viewer.add_image(original, name="Original", colormap="gray", scale=scale,
                         contrast_limits=limits)
        # Shared symmetric limits preserve relative response strengths.
        # Zero is gray; positive values are bright and negative values dark.
        # The unnormalized Laplacian can be much stronger than the detail masks.
        amplitude = max(float(np.abs(data).max()) for data in results.values()) or 1.0
        labels = {
            "laplacian_center8": "Laplacian (center 8, neighbors -1)",
            "unsharp_masked_5x5": f"Original minus 5x5 blur (sigma {args.sigma_5:g})",
            "unsharp_masked_21x21": f"Original minus 21x21 blur (sigma {args.sigma_21:g})",
        }
        for name, data in results.items():
            viewer.add_image(data, name=labels[name], colormap="gray", scale=scale,
                             contrast_limits=(-amplitude, amplitude))
        viewer.grid.enabled = True
        viewer.grid.shape = (2, 2)
        viewer.reset_view()
        napari.run()


if __name__ == "__main__":
    main()
