"""Find mitochondrial template matches with normalized cross-correlation.

Run from the project folder:
    uv run src/Codex_scripts/normalized_cross_correlate_mito.py
    uv run src/Codex_scripts/normalized_cross_correlate_mito.py --threshold 0.8 --min-distance 4

Keep median_filters.py beside this script. --no-view saves without napari.
The score TIFF uses template top-left coordinates. The CSV includes those
coordinates and template centers in original-image, zero-based pixels.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from skimage.feature import match_template, peak_local_max
import tifffile

from median_filters import read_image


def normalized_matches(image, template, threshold=0.8, min_distance=4):
    """Return a valid-overlap NCC map, peak top-lefts, and template centers."""
    if image.ndim != 2 or template.ndim != 2 or not image.size or not template.size:
        raise ValueError("Image and template must be nonempty 2D grayscale arrays.")
    if any(t > n for t, n in zip(template.shape, image.shape)):
        raise ValueError("The template must fit inside the image.")
    if not np.isfinite(image).all() or not np.isfinite(template).all():
        raise ValueError("Input pixels must be finite.")
    if np.ptp(template.astype(np.float64)) == 0:
        raise ValueError("A constant template has no contrast for normalized correlation.")
    if not np.isfinite(threshold) or not -1 <= threshold <= 1:
        raise ValueError("Threshold must be between -1 and 1.")
    if min_distance < 1 or int(min_distance) != min_distance:
        raise ValueError("Minimum distance must be a positive integer number of pixels.")

    # Removes local mean and divides by contrast: scores approach 1 for matching
    # patterns even if their brightness and contrast differ. No partial overlaps.
    scores = match_template(image.astype(np.float64), template.astype(np.float64),
                            pad_input=False)
    # First identify local maxima in 3x3 neighborhoods, then suppress weaker
    # peaks less than min_distance away using Euclidean distance (p_norm=2).
    peaks = peak_local_max(scores, threshold_abs=threshold,
                           min_distance=min_distance, p_norm=2,
                           footprint=np.ones((3, 3), dtype=bool), exclude_border=False)
    peaks = peaks[scores[tuple(peaks.T)] > threshold]  # Strictly above threshold.
    offset = (np.asarray(template.shape) - 1) / 2
    centers = peaks.astype(float) + offset
    return scores, peaks, centers


def main():
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path,
                        default=repo / "testImages/Mito_Cropped.tif")
    parser.add_argument("template", nargs="?", type=Path,
                        default=repo / "testImages/Mito_template.tif")
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--min-distance", type=int, default=4)
    parser.add_argument("--no-view", action="store_true")
    args = parser.parse_args()
    image, save_options, scale, unit = read_image(args.input)
    template, _, template_scale, template_unit = read_image(args.template)
    if unit != template_unit or not np.allclose(scale, template_scale):
        raise ValueError("Image and template must have matching pixel sizes and units.")
    scores, peaks, centers = normalized_matches(image, template, args.threshold, args.min_distance)

    base = f"{args.input.stem}_ncc_{args.template.stem}"
    score_path = args.input.with_name(f"{base}.tif")
    csv_path = args.input.with_name(
        f"{base}_peaks_gt{args.threshold}_distance{args.min_distance}.csv")
    # Adjacent map pixels remain one source-image pixel apart. Origin denotes
    # template top-left placements, not template centers (as documented above).
    tifffile.imwrite(score_path, scores.astype(np.float32), **save_options)
    with csv_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["top_left_row", "top_left_column", "center_row", "center_column", "ncc_score"])
        for (row, col), (cy, cx) in zip(peaks, centers):
            writer.writerow([row, col, cy, cx, scores[row, col]])
    print(f"Saved score map: {score_path}")
    print(f"Saved {len(peaks)} peaks: {csv_path}")
    print(f"Criteria: score > {args.threshold}, centers at least {args.min_distance} pixels apart (Euclidean).")
    row, col = np.unravel_index(np.argmax(scores), scores.shape)
    offset = (np.asarray(template.shape) - 1) / 2
    print(f"Highest NCC: {scores[row, col]:.8g}; center row={row + offset[0]:g}, "
          f"column={col + offset[1]:g} (zero-based pixels)")
    if not len(peaks):
        print("No peaks meet the criteria; the CSV contains its header only.")

    if not args.no_view:
        import napari

        viewer = napari.Viewer(title=f"Normalized template matching — scale in {unit}")
        point_options = dict(symbol="o", size=max(template.shape), face_color="transparent",
                             border_color="red", border_width=0.12, scale=scale,
                             features={"ncc_score": scores[tuple(peaks.T)]})
        viewer.add_image(image, name="Mito original", colormap="gray", scale=scale)
        viewer.add_points(centers, name="Matches on original", **point_options)
        # Shift the valid score map to template centers so the circles line up.
        viewer.add_image(scores, name="Normalized cross-correlation", colormap="viridis",
                         contrast_limits=(-1, 1), scale=scale,
                         translate=offset * np.asarray(scale))
        viewer.add_points(centers, name="Matches on NCC map", **point_options)
        viewer.add_image(template, name="Template", colormap="gray", scale=scale)
        # Pair each image with its circles; template occupies its own panel.
        viewer.grid.stride = 2
        viewer.grid.shape = (1, 3)
        viewer.grid.enabled = True
        viewer.reset_view()
        napari.run()


if __name__ == "__main__":
    main()
