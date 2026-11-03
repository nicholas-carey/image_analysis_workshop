"""Create a randomly shifted TIFF, then recover its movement by cross-correlation.

Run from the project folder:
    uv run src/shift_and_register_mito.py
    uv run src/shift_and_register_mito.py --seed 42

To estimate movement from an existing shifted TIFF without creating a new one:
    uv run src/shift_and_register_mito.py --shifted testImages/Mito_Cropped_shifted.tif

Keep median_filters.py beside this script for the calibrated TIFF reader.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import correlate
import tifffile

from median_filters import read_image


def shift_image(image, dy, dx):
    """Translate whole pixels without wrapping; fill exposed edges with zeros."""
    if image.ndim != 2:
        raise ValueError("A 2D grayscale image is required.")
    if int(dy) != dy or int(dx) != dx:
        raise ValueError("Shifts must be whole numbers of pixels.")
    dy, dx = int(dy), int(dx)
    height, width = image.shape
    if abs(dy) >= height or abs(dx) >= width:
        raise ValueError("The shift must leave some image overlap.")
    shifted = np.zeros_like(image)
    source_rows = slice(max(0, -dy), min(height, height - dy))
    source_cols = slice(max(0, -dx), min(width, width - dx))
    target_rows = slice(max(0, dy), min(height, height + dy))
    target_cols = slice(max(0, dx), min(width, width + dx))
    shifted[target_rows, target_cols] = image[source_rows, source_cols]
    return shifted


def estimate_shift(original, shifted):
    """Estimate displacement using only the two arrays, not the applied shift."""
    if original.ndim != 2 or shifted.shape != original.shape or not original.size:
        raise ValueError("The two images must be nonempty 2D arrays of the same shape.")
    if not np.isfinite(original).all() or not np.isfinite(shifted).all():
        raise ValueError("Both images must contain finite values.")
    reference = original.astype(np.float64)
    moving = shifted.astype(np.float64)
    if np.ptp(reference) == 0 or np.ptp(moving) == 0:
        raise ValueError("Both images need intensity variation to estimate movement.")
    # Remove mean brightness so structure dominates the correlation.
    reference -= reference.mean()
    moving -= moving.mean()
    # Full LINEAR correlation considers negative and positive displacements.
    # FFT accelerates the calculation; mode='full' avoids circular wraparound.
    scores = correlate(moving, reference, mode="full", method="fft")
    peak = np.asarray(np.unravel_index(np.argmax(scores), scores.shape))
    zero_lag = np.asarray(original.shape) - 1
    dy, dx = (int(value) for value in peak - zero_lag)
    return dy, dx, scores


def main():
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path,
                        default=repo / "testImages/Mito_Cropped.tif")
    parser.add_argument("--seed", type=int, help="Optional random seed for reproducible shifts")
    parser.add_argument("--shifted", type=Path, help="Analyze an existing shifted image instead")
    args = parser.parse_args()
    original, save_options, scale, unit = read_image(args.input)
    applied = None

    if args.shifted is None:
        # Uniform integer distributions centered on 7 down and 4 right.
        rng = np.random.default_rng(args.seed)
        applied = (int(rng.integers(5, 10)), int(rng.integers(2, 7)))
        shifted = shift_image(original, *applied)
        shifted_path = args.input.with_name(f"{args.input.stem}_shifted.tif")
        tifffile.imwrite(shifted_path, shifted, **save_options)
        print(f"Saved shifted image: {shifted_path}")
    else:
        shifted_path = args.shifted
        shifted, _, shifted_scale, shifted_unit = read_image(shifted_path)
        if unit != shifted_unit or not np.allclose(scale, shifted_scale):
            raise ValueError("Original and shifted images must have matching pixel sizes and units.")

    # The registration function does not receive the known displacement.
    dy, dx, scores = estimate_shift(original, shifted)
    map_path = shifted_path.with_name(f"{shifted_path.stem}_registration_xcorr.tif")
    report_path = shifted_path.with_name(f"{shifted_path.stem}_registration.json")
    # Each correlation-map pixel represents a one-pixel displacement step.
    # Spatial spacing is preserved; its zero-displacement index is recorded below.
    tifffile.imwrite(map_path, scores.astype(np.float32), **save_options)
    report = {
        "original": str(args.input),
        "shifted": str(shifted_path),
        "convention": "(dy, dx); positive means down, right",
        "applied_shift_pixels": applied,
        "estimated_shift_pixels": [dy, dx],
        "alignment_correction_pixels": [-dy, -dx],
        "correlation_zero_lag_row_column": [int(n - 1) for n in original.shape],
        "estimated_displacement_physical_y_x": [dy * scale[0], dx * scale[1]],
        "spatial_unit": unit,
        "method": "Full linear cross-correlation after subtracting each image mean",
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print("Convention: positive dy = down; positive dx = right.")
    if applied is not None:
        print(f"Applied movement:   dy={applied[0]}, dx={applied[1]} pixels")
    print(f"Estimated movement: dy={dy}, dx={dx} pixels")
    print(f"To align shifted to original: dy={-dy}, dx={-dx} pixels")
    if applied is not None:
        print(f"Estimation error: dy={dy - applied[0]}, dx={dx - applied[1]} pixels")
    print(f"Saved correlation map: {map_path}")
    print(f"Saved coordinates and report: {report_path}")


if __name__ == "__main__":
    main()
