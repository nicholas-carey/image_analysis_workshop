"""Apply a soft Gaussian high-pass filter and compare images in napari.

Run from the project folder:
    uv run src/fourier_highpass_actin.py
    uv run src/fourier_highpass_actin.py --sigma 0.05 --no-view

Sigma is measured in cycles per pixel. Larger values remove a wider range
of slow waves. This extracts detail rather than adding detail to the original.
Uses the existing metadata reader in src/Codex_scripts/median_filters.py.
"""

import argparse
from pathlib import Path

import numpy as np
import tifffile

from Codex_scripts.median_filters import read_image


def highpass(image, sigma):
    """Multiply the Fourier transform by one minus a Gaussian low-pass mask."""
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("Sigma must be positive and finite.")
    # fftfreq matches the unshifted FFT: zero frequency is at index zero.
    fy = np.fft.fftfreq(image.shape[0])[:, None]
    fx = np.fft.fftfreq(image.shape[1])[None, :]
    lowpass = np.exp(-(fx**2 + fy**2) / (2 * sigma**2))
    mask = 1 - lowpass  # zero at the center frequency, smoothly approaching 1
    # FFT filtering assumes the image repeats at its edges (wrap boundaries).
    return np.fft.ifft2(np.fft.fft2(image.astype(float)) * mask).real.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path,
                        default=Path(__file__).resolve().parent.parent /
                        "testImages/Actin_cropped.tif")
    parser.add_argument("--sigma", type=float, default=0.05,
                        help="Gaussian frequency width in cycles/pixel (default: 0.05)")
    parser.add_argument("--no-view", action="store_true", help="Save without opening napari")
    args = parser.parse_args()
    original, save_options, scale, unit = read_image(args.input)
    filtered = highpass(original, args.sigma)
    output = args.input.with_name(f"{args.input.stem}_highpass_gaussian.tif")
    # Float32 retains negative values. Copy the original spatial calibration.
    tifffile.imwrite(output, filtered, **save_options)
    print(f"Saved {output}", flush=True)
    print(f"High-pass range: {filtered.min():.3g} to {filtered.max():.3g}", flush=True)
    print("The background average is removed; negative values are expected.", flush=True)

    if not args.no_view:
        import napari

        viewer = napari.Viewer(title=f"Actin: Gaussian high-pass comparison (scale in {unit})")
        viewer.add_image(original, name="Original", colormap="gray", scale=scale)
        # Separate display ranges make detail visible despite the changed baseline.
        # In the result, zero is middle gray; dark/bright mean negative/positive.
        limit = max(float(np.abs(filtered).max()), 1e-12)
        viewer.add_image(filtered, name=f"High-pass (sigma {args.sigma:g} cycles/pixel)",
                         colormap="gray", scale=scale, contrast_limits=(-limit, limit))
        viewer.grid.enabled = True
        viewer.grid.shape = (1, 2)
        viewer.reset_view()
        napari.run()


if __name__ == "__main__":
    main()
