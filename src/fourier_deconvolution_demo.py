"""Undo 3 x 3 smoothing by division in Fourier space.

Run from the repository folder:
    uv run src/fourier_deconvolution_demo.py

Saves a comparison figure to Pictures/fourier2d/deconvolution_theorem.png.
The figure is an illustration, not a calibrated microscopy image export.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from scipy import ndimage

REPO = Path(__file__).resolve().parent.parent
INPUT = REPO / "testImages" / "Nucleus_cropped_noise.tif"
OUTPUT = REPO / "Pictures" / "fourier2d" / "deconvolution_theorem.png"
KERNEL = np.ones((3, 3)) / 9


def kernel_as_image(kernel, shape):
    """Pad the kernel and move its center to (0, 0) for the Fourier transform."""
    padded = np.zeros(shape)
    rows, cols = kernel.shape
    padded[:rows, :cols] = kernel
    return np.roll(padded, (-(rows // 2), -(cols // 2)), axis=(0, 1))


def recover(smoothed, kernel_f):
    """Divide complex Fourier values, skipping numerically zero divisors."""
    smoothed_f = np.fft.fft2(smoothed)
    usable = np.abs(kernel_f) > 1e-12 * np.abs(kernel_f).max()
    recovered_f = np.zeros_like(smoothed_f)
    # Use the full complex values (amplitude AND phase), not displayed amplitudes.
    np.divide(smoothed_f, kernel_f, out=recovered_f, where=usable)
    return np.fft.ifft2(recovered_f).real, recovered_f, usable


image = tifffile.imread(INPUT).astype(float)
if image.ndim != 2 or min(image.shape) < 3:
    raise ValueError("This demonstration needs a 2D image at least 3 x 3 pixels.")
kernel_image = kernel_as_image(KERNEL, image.shape)
kernel_f = np.fft.fft2(kernel_image)
# Wrap boundaries match the repeating-image assumption of the Fourier transform.
# Keep floating-point values: rounding or clipping would lose information.
smoothed = ndimage.convolve(image, KERNEL, mode="wrap")
recovered, recovered_f, usable = recover(smoothed, kernel_f)

missing = np.count_nonzero(~usable)
print(f"Unrecoverable Fourier components (set to zero): {missing} / {usable.size}")
print(f"Largest reconstruction error: {np.max(np.abs(recovered - image)):.6g}")
print(f"Root mean square error: {np.sqrt(np.mean((recovered - image)**2)):.6g}")
print("Reconstruction restores the original noise too; this is not denoising.")
print("Division by small kernel values amplifies rounding errors or noise added after smoothing.")
if missing:
    print("The kernel removed some frequencies, so exact recovery is impossible.")

# --- The picture --------------------------------------------------------------
# Shift and log-scale amplitudes only for display, after doing the division.
images = [image, smoothed, recovered]
spectra = [np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(im)))) for im in images]
image_scale = dict(vmin=min(im.min() for im in images),
                   vmax=max(im.max() for im in images))
spectrum_scale = dict(vmin=min(im.min() for im in spectra),
                      vmax=max(im.max() for im in spectra))
# A small padded view makes the 3 x 3 kernel easy to see.
kernel_zoom = np.pad(KERNEL, 4)
fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
panels = [
    (axes[0, 0], image, "Original noisy image", image_scale),
    (axes[0, 1], smoothed, "Smoothed image", image_scale),
    (axes[0, 2], kernel_zoom, "3 x 3 Smooth kernel (ones / 9)", {}),
    (axes[0, 3], recovered, "Reconstructed image", image_scale),
    (axes[1, 0], spectra[0], "Original Fourier amplitudes (log)", spectrum_scale),
    (axes[1, 1], spectra[1], "Smoothed Fourier amplitudes (log)", spectrum_scale),
    (axes[1, 2], np.abs(np.fft.fftshift(kernel_f)),
     "Kernel Fourier amplitudes (linear)", dict(vmin=0, vmax=1)),
    (axes[1, 3], spectra[2], "Reconstructed Fourier amplitudes (log)", spectrum_scale),
]
for ax, data, title, scale in panels:
    ax.imshow(data, cmap="gray", interpolation="nearest", **scale)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
fig.suptitle("Reconstruction = inverse Fourier transform of [FFT(smoothed) / FFT(kernel)]"
             f"\n{missing} unavailable Fourier components set to zero", fontsize=13)
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPUT, dpi=120)
plt.close(fig)
print(f"Saved {OUTPUT}")
