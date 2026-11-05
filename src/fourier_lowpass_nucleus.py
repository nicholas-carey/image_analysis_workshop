"""Remove noise by keeping only the slow waves: a "low-pass filter" in Fourier space.

Run it from the top folder of the workshop repository with:

    uv run src/fourier_lowpass_nucleus.py

The script calculates the Fourier image of Nucleus_cropped_noise.tif,
multiplies it by a Gaussian mask (brightest at the center, fading smoothly
outward), and transforms back.
Two images are saved next to the original in testImages/:

    Nucleus_cropped_noise_fourier_gaussian.tif  the filtered Fourier image
                                                (amplitudes, log)
    Nucleus_cropped_noise_lowpass.tif         the image transformed back
"""

from pathlib import Path

import numpy as np
import tifffile

REPO = Path(__file__).resolve().parent.parent
INPUT = REPO / "testImages" / "Nucleus_cropped_noise.tif"

# Gaussian width (sigma) as a fraction of the shorter image side.
# A smaller sigma suppresses more high frequencies, but blurs more.
# 0.2 uses the old circle radius as sigma; the filters are not equivalent.
SIGMA_FRACTION = 0.2

with tifffile.TiffFile(INPUT) as tif:
    image = tif.asarray().astype(float)
    # Keep the pixel size, so the filtered image has the same size in microns.
    page = tif.pages[0]
    resolution = (page.tags["XResolution"].value, page.tags["YResolution"].value)
    unit = (tif.imagej_metadata or {}).get("unit", "micron")
    resolution_unit = page.tags["ResolutionUnit"].value

rows, cols = image.shape
print(f"Read {INPUT.name}: {rows} x {cols} pixels")

# The Fourier image, with the zero frequency (the average brightness) moved to
# the center, so that the slow waves are in the middle and the fast waves
# (fine details and noise) further out.
fourier = np.fft.fftshift(np.fft.fft2(image))

# Make a 2D bell curve centered on the zero frequency. Its peak is 1,
# preserving the average brightness; weights fade toward 0 further out.
# Sigma is measured in Fourier-image pixels, not microns.
if not np.isfinite(SIGMA_FRACTION) or SIGMA_FRACTION <= 0:
    raise ValueError("SIGMA_FRACTION must be a positive, finite number.")
sigma = SIGMA_FRACTION * min(rows, cols)
row_distance, col_distance = np.ogrid[:rows, :cols]
distance_squared = (row_distance - rows // 2)**2 + (col_distance - cols // 2)**2
mask = np.exp(-distance_squared / (2 * sigma**2))
print(f"Gaussian mask with sigma = {sigma:.1f} Fourier-image pixels")

# Multiply the full complex Fourier values by the smoothly varying weights.
filtered_fourier = fourier * mask

# Transform back.  ifftshift undoes the fftshift from above.  The result has
# tiny imaginary parts left over from rounding; "real" drops them.
filtered = np.real(np.fft.ifft2(np.fft.ifftshift(filtered_fourier)))
print(f"Pixel values: original {image.min():.0f} - {image.max():.0f}, "
      f"filtered {filtered.min():.0f} - {filtered.max():.0f}")

# The Fourier image to look at: the amplitudes, on a log scale (otherwise only
# the very bright center is visible).  Pixels in Fourier space are frequencies,
# not distances, so the pixel size of the image does not apply to this one.
fourier_view = np.log1p(np.abs(filtered_fourier))
output = INPUT.with_name(f"{INPUT.stem}_fourier_gaussian.tif")
tifffile.imwrite(output, fourier_view.astype(np.float32), imagej=True)
print(f"Saved {output.name}")

# Saved as a 32-bit float image: the filtered values have fractions, and can be
# slightly negative near sharp edges.
output = INPUT.with_name(f"{INPUT.stem}_lowpass.tif")
tifffile.imwrite(output, filtered.astype(np.float32), imagej=True,
                 resolution=resolution, resolutionunit=resolution_unit,
                 metadata={"unit": unit})
print(f"Saved {output.name}")
