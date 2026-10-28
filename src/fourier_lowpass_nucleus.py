"""Remove noise by keeping only the slow waves: a "low-pass filter" in Fourier space.

Run it from the top folder of the workshop repository with:

    uv run src/fourier_lowpass_nucleus.py

The script calculates the Fourier image of Nucleus_cropped_noise.tif, makes
everything outside a circle in the center black (0), and transforms back.
Two images are saved next to the original in testImages/:

    Nucleus_cropped_noise_fourier_circle.tif  the Fourier image (amplitudes, log)
                                              after blacking out outside the circle
    Nucleus_cropped_noise_lowpass.tif         the image transformed back
"""

from pathlib import Path

import numpy as np
import tifffile

REPO = Path(__file__).resolve().parent.parent
INPUT = REPO / "testImages" / "Nucleus_cropped_noise.tif"

# The diameter of the circle, as a fraction of the image size (0.1 = 10%).
# Try other values: a smaller circle removes more noise, but blurs more.
DIAMETER_FRACTION = 0.1

with tifffile.TiffFile(INPUT) as tif:
    image = tif.asarray().astype(float)
    # Keep the pixel size, so the filtered image has the same size in microns.
    page = tif.pages[0]
    resolution = (page.tags["XResolution"].value, page.tags["YResolution"].value)
    unit = tif.imagej_metadata.get("unit", "micron")

rows, cols = image.shape
print(f"Read {INPUT.name}: {rows} x {cols} pixels")

# The Fourier image, with the zero frequency (the average brightness) moved to
# the center, so that the slow waves are in the middle and the fast waves
# (fine details and noise) further out.
fourier = np.fft.fftshift(np.fft.fft2(image))

# Make the circle: for every pixel of the Fourier image, calculate its distance
# to the center, and keep it if that distance is less than the radius.
# The image is not exactly square, so use the smaller side for the size.
radius = DIAMETER_FRACTION * min(rows, cols) / 2
row_distance, col_distance = np.ogrid[:rows, :cols]
distance = np.hypot(row_distance - rows // 2, col_distance - cols // 2)
inside = distance <= radius
print(f"Circle with a diameter of {2 * radius:.1f} pixels: keeps {inside.sum()} of "
      f"{inside.size} waves ({100 * inside.mean():.2f}%)")

# Black out everything outside the circle: set those waves to 0.
filtered_fourier = np.where(inside, fourier, 0)

# Transform back.  ifftshift undoes the fftshift from above.  The result has
# tiny imaginary parts left over from rounding; "real" drops them.
filtered = np.real(np.fft.ifft2(np.fft.ifftshift(filtered_fourier)))
print(f"Pixel values: original {image.min():.0f} - {image.max():.0f}, "
      f"filtered {filtered.min():.0f} - {filtered.max():.0f}")

# The Fourier image to look at: the amplitudes, on a log scale (otherwise only
# the very bright center is visible).  Pixels in Fourier space are frequencies,
# not distances, so the pixel size of the image does not apply to this one.
fourier_view = np.log1p(np.abs(filtered_fourier))
output = INPUT.with_name(f"{INPUT.stem}_fourier_circle.tif")
tifffile.imwrite(output, fourier_view.astype(np.float32), imagej=True)
print(f"Saved {output.name}")

# Saved as a 32-bit float image: the filtered values have fractions, and can be
# slightly negative near sharp edges.
output = INPUT.with_name(f"{INPUT.stem}_lowpass.tif")
tifffile.imwrite(output, filtered.astype(np.float32), imagej=True,
                 resolution=resolution, metadata={"unit": unit})
print(f"Saved {output.name}")
