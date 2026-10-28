"""Take the DAPI (nucleus) channel out of Cell_cropped.tif and add random pixel noise.

Run it from the top folder of the workshop repository with:

    uv run src/make_noisy_nucleus.py

The result is saved as testImages/Nucleus_cropped_noise.tif.
"""

from pathlib import Path

import numpy as np
import tifffile

REPO = Path(__file__).resolve().parent.parent
INPUT = REPO / "testImages" / "Cell_cropped.tif"
OUTPUT = REPO / "testImages" / "Nucleus_cropped_noise.tif"

# Cell_cropped.tif holds 3 channels.  Python counts from zero, so the third
# channel, the DAPI channel showing the nucleus, has index 2.
DAPI_INDEX = 2

# The fraction of pixels that get hit by noise (0.05 = 5 percent).
NOISE_FRACTION = 0.02

# A fixed "seed" makes the random numbers the same every time you run this, so
# everyone in the class gets the same noisy image.  Change it to get other noise.
rng = np.random.default_rng(seed=42)

with tifffile.TiffFile(INPUT) as tif:
    stack = tif.asarray()  # shape (3, rows, columns)
    # The pixel size is stored in the file as "pixels per micron" in the
    # XResolution and YResolution tags.  Keep it, so that the new image has the
    # same size in microns as the original (napari uses it to scale the image).
    page = tif.pages[0]
    resolution = (page.tags["XResolution"].value, page.tags["YResolution"].value)
    unit = tif.imagej_metadata.get("unit", "micron")
nucleus = stack[DAPI_INDEX]
max_value = nucleus.max()

# Pick random pixels, and give each of them a random value between 0 and the
# brightest value in the image.  The other pixels keep their original value.
noisy = nucleus.copy()
hit = rng.random(nucleus.shape) < NOISE_FRACTION
noisy[hit] = rng.integers(0, max_value, size=hit.sum(), endpoint=True)

# Save as an ImageJ-style tiff, with the pixel size and its unit.
tifffile.imwrite(OUTPUT, noisy, imagej=True, resolution=resolution,
                 metadata={"unit": unit})

print(f"DAPI channel: {nucleus.shape} pixels, type {nucleus.dtype}, "
      f"values {nucleus.min()} - {max_value}")
print(f"Replaced {hit.sum()} of {nucleus.size} pixels ({hit.mean():.1%}) "
      f"with random values between 0 and {max_value}")
print(f"Saved {OUTPUT}")
