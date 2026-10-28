"""Cut a small piece (a "template") out of Mito_Cropped.tif for cross-correlation.

Run it from the top folder of the workshop repository with:

    uv run src/make_mito_template.py

The result is saved as testImages/Mito_template.tif.
"""

from pathlib import Path

import tifffile

REPO = Path(__file__).resolve().parent.parent
INPUT = REPO / "testImages" / "Mito_Cropped.tif"
OUTPUT = REPO / "testImages" / "Mito_template.tif"

# The center of the piece (row, column) and how far it extends on each side.
# This spot is a single, small, round mitochondrion below and to the right of
# the nucleus.  A half-width of 5 gives an 11 x 11 pixel piece.
CENTER_ROW, CENTER_COL = 225, 172
HALF_WIDTH = 5

with tifffile.TiffFile(INPUT) as tif:
    image = tif.asarray()
    # Keep the pixel size, so the template has the same size in microns.
    page = tif.pages[0]
    resolution = (page.tags["XResolution"].value, page.tags["YResolution"].value)
    unit = tif.imagej_metadata.get("unit", "micron")

# "start:stop" picks rows (or columns) start up to, but not including, stop.
top, left = CENTER_ROW - HALF_WIDTH, CENTER_COL - HALF_WIDTH
template = image[top:CENTER_ROW + HALF_WIDTH + 1, left:CENTER_COL + HALF_WIDTH + 1]

tifffile.imwrite(OUTPUT, template, imagej=True, resolution=resolution,
                 metadata={"unit": unit})

print(f"Cut rows {top} - {top + template.shape[0] - 1}, "
      f"columns {left} - {left + template.shape[1] - 1} "
      f"({template.shape[0]} x {template.shape[1]} pixels)")
print(f"Saved {OUTPUT}")
