"""Threshold the DAPI channel with Otsu's method, count objects, and plot their sizes.

Steps:
  1. Read the channel named "DAPI" from an OME-TIFF.
  2. Let Otsu's method pick the threshold; pixels above it become foreground.
  3. Find the objects: groups of foreground pixels that touch (including at
     corners, i.e. all 8 neighbors - scikit-image's default).
  4. Count all objects, no matter how small, and show a histogram of their
     sizes in pixels.
  5. Show the original DAPI image next to the labeled objects in napari.
     Close the napari window to end the script.

Run (from the workshop folder):
    uv run src/otsu_count_objects.py testImages/3Channel_FF_Corrected_1/3Channel_FF_Corrected_1_MMStack_Pos0.ome.tif
"""

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib

# napari is built on Qt; telling matplotlib to use Qt too lets the histogram
# window and the napari window stay open at the same time.
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
import napari
import numpy as np
import tifffile
from skimage.filters import threshold_otsu
from skimage.measure import label


def read_dapi(path):
    """Return the 2D DAPI channel and its pixel size (scale, units) in Y, X order."""
    with tifffile.TiffFile(path) as tif:
        pixels = ET.fromstring(tif.ome_metadata).find("{*}Image/{*}Pixels")
        names = [c.get("Name", "") for c in pixels.findall("{*}Channel")]
        if "DAPI" not in names:
            raise SystemExit(f"No channel named DAPI; found channels: {names}")
        series = tif.series[0]
        if series.axes != "CYX":
            raise SystemExit(f"Expected a 2D multichannel image (CYX); found {series.axes}.")
        dapi = series.asarray()[names.index("DAPI")]
        # Pixel size, so napari can show distances in micrometers.
        scale = tuple(float(pixels.get(f"PhysicalSize{a}", 1.0)) for a in "YX")
        units = tuple(pixels.get(f"PhysicalSize{a}Unit", "µm")
                      if f"PhysicalSize{a}" in pixels.attrib else "pixel" for a in "YX")
        return dapi, scale, units


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", type=Path, help="Input OME-TIFF image.")
    args = parser.parse_args()
    if not args.image.is_file():
        parser.error(f"Input file not found: {args.image}")

    dapi, scale, units = read_dapi(args.image)

    # Otsu looks at the histogram and picks the value that best splits the
    # pixels into two groups (each group as uniform as possible).
    threshold = threshold_otsu(dapi)
    mask = dapi > threshold

    # label() gives every connected group of foreground pixels its own number:
    # 0 = background, 1 = first object, 2 = second object, ...
    labels = label(mask)
    n_objects = labels.max()

    # Size of each object = how many pixels carry its number.
    # bincount counts pixels per label; [1:] drops label 0 (the background).
    sizes = np.bincount(labels.ravel())[1:]

    print(f"Otsu threshold: {threshold:g}", flush=True)
    print(f"Foreground: {mask.mean() * 100:.1f}% of pixels", flush=True)
    print(f"Number of objects: {n_objects}", flush=True)
    print(f"Object size (pixels): smallest {sizes.min()}, median {np.median(sizes):g}, "
          f"largest {sizes.max()}", flush=True)
    print(f"Objects of 1 pixel: {(sizes == 1).sum()}; "
          f"objects under 10 pixels: {(sizes < 10).sum()}", flush=True)

    # Histogram of object sizes. Many objects are tiny, so the y-axis is
    # logarithmic: otherwise the few large objects (nuclei) would be invisible.
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(sizes, bins=100, color="steelblue", edgecolor="black", linewidth=0.3)
    ax.set_yscale("log")
    ax.set_xlabel("Object size (number of pixels)")
    ax.set_ylabel("Number of objects (log scale)")
    ax.set_title(f"DAPI, Otsu threshold = {threshold:g}: {n_objects} objects")
    fig.tight_layout()
    fig.show()  # Non-blocking: the script continues to napari.

    # Original image and segmented objects side by side. A Labels layer gives
    # each object number its own color, so you can see which pixels were
    # grouped into one object (hover over an object to see its number).
    viewer = napari.Viewer(title="DAPI: original vs Otsu objects")
    viewer.add_image(dapi, name="DAPI (original)", colormap="gray",
                     scale=scale, units=units,
                     contrast_limits=(float(np.percentile(dapi, 0.5)),
                                      float(np.percentile(dapi, 99.8))))
    viewer.add_labels(labels, name=f"Objects (Otsu > {threshold:g})",
                      scale=scale, units=units)
    # Grid mode places the layers next to each other instead of on top.
    # Toggle it with the grid button (bottom left) to overlay them instead.
    viewer.canvas.grid.enabled = True
    viewer.canvas.overlays.scale_bar.visible = True
    napari.run()


if __name__ == "__main__":
    main()
