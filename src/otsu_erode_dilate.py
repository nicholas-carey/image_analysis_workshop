"""Otsu threshold the DAPI channel, then erode twice and dilate twice ("opening").

Steps:
  1. Read the channel named "DAPI" from an OME-TIFF.
  2. Otsu threshold -> binary image (1 = foreground, 0 = background).
  3. Two rounds of erosion: a pixel stays 1 only if ALL pixels under the
     structuring element are 1. Isolated specks vanish; objects shrink.
  4. Two rounds of dilation: a pixel becomes 1 if ANY pixel under the
     structuring element is 1. Surviving objects grow back to about their
     original size, but the specks that vanished do not come back.
  5. Count the remaining objects and show a histogram of their sizes
     (compared with the objects before erosion/dilation).
  6. Show every intermediate image side by side in napari.
     Close the napari window to end the script.

Run (from the workshop folder):
    uv run src/otsu_erode_dilate.py testImages/3Channel_FF_Corrected_1/3Channel_FF_Corrected_1_MMStack_Pos0.ome.tif

Options:
    --rounds N         number of erosions, then the same number of dilations (default 2)
    --footprint plus   3x3 plus sign (default; scikit-image's default)
    --footprint square full 3x3 square (all 8 neighbors; removes more per round)
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
from skimage.morphology import dilation, erosion

# The structuring element ("footprint"): which neighbors are looked at.
FOOTPRINTS = {
    "plus": np.array([[0, 1, 0],
                      [1, 1, 1],
                      [0, 1, 0]], dtype=bool),
    "square": np.ones((3, 3), dtype=bool),
}


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


def object_sizes(binary):
    """Label connected objects and return (labels, size of each object in pixels)."""
    labels = label(binary)  # 8-connected (corners touch) - scikit-image's default
    return labels, np.bincount(labels.ravel())[1:]  # [1:] drops the background


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", type=Path, help="Input OME-TIFF image.")
    parser.add_argument("--rounds", type=int, default=2,
                        help="Rounds of erosion, then the same number of dilations.")
    parser.add_argument("--footprint", choices=FOOTPRINTS, default="plus",
                        help="Structuring element shape (default: plus).")
    args = parser.parse_args()
    if not args.image.is_file():
        parser.error(f"Input file not found: {args.image}")
    footprint = FOOTPRINTS[args.footprint]

    dapi, scale, units = read_dapi(args.image)

    # --- Otsu threshold -------------------------------------------------------
    threshold = threshold_otsu(dapi)
    otsu = dapi > threshold
    steps = [("Otsu threshold", otsu)]  # (name, image) after each step

    # --- Erosion, N rounds ----------------------------------------------------
    current = otsu
    for i in range(1, args.rounds + 1):
        current = erosion(current, footprint)
        steps.append((f"Erosion {i}", current))

    # --- Dilation, N rounds ---------------------------------------------------
    for i in range(1, args.rounds + 1):
        current = dilation(current, footprint)
        steps.append((f"Dilation {i}", current))

    # --- Count objects before and after ---------------------------------------
    _, sizes_before = object_sizes(otsu)
    labels_after, sizes_after = object_sizes(current)

    print(f"Otsu threshold: {threshold:g}; footprint: {args.footprint}; "
          f"{args.rounds} erosions + {args.rounds} dilations", flush=True)
    for name, image in steps:
        print(f"  {name:<15} {image.sum():>7} foreground pixels, "
              f"{label(image).max():>3} objects", flush=True)
    for name, sizes in [("Before", sizes_before), ("After", sizes_after)]:
        print(f"{name}: {len(sizes)} objects; size smallest {sizes.min()}, "
              f"median {np.median(sizes):g}, largest {sizes.max()} pixels; "
              f"under 10 pixels: {(sizes < 10).sum()}", flush=True)

    # --- Histogram of object sizes --------------------------------------------
    # Same bins for both, so the bars are directly comparable. The y-axis is
    # logarithmic so that bins with only one or two objects stay visible.
    bins = np.linspace(0, max(sizes_before.max(), sizes_after.max()), 101)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(sizes_before, bins=bins, color="lightgray", edgecolor="gray",
            linewidth=0.3, label=f"Otsu only ({len(sizes_before)} objects)")
    ax.hist(sizes_after, bins=bins, color="steelblue", edgecolor="black", alpha=0.8,
            linewidth=0.3,
            label=f"After {args.rounds}x erosion + {args.rounds}x dilation "
                  f"({len(sizes_after)} objects)")
    ax.set_yscale("log")
    ax.set_xlabel("Object size (number of pixels)")
    ax.set_ylabel("Number of objects (log scale)")
    ax.set_title(f"DAPI, Otsu threshold = {threshold:g}, {args.footprint} footprint")
    ax.legend()
    fig.tight_layout()
    fig.show()  # Non-blocking: the script continues to napari.

    # --- napari: every step side by side --------------------------------------
    # Layers fill the grid left to right, top to bottom, in the order added.
    viewer = napari.Viewer(title="DAPI: Otsu, erosion, dilation")
    viewer.add_image(dapi, name="DAPI (original)", colormap="gray",
                     scale=scale, units=units,
                     contrast_limits=(float(np.percentile(dapi, 0.5)),
                                      float(np.percentile(dapi, 99.8))))
    for name, image in steps:
        viewer.add_image(image.astype(np.uint8), name=name, colormap="gray",
                         contrast_limits=(0, 1), scale=scale, units=units)
    # Final objects, each with its own color (hover to see the object number).
    viewer.add_labels(labels_after, name="Remaining objects", scale=scale, units=units)
    viewer.canvas.grid.enabled = True
    viewer.canvas.overlays.scale_bar.visible = True
    napari.run()


if __name__ == "__main__":
    main()
