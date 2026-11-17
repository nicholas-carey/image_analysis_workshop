"""Estimate and remove the DAPI background with a White Top Hat filter.

How it works:
  1. Opening = erosion followed by dilation with a large disk ("footprint").
     Erosion replaces every pixel by the DIMMEST value within the disk. If the
     disk is bigger than a nucleus, it cannot fit inside any nucleus, so every
     nucleus is wiped out and only the surrounding background level remains.
     Dilation (brightest value within the disk) then restores the size of the
     background structures. The result is the BACKGROUND estimate.
  2. White Top Hat = original - opening. What is left are the bright objects
     smaller than the disk (the nuclei) sitting on a background of ~0.

  The disk radius matters: it must be larger than the largest nucleus radius.
  Too small -> the disk fits inside big nuclei, and part of their signal is
  counted as background (nuclei get "hollowed out"). Too large -> the background
  estimate cannot follow real changes in the background. Try --radius 3 and
  --radius 40 to see both effects.

Outputs (OME-TIFFs next to the input, with the original pixel size):
  <name>_DAPI_background_r<radius>um.ome.tif   the background estimate
  <name>_DAPI_tophat_r<radius>um.ome.tif       the background-corrected image
Then the original, the background and the corrected image are shown in napari
side by side. Close napari to end the script.

Run (from the workshop folder):
    uv run src/tophat_background.py testImages/3Channel_FF_Corrected_1/3Channel_FF_Corrected_1_MMStack_Pos0.ome.tif
"""

import argparse
from pathlib import Path

import numpy as np
import tifffile
from skimage.morphology import disk, opening

# Reuse the DAPI reader (returns the image and the pixel size in µm).
from otsu_open_label_save import read_dapi


def save_ome(path, image, pixel_size_um, name):
    """Save a 2D image as OME-TIFF and keep the pixel size of the original."""
    tifffile.imwrite(path, image, ome=True, photometric="minisblack",
                     metadata={"axes": "YX", "Channel": {"Name": [name]},
                               "PhysicalSizeY": pixel_size_um[0], "PhysicalSizeYUnit": "µm",
                               "PhysicalSizeX": pixel_size_um[1], "PhysicalSizeXUnit": "µm"})
    print(f"Saved: {path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", type=Path, help="Input OME-TIFF with a DAPI channel.")
    parser.add_argument("--radius", type=float, default=15,
                        help="Disk radius in µm; larger than the biggest nucleus radius (default 15).")
    args = parser.parse_args()
    if not args.image.is_file():
        parser.error(f"Input file not found: {args.image}")

    dapi, pixel_size_um = read_dapi(args.image)
    radius_px = int(round(args.radius / pixel_size_um[1]))   # µm -> pixels

    # --- 1. Background = opening with a disk larger than a nucleus -----------
    # decomposition="sequence" builds the big disk from a few small steps:
    # the same result, but much faster than one huge disk.
    footprint = disk(radius_px, decomposition="sequence")
    background = opening(dapi, footprint)

    # --- 2. White Top Hat = original - background ----------------------------
    # The opening is never brighter than the original, so this never goes below 0.
    corrected = dapi - background

    print(f"Disk radius: {args.radius:g} µm = {radius_px} pixels "
          f"(pixel size {pixel_size_um[1]:g} µm)", flush=True)
    print(f"Median pixel value - original: {np.median(dapi):g}, "
          f"background: {np.median(background):g}, corrected: {np.median(corrected):g}",
          flush=True)

    # --- 3. Save, with the original pixel size --------------------------------
    stem = args.image.name.removesuffix(".ome.tif").removesuffix(".tif")
    tag = f"r{args.radius:g}um"
    save_ome(args.image.with_name(f"{stem}_DAPI_background_{tag}.ome.tif"),
             background, pixel_size_um, "DAPI background")
    save_ome(args.image.with_name(f"{stem}_DAPI_tophat_{tag}.ome.tif"),
             corrected, pixel_size_um, "DAPI top hat")

    # --- 4. napari: the three images side by side ------------------------------
    import napari

    common = dict(scale=pixel_size_um, units=("µm", "µm"), colormap="gray")
    # Original and background share the same contrast, so you can compare their levels.
    limits = (float(np.percentile(dapi, 0.5)), float(np.percentile(dapi, 99.8)))
    viewer = napari.Viewer(title="White Top Hat background")
    viewer.add_image(dapi, name="DAPI original", contrast_limits=limits, **common)
    viewer.add_image(background, name=f"Background (opening, {args.radius:g} µm disk)",
                     contrast_limits=limits, **common)
    viewer.add_image(corrected, name="Background corrected (top hat)",
                     contrast_limits=(0, float(np.percentile(corrected, 99.8))), **common)
    viewer.canvas.grid.enabled = True   # show the layers next to each other
    viewer.canvas.overlays.scale_bar.visible = True
    napari.run()


if __name__ == "__main__":
    main()
