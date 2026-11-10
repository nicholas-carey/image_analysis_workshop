"""Otsu threshold DAPI, erode/dilate, then split touching nuclei with watershed.

Steps:
  1. Read the channel named "DAPI" and threshold it with Otsu's method.
  2. 3 rounds of erosion, then 3 rounds of dilation ("opening"; --rounds).
  3. Distance transform: for every foreground pixel, its distance to the
     nearest background pixel. Nucleus centers become peaks; where two
     nuclei touch there is a narrow, lower "waist".
  4. Seeds: the local peaks of the distance image, at least --min-distance
     pixels apart. One seed should end up in each nucleus.
  5. Watershed on the NEGATIVE distance image (peaks become valleys): water
     rises from each seed, and where two floods meet a border is drawn.
     The thresholded image limits where the water can go.
  6. Save the labeled image (TIFF with a color LUT and the original pixel size)
     and show every step side by side in napari. Close napari to end the script.

Run (from the workshop folder):
    uv run src/otsu_open_watershed.py testImages/3Channel_FF_Corrected_1/3Channel_FF_Corrected_1_MMStack_Pos0.ome.tif

Try --min-distance 5 or 20 to see too many or too few seeds.
"""

import argparse
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.filters import threshold_otsu
from skimage.measure import label
from skimage.morphology import dilation, erosion
from skimage.segmentation import watershed

# Reuse the reading, footprint and color-LUT code from the previous script.
from otsu_open_label_save import FOOTPRINT, distinct_color_lut, read_dapi


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", type=Path, help="Input OME-TIFF image.")
    parser.add_argument("--rounds", type=int, default=3,
                        help="Rounds of erosion, then the same number of dilations (default 3).")
    parser.add_argument("--min-distance", type=int, default=10,
                        help="Minimum distance in pixels between two seeds (default 10).")
    parser.add_argument("--output", type=Path, help="Output TIFF path.")
    args = parser.parse_args()
    if not args.image.is_file():
        parser.error(f"Input file not found: {args.image}")
    name = args.image.name.removesuffix(".ome.tif").removesuffix(".tif")
    output = args.output or args.image.with_name(
        f"{name}_DAPI_otsu_open{args.rounds}_watershed_md{args.min_distance}_labels.tif")

    dapi, pixel_size_um = read_dapi(args.image)

    # --- 1-2. Threshold and opening (same as the previous script) ------------
    threshold = threshold_otsu(dapi)
    binary = dapi > threshold
    for _ in range(args.rounds):
        binary = erosion(binary, FOOTPRINT)
    for _ in range(args.rounds):
        binary = dilation(binary, FOOTPRINT)
    objects = label(binary)  # objects before watershed, for comparison

    # --- 3. Distance transform -------------------------------------------------
    distance = ndi.distance_transform_edt(binary)

    # --- 4. Seeds: local peaks of the distance image -------------------------
    # labels=objects: look for peaks within each object separately.
    # exclude_border=False: nuclei at the image edge get seeds too.
    peaks = peak_local_max(distance, min_distance=args.min_distance,
                           labels=objects, exclude_border=False)
    # Every object needs at least one seed, otherwise watershed drops it.
    # For an object without a peak, use its pixel farthest from the background.
    has_seed = set(objects[tuple(peaks.T)])
    extra = [np.unravel_index(np.argmax(np.where(objects == i, distance, 0)), distance.shape)
             for i in range(1, objects.max() + 1) if i not in has_seed]
    if extra:
        peaks = np.vstack([peaks, np.array(extra)])
    markers = np.zeros(dapi.shape, dtype=np.int32)
    markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)  # seed numbers 1, 2, 3, ...

    # --- 5. Watershed ---------------------------------------------------------
    # -distance turns peaks into valleys; mask keeps the background empty.
    nuclei = watershed(-distance, markers, mask=binary)
    n_nuclei = int(nuclei.max())

    print(f"Otsu threshold: {threshold:g}; {args.rounds} erosions + "
          f"{args.rounds} dilations (plus-sign footprint)", flush=True)
    print(f"Objects before watershed: {objects.max()}", flush=True)
    print(f"Seeds (min distance {args.min_distance} px): {len(peaks)}"
          f"{f' ({len(extra)} added for objects without a peak)' if extra else ''}", flush=True)
    print(f"Objects after watershed: {n_nuclei}; split objects: "
          f"{sum(len(set(np.unique(nuclei[objects == i])) - {0}) > 1 for i in range(1, objects.max() + 1))}",
          flush=True)

    # --- 6. Save with color LUT and pixel size (as in the previous script) ----
    dtype = np.uint8 if n_nuclei <= 255 else np.uint16
    pixels_per_cm = tuple(1e4 / size for size in reversed(pixel_size_um))  # (X, Y)
    tifffile.imwrite(output, nuclei.astype(dtype), photometric="palette",
                     colormap=distinct_color_lut(np.iinfo(dtype).max + 1),
                     resolution=pixels_per_cm, resolutionunit="CENTIMETER", metadata=None)
    print(f"Saved: {output}", flush=True)

    # --- napari: every step side by side ---------------------------------------
    import napari

    # Three panels, each made of two layers (grid stride 2 = 2 layers per panel):
    #   1. DAPI with the outlines of the objects BEFORE watershed
    #   2. Distance image with the seeds on top
    #   3. DAPI with the outlines of the objects AFTER watershed
    # Compare panels 1 and 3 to see which touching nuclei were split.
    common = dict(scale=pixel_size_um, units=("µm", "µm"))
    limits = (float(np.percentile(dapi, 0.5)), float(np.percentile(dapi, 99.8)))
    viewer = napari.Viewer(title="DAPI watershed")

    viewer.add_image(dapi, name="DAPI", colormap="gray", contrast_limits=limits, **common)
    before = viewer.add_labels(objects, name="Objects before watershed", **common)
    before.contour = 2  # draw outlines (2 pixels wide) instead of filled areas

    viewer.add_image(distance, name="Distance to background", colormap="magma", **common)
    viewer.add_points(peaks * np.array(pixel_size_um), name="Seeds",
                      size=8 * pixel_size_um[0], face_color="cyan", border_width=0)

    viewer.add_image(dapi, name="DAPI ", colormap="gray", contrast_limits=limits, **common)
    after = viewer.add_labels(nuclei, name="Watershed result", **common)
    after.contour = 2  # set to 0 (in the layer controls) for filled labels

    viewer.canvas.grid.stride = 2
    viewer.canvas.grid.enabled = True
    viewer.canvas.overlays.scale_bar.visible = True
    napari.run()


if __name__ == "__main__":
    main()
