"""Segment nuclei (DAPI) and whole cells (FITC cytoplasm) with Cellpose-SAM.

Steps:
  1. Read the DAPI and FITC channels (by name) and the pixel size.
  2. Nuclei: run Cellpose-SAM on the DAPI channel alone.
  3. Cells: run Cellpose-SAM on FITC (cytoplasm) plus DAPI. The DAPI channel
     tells Cellpose where each cell's nucleus is, which helps it separate
     neighboring cells whose cytoplasm touches.
  4. Save both label images (TIFF with a color LUT and the original pixel size).
  5. Show the original channels and both masks in napari. Close napari to end.

Cellpose-SAM runs on the Mac GPU (Apple "MPS") by default; that takes a few
seconds per mask. Use --cpu to compare (this is much slower: minutes).

Run (from the workshop folder):
    uv run src/cellpose_nuclei_cytoplasm.py testImages/3Channel_FF_Corrected_1/3Channel_FF_Corrected_1_MMStack_Pos0.ome.tif

Try --cells-from-fitc-only to segment cells from the FITC channel alone and
see what the DAPI channel adds.
"""

import argparse
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import tifffile

# Reuse the color-LUT code from an earlier script.
from otsu_open_label_save import distinct_color_lut


def read_channels(path, wanted):
    """Return {channel name: 2D image} for the wanted channels, plus the pixel size (µm, Y and X)."""
    to_um = {"µm": 1.0, "um": 1.0, "nm": 0.001, "mm": 1000.0}
    with tifffile.TiffFile(path) as tif:
        pixels = ET.fromstring(tif.ome_metadata).find("{*}Image/{*}Pixels")
        names = [c.get("Name", "") for c in pixels.findall("{*}Channel")]
        series = tif.series[0]
        if series.axes != "CYX":
            raise SystemExit(f"Expected a 2D multichannel image (CYX); found {series.axes}.")
        data = series.asarray()
        missing = [n for n in wanted if n not in names]
        if missing:
            raise SystemExit(f"Channel(s) {missing} not found; the image has: {names}")
        pixel_size_um = []
        for axis in "YX":
            unit = pixels.get(f"PhysicalSize{axis}Unit", "µm")
            if f"PhysicalSize{axis}" not in pixels.attrib or unit not in to_um:
                raise SystemExit(f"Cannot read the pixel size of the {axis} axis.")
            pixel_size_um.append(float(pixels.get(f"PhysicalSize{axis}")) * to_um[unit])
        all_channels = {n: data[i] for i, n in enumerate(names)}
    return all_channels, tuple(pixel_size_um)


def save_labels(labels, path, pixel_size_um):
    """Save a label image as TIFF with a color LUT and the pixel size."""
    dtype = np.uint8 if labels.max() <= 255 else np.uint16
    pixels_per_cm = tuple(1e4 / size for size in reversed(pixel_size_um))  # (X, Y)
    tifffile.imwrite(path, labels.astype(dtype), photometric="palette",
                     colormap=distinct_color_lut(np.iinfo(dtype).max + 1),
                     resolution=pixels_per_cm, resolutionunit="CENTIMETER", metadata=None)
    print(f"Saved: {path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", type=Path, help="Input OME-TIFF image.")
    parser.add_argument("--cpu", action="store_true",
                        help="Run on the CPU instead of the GPU (much slower).")
    parser.add_argument("--cells-from-fitc-only", action="store_true",
                        help="Segment cells from FITC alone (without the DAPI helper channel).")
    args = parser.parse_args()
    if not args.image.is_file():
        parser.error(f"Input file not found: {args.image}")
    name = args.image.name.removesuffix(".ome.tif").removesuffix(".tif")

    channels, pixel_size_um = read_channels(args.image, ["DAPI", "FITC"])
    dapi, fitc = channels["DAPI"], channels["FITC"]

    # Import Cellpose here: it takes a few seconds to load.
    from cellpose import models

    # --- Load the Cellpose-SAM model (downloaded once, ~1 GB) -----------------
    model = models.CellposeModel(gpu=not args.cpu)
    print(f"Cellpose-SAM running on: {model.device}", flush=True)

    # --- Nuclei from DAPI ------------------------------------------------------
    # eval() returns (masks, flows, styles); we only need the masks: an image
    # where every pixel holds the number of the object it belongs to (0 = background).
    start = time.time()
    nuclei, _, _ = model.eval(dapi)
    print(f"Nuclei: {nuclei.max()} found in {time.time() - start:.1f} s", flush=True)

    # --- Cells from FITC (+ DAPI) ----------------------------------------------
    # Stack the channels on the last axis (Y, X, channel). Cellpose-SAM reads
    # them in order: first the cytoplasm, then the nucleus channel.
    if args.cells_from_fitc_only:
        cell_input, channel_axis = fitc, None
    else:
        cell_input, channel_axis = np.stack([fitc, dapi], axis=-1), -1
    start = time.time()
    cells, _, _ = model.eval(cell_input, channel_axis=channel_axis)
    print(f"Cells:  {cells.max()} found in {time.time() - start:.1f} s", flush=True)

    # --- Save -------------------------------------------------------------------
    suffix = "_fitconly" if args.cells_from_fitc_only else ""
    save_labels(nuclei, args.image.with_name(f"{name}_DAPI_cellpose_nuclei_labels.tif"), pixel_size_um)
    save_labels(cells, args.image.with_name(f"{name}_FITC_cellpose_cells{suffix}_labels.tif"), pixel_size_um)

    # --- napari --------------------------------------------------------------------
    import napari

    common = dict(scale=pixel_size_um, units=("µm", "µm"))
    viewer = napari.Viewer(title="Cellpose-SAM: nuclei and cells")

    # Original image: every channel as its own colored layer, blended together.
    colors = {"DAPI": "blue", "FITC": "green", "Cy3": "magenta"}
    for channel_name, image in channels.items():
        limits = (float(np.percentile(image, 0.5)), float(np.percentile(image, 99.8)))
        viewer.add_image(image, name=channel_name, colormap=colors.get(channel_name, "gray"),
                         blending="additive", contrast_limits=limits,
                         visible=channel_name in ("DAPI", "FITC"), **common)

    # Masks as outlines on top. Set "contour" to 0 in the layer controls for filled labels.
    cell_layer = viewer.add_labels(cells, name="Cells (FITC)", **common)
    cell_layer.contour = 2
    nuclei_layer = viewer.add_labels(nuclei, name="Nuclei (DAPI)", **common)
    nuclei_layer.contour = 2

    viewer.canvas.overlays.scale_bar.visible = True
    napari.run()


if __name__ == "__main__":
    main()
