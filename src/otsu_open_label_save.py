"""Otsu threshold DAPI, erode/dilate, label the objects, and save with a color LUT.

Steps:
  1. Read the channel named "DAPI" from an OME-TIFF.
  2. Otsu threshold -> binary image.
  3. N rounds of erosion, then N rounds of dilation ("opening"; default N = 3).
  4. Label: every connected object gets its own number (1, 2, 3, ...; 0 = background).
  5. Save the label image as a TIFF with a LUT (lookup table, "palette") that gives
     every number its own distinct color, and with the original pixel size.
  6. Open the saved file in napari (with the napari-tiff reader) next to the
     original DAPI image. Right-click the labels image layer and choose
     "Convert to Labels". Close the napari window to end the script.

Run (from the workshop folder):
    uv run src/otsu_open_label_save.py testImages/3Channel_FF_Corrected_1/3Channel_FF_Corrected_1_MMStack_Pos0.ome.tif

Why a plain .tif and not .ome.tif? napari-tiff only shows a TIFF's color LUT
when it is stored as a standard TIFF palette; in an OME-TIFF it is ignored.
"""

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import tifffile
from matplotlib.colors import hsv_to_rgb
from skimage.filters import threshold_otsu
from skimage.measure import label
from skimage.morphology import dilation, erosion

# The structuring element ("footprint"): a 3x3 plus sign, scikit-image's default.
FOOTPRINT = np.array([[0, 1, 0],
                      [1, 1, 1],
                      [0, 1, 0]], dtype=bool)


def read_dapi(path):
    """Return the 2D DAPI channel and its pixel size in micrometers (Y, X)."""
    to_um = {"µm": 1.0, "um": 1.0, "nm": 0.001, "mm": 1000.0}
    with tifffile.TiffFile(path) as tif:
        pixels = ET.fromstring(tif.ome_metadata).find("{*}Image/{*}Pixels")
        names = [c.get("Name", "") for c in pixels.findall("{*}Channel")]
        if "DAPI" not in names:
            raise SystemExit(f"No channel named DAPI; found channels: {names}")
        series = tif.series[0]
        if series.axes != "CYX":
            raise SystemExit(f"Expected a 2D multichannel image (CYX); found {series.axes}.")
        dapi = series.asarray()[names.index("DAPI")]
        pixel_size_um = []
        for axis in "YX":
            unit = pixels.get(f"PhysicalSize{axis}Unit", "µm")
            if f"PhysicalSize{axis}" not in pixels.attrib or unit not in to_um:
                raise SystemExit(f"Cannot read the pixel size of the {axis} axis.")
            pixel_size_um.append(float(pixels.get(f"PhysicalSize{axis}")) * to_um[unit])
        return dapi, tuple(pixel_size_um)


def distinct_color_lut(n_entries, seed=0):
    """Make a LUT: entry 0 (background) is black, every other entry a random
    bright, saturated color, so that neighboring numbers look clearly different.

    TIFF stores a LUT as 3 rows (red, green, blue) of 16-bit values (0-65535),
    one column per possible pixel value.
    """
    rng = np.random.default_rng(seed)
    hsv = np.column_stack([rng.uniform(0, 1, n_entries),       # hue: any color
                           rng.uniform(0.6, 1.0, n_entries),   # saturation: vivid
                           rng.uniform(0.75, 1.0, n_entries)])  # value: bright
    rgb = hsv_to_rgb(hsv)
    rgb[0] = 0  # background stays black
    return np.round(rgb.T * 65535).astype(np.uint16)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", type=Path, help="Input OME-TIFF image.")
    parser.add_argument("--rounds", type=int, default=3,
                        help="Rounds of erosion, then the same number of dilations (default 3).")
    parser.add_argument("--output", type=Path, help="Output TIFF path.")
    args = parser.parse_args()
    if not args.image.is_file():
        parser.error(f"Input file not found: {args.image}")
    name = args.image.name.removesuffix(".ome.tif").removesuffix(".tif")
    output = args.output or args.image.with_name(
        f"{name}_DAPI_otsu_open{args.rounds}_labels.tif")

    dapi, pixel_size_um = read_dapi(args.image)

    # --- Threshold, then opening (erosions followed by dilations) ------------
    threshold = threshold_otsu(dapi)
    binary = dapi > threshold
    for _ in range(args.rounds):
        binary = erosion(binary, FOOTPRINT)
    for _ in range(args.rounds):
        binary = dilation(binary, FOOTPRINT)

    # --- Label: each connected object gets its own number --------------------
    labels = label(binary)
    n_objects = int(labels.max())

    # A LUT needs one color per possible pixel value: 256 for 8-bit images,
    # 65536 for 16-bit. Use 8-bit if there are few enough objects (smaller file).
    dtype = np.uint8 if n_objects <= 255 else np.uint16
    labels = labels.astype(dtype)
    lut = distinct_color_lut(np.iinfo(dtype).max + 1)

    # --- Save: palette TIFF with the original pixel size ---------------------
    # TIFF resolution is "pixels per unit", and the only metric unit TIFF knows
    # is the centimeter: 1 cm = 10,000 µm.
    pixels_per_cm = tuple(1e4 / size for size in reversed(pixel_size_um))  # (X, Y)
    tifffile.imwrite(output, labels, photometric="palette", colormap=lut,
                     resolution=pixels_per_cm, resolutionunit="CENTIMETER",
                     metadata=None)
    # Check that writing the file did not change any label numbers.
    if not np.array_equal(tifffile.imread(output), labels):
        raise RuntimeError("The saved labels differ from the computed labels.")

    print(f"Otsu threshold: {threshold:g}; {args.rounds} erosions + "
          f"{args.rounds} dilations (plus-sign footprint)", flush=True)
    print(f"Objects labeled: {n_objects} (numbers 1-{n_objects}; 0 = background)", flush=True)
    print(f"Saved ({np.dtype(dtype).name}, color LUT, pixel size "
          f"{pixel_size_um[1]:g} x {pixel_size_um[0]:g} µm): {output}", flush=True)

    # --- Open in napari, the same way File > Open with napari-tiff would -----
    import napari

    viewer = napari.Viewer(title="DAPI labels with LUT")
    viewer.add_image(dapi, name="DAPI (original)", colormap="gray",
                     scale=pixel_size_um, units=("µm", "µm"),
                     contrast_limits=(float(np.percentile(dapi, 0.5)),
                                      float(np.percentile(dapi, 99.8))))
    [layer] = viewer.open(output, plugin="napari-tiff")
    # The LUT has a color for every possible value (0-255 or 0-65535); napari
    # would otherwise stretch it over just 0-n_objects and shift the colors.
    layer.contrast_limits = (0, np.iinfo(dtype).max)
    viewer.canvas.grid.enabled = True  # side by side; toggle bottom-left
    viewer.canvas.overlays.scale_bar.visible = True
    print('In napari: right-click the labels layer and choose "Convert to Labels".',
          flush=True)
    napari.run()


if __name__ == "__main__":
    main()
