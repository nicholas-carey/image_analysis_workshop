"""Segment the DAPI channel of an OME-TIFF with a fixed (manual) threshold.

Every pixel brighter than the threshold becomes 1 (foreground, "nucleus");
every pixel at or below it becomes 0 (background). The binary result is saved
as an OME-TIFF with the same pixel size, then shown in napari next to the
original DAPI channel. Close the napari window to end the script.

Run (from the workshop folder):
    uv run src/threshold_dapi.py testImages/3Channel_FF_Corrected_1/3Channel_FF_Corrected_1_MMStack_Pos0.ome.tif --threshold 1500
"""

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import tifffile


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", type=Path, help="Input OME-TIFF image.")
    parser.add_argument("--threshold", type=float, default=1500,
                        help="Pixels ABOVE this value become foreground (default: 1500).")
    parser.add_argument("--output", type=Path, help="Output OME-TIFF path.")
    args = parser.parse_args()

    source = args.image.expanduser()
    if not source.is_file():
        parser.error(f"Input file not found: {source}")
    name = source.name.removesuffix(".ome.tif").removesuffix(".tif")
    output = (args.output.expanduser() if args.output else
              source.with_name(f"{name}_DAPI_threshold{args.threshold:g}.ome.tif"))

    # --- 1. Read the image and find the channel named "DAPI" -----------------
    with tifffile.TiffFile(source) as tif:
        if not tif.ome_metadata:
            parser.error("An OME-TIFF with channel names is required to find DAPI.")
        pixels = ET.fromstring(tif.ome_metadata).find("{*}Image/{*}Pixels")
        names = [c.get("Name", "") for c in pixels.findall("{*}Channel")]
        if "DAPI" not in names:
            parser.error(f"No channel named DAPI; found channels: {names}")
        channel = names.index("DAPI")
        series = tif.series[0]
        if series.axes != "CYX":
            parser.error(f"Expected a 2D multichannel image (CYX); found {series.axes}.")
        dapi = series.asarray()[channel]

        # Remember the pixel size so the output keeps the same physical scale.
        pixel_size = {}
        for axis in "XY":
            key = f"PhysicalSize{axis}"
            if key in pixels.attrib:
                pixel_size[key] = float(pixels.get(key))
                pixel_size[key + "Unit"] = pixels.get(key + "Unit", "µm")

    # --- 2. Threshold: the actual segmentation is this one line --------------
    # "dapi > threshold" gives True/False for every pixel; we store it as 1/0.
    mask = (dapi > args.threshold).astype(np.uint8)

    # --- 3. Save the binary image with the original pixel size ---------------
    tifffile.imwrite(output, mask, ome=True, photometric="minisblack",
                     metadata={"axes": "YX", "Channel": {"Name": ["DAPI mask"]},
                               **pixel_size})

    foreground = mask.mean() * 100
    print(f"DAPI channel: {channel} (counting from 0), values {dapi.min()}–{dapi.max()}", flush=True)
    print(f"Threshold: > {args.threshold:g}  ->  {foreground:.1f}% of pixels are foreground", flush=True)
    print(f"Saved: {output}", flush=True)

    # --- 4. Show original and mask side by side in napari --------------------
    import napari

    # Give napari the pixel size so both images are shown in micrometers.
    scale = (pixel_size.get("PhysicalSizeY", 1.0), pixel_size.get("PhysicalSizeX", 1.0))
    units = (pixel_size.get("PhysicalSizeYUnit", "pixel"),
             pixel_size.get("PhysicalSizeXUnit", "pixel"))
    viewer = napari.Viewer(title="DAPI threshold segmentation")
    viewer.add_image(dapi, name="DAPI (original)", colormap="gray",
                     scale=scale, units=units,
                     contrast_limits=(float(np.percentile(dapi, 0.5)),
                                      float(np.percentile(dapi, 99.8))))
    viewer.add_image(mask, name=f"DAPI > {args.threshold:g}", colormap="gray",
                     contrast_limits=(0, 1), scale=scale, units=units)
    # Grid mode places the layers next to each other instead of on top.
    # Toggle it with the grid button (bottom left) to overlay them instead.
    viewer.canvas.grid.enabled = True
    viewer.canvas.overlays.scale_bar.visible = True
    napari.run()


if __name__ == "__main__":
    main()
