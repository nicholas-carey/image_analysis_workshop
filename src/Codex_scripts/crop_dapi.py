"""Crop a rectangular region from the named DAPI channel of an OME-TIFF.

Run: uv run src/Codex_scripts/crop_dapi.py path/to/image.ome.tif
Coordinates are [y, x], in pixels by default or micrometers with --coordinate-units um.
Pixel coordinates are zero-based; both endpoint pixels are included.
The crop is saved, then displayed in napari. Close the window to end the script.
"""

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import tifffile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Input OME-TIFF image.")
    parser.add_argument("--output", type=Path, help="Output OME-TIFF path.")
    parser.add_argument("--y", nargs=2, type=float, default=[288, 320],
                        metavar=("FIRST", "LAST"), help="Inclusive row limits.")
    parser.add_argument("--x", nargs=2, type=float, default=[211, 248],
                        metavar=("FIRST", "LAST"), help="Inclusive column limits.")
    parser.add_argument("--coordinate-units", choices=("pixels", "um"), default="pixels",
                        help="Units of --y and --x coordinates (default: pixels).")
    args = parser.parse_args()
    source = args.image.expanduser()
    if not source.is_file():
        parser.error(f"Input file not found: {source}")
    y0, y1 = args.y
    x0, x1 = args.x
    output = (args.output.expanduser() if args.output else source.with_name(
        f"{source.stem}_DAPI_y{y0:g}-{y1:g}_x{x0:g}-{x1:g}_{args.coordinate_units}.ome.tif"))
    if output.exists():
        parser.error(f"Output already exists; choose another --output path: {output}")
    if not output.parent.is_dir():
        parser.error(f"Output folder does not exist: {output.parent}")

    with tifffile.TiffFile(source) as tif:
        if not tif.ome_metadata:
            parser.error("An OME-TIFF with channel names is required to identify DAPI.")
        # Read the first image series and its matching OME metadata.
        root = ET.fromstring(tif.ome_metadata)
        metadata = root.find("{*}Image/{*}Pixels")
        if metadata is None:
            parser.error("No OME pixel metadata found.")
        names = [c.get("Name", "") for c in metadata.findall("{*}Channel")]
        matches = [i for i, name in enumerate(names) if name.strip().casefold() == "dapi"]
        if len(matches) != 1:
            parser.error(f"Expected one channel named DAPI; found channels: {names}")
        channel = matches[0]
        series = tif.series[0]
        image = series.asarray()
        axes = series.axes

        # Select DAPI and remove single-length dimensions such as Z or time.
        # Refuse multi-plane stacks rather than silently choosing a plane.
        selection = []
        remaining_axes = ""
        for axis, length in zip(axes, image.shape):
            if axis == "C":
                selection.append(channel)
            elif axis in "YX":
                selection.append(slice(None))
                remaining_axes += axis
            elif length == 1:
                selection.append(0)
            else:
                parser.error(f"Expected a single 2D multichannel image; found {axes} {image.shape}.")
        if "C" not in axes and len(names) != 1:
            parser.error(f"Cannot identify the channel dimension in axes {axes}.")
        if sorted(remaining_axes) != ["X", "Y"]:
            parser.error(f"Missing row/column dimensions in axes {axes}.")
        dapi = image[tuple(selection)]
        if remaining_axes == "XY":
            dapi = dapi.T

        # Napari world coordinates may be micrometers when the image is scaled.
        # Convert physical bounds to pixel centers inside the requested region.
        if y0 > y1 or x0 > x1:
            parser.error("FIRST must not exceed LAST for either coordinate range.")
        if args.coordinate_units == "um":
            sizes = {}
            to_um = {"µm": 1.0, "um": 1.0, "nm": 0.001, "mm": 1000.0,
                     "cm": 10000.0, "m": 1000000.0}
            for axis in "YX":
                key = f"PhysicalSize{axis}"
                unit = metadata.get(key + "Unit", "µm")
                if key not in metadata.attrib or unit not in to_um:
                    parser.error(f"Cannot convert {axis} coordinates to micrometers.")
                sizes[axis] = float(metadata.get(key)) * to_um[unit]
                if sizes[axis] <= 0:
                    parser.error("Physical pixel sizes must be positive.")
            y0, y1 = int(np.ceil(y0 / sizes["Y"])), int(np.floor(y1 / sizes["Y"]))
            x0, x1 = int(np.ceil(x0 / sizes["X"])), int(np.floor(x1 / sizes["X"]))
        else:
            if any(not float(v).is_integer() for v in (y0, y1, x0, x1)):
                parser.error("Pixel indices must be whole numbers.")
            y0, y1, x0, x1 = map(int, (y0, y1, x0, x1))

        rows, columns = dapi.shape
        if not (0 <= y0 <= y1 < rows and 0 <= x0 <= x1 < columns):
            parser.error(f"Crop limits must be within {rows} rows and {columns} columns.")
        # Python excludes the slice endpoint, so add 1 to include each corner.
        crop = dapi[y0:y1 + 1, x0:x1 + 1]

        # Cropping changes the image dimensions, but not the size of each pixel.
        out_metadata = {"axes": "YX", "Channel": {"Name": ["DAPI"]}}
        for axis in "XY":
            key = f"PhysicalSize{axis}"
            if key in metadata.attrib:
                out_metadata[key] = float(metadata.get(key))
                out_metadata[key + "Unit"] = metadata.get(key + "Unit", "µm")
        # Also copy the TIFF resolution tags for readers that use those tags.
        tags = series.pages[0].tags
        resolution_options = {}
        if "XResolution" in tags and "YResolution" in tags:
            resolution_options["resolution"] = (
                tags["XResolution"].value, tags["YResolution"].value)
            if "ResolutionUnit" in tags:
                resolution_options["resolutionunit"] = tags["ResolutionUnit"].value

    tifffile.imwrite(output, np.ascontiguousarray(crop), ome=True,
                     photometric="minisblack", metadata=out_metadata,
                     **resolution_options)
    # Verify that writing the file did not change any cropped pixel values.
    if not np.array_equal(tifffile.imread(output), crop):
        raise RuntimeError("The saved crop differs from the source pixels.")
    print("Verified: saved crop exactly matches the selected source pixels.")
    print(f"DAPI channel: {channel} (zero-based); source: {source.name}")
    print(f"Included rows {y0}–{y1} and columns {x0}–{x1}")
    print(f"Crop: {crop.shape[0]} rows x {crop.shape[1]} columns; dtype: {crop.dtype}")
    print(f"Saved: {output}")

    import napari

    # Give napari the pixel spacing in row (Y), column (X) order.
    display_options = {}
    if all(f"PhysicalSize{axis}" in out_metadata for axis in "YX"):
        display_options["scale"] = tuple(
            out_metadata[f"PhysicalSize{axis}"] for axis in "YX")
        display_options["units"] = tuple(
            out_metadata[f"PhysicalSize{axis}Unit"] for axis in "YX")

    viewer = napari.Viewer()
    # Share a contrast range so background noise is not stretched independently.
    limits = (float(dapi.min()), float(dapi.max()))
    viewer.add_image(dapi, name="Full DAPI source (toggle to inspect)",
                     colormap="gray", contrast_limits=limits, visible=False,
                     **display_options)
    scale = np.array(display_options.get("scale", (1.0, 1.0)))
    viewer.add_image(crop, name=output.name, colormap="gray", contrast_limits=limits,
                     translate=tuple(np.array([y0, x0]) * scale), **display_options)
    corners = np.array([[y0 - 0.5, x0 - 0.5], [y0 - 0.5, x1 + 0.5],
                        [y1 + 0.5, x1 + 0.5], [y1 + 0.5, x0 - 0.5]]) * scale
    viewer.add_shapes([corners], shape_type="rectangle", name="Crop boundary",
                      edge_color="yellow", face_color="transparent", edge_width=1,
                      visible=False)
    print("In napari, enable Full DAPI source and Crop boundary, then reset the view")
    print("to inspect the crop location. All image layers use the same contrast range.")
    viewer.dims.axis_labels = ("row", "column")
    if display_options:
        viewer.scale_bar.visible = True
    napari.run()


if __name__ == "__main__":
    main()
