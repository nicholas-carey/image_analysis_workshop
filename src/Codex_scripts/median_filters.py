"""Compare three median filters on a 2D grayscale TIFF.

Run from the project folder:
    uv run src/Codex_scripts/median_filters.py testImages/Nucleus_cropped_noise.tif
Use --no-view to save the results without opening napari.
"""

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.ndimage import median_filter
import tifffile


# True marks the neighbors included in each median calculation.
FOOTPRINTS = {
    "cross_3x3": np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    "square_3x3": np.ones((3, 3), dtype=bool),
    "square_5x5": np.ones((5, 5), dtype=bool),
}


def read_image(path):
    """Read the image and retain its spatial calibration for TIFF and napari."""
    with tifffile.TiffFile(path) as tif:
        image = tif.series[0].asarray()
        if image.ndim != 2 or tif.pages[0].photometric.name != "MINISBLACK":
            raise ValueError("Please supply a single 2D grayscale (MINISBLACK) TIFF.")
        if not np.isfinite(image).all():
            raise ValueError("The input contains NaN or infinite pixel values.")
        tags = tif.pages[0].tags
        save_options = {"photometric": "minisblack"}
        scale = [1.0, 1.0]  # napari expects row (Y), then column (X).
        unit = "pixels"
        if "XResolution" in tags and "YResolution" in tags:
            resolution = tuple(tags[key].value for key in ("XResolution", "YResolution"))
            resolution_unit = int(tags["ResolutionUnit"].value) if "ResolutionUnit" in tags else 1
            save_options.update(resolution=resolution, resolutionunit=resolution_unit)
            if all(numerator > 0 and denominator > 0 for numerator, denominator in resolution):
                scale = [d / n for n, d in reversed(resolution)]
            unit = {2: "inch", 3: "cm"}.get(resolution_unit, "pixels")
        if tif.is_imagej:
            metadata = tif.imagej_metadata or {}
            unit = metadata.get("unit", unit)
            save_options.update(imagej=True, metadata={"axes": "YX", "unit": unit})
        elif tif.ome_metadata:
            # Rebuild OME metadata rather than copying references to the old file.
            root = ET.fromstring(tif.ome_metadata)
            pixels = root.find("{*}Image/{*}Pixels")
            metadata = {"axes": "YX"}
            if pixels is not None:
                for axis in "XY":
                    key = f"PhysicalSize{axis}"
                    if key in pixels.attrib:
                        metadata[key] = float(pixels.attrib[key])
                        metadata[key + "Unit"] = pixels.attrib.get(key + "Unit", "µm")
                # Only use physical scaling when both axes have matching units.
                if all(f"PhysicalSize{axis}" in metadata for axis in "XY"):
                    if metadata["PhysicalSizeXUnit"] == metadata["PhysicalSizeYUnit"]:
                        scale = [metadata["PhysicalSizeY"], metadata["PhysicalSizeX"]]
                        unit = metadata["PhysicalSizeXUnit"]
                    else:
                        scale, unit = [1.0, 1.0], "pixels"
            save_options.update(ome=True, metadata=metadata)
    return image, save_options, scale, unit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path,
                        default=Path(__file__).resolve().parents[2] / "testImages/Nucleus_cropped_noise.tif")
    parser.add_argument("--output-dir", type=Path, help="Default: same folder as the input")
    parser.add_argument("--no-view", action="store_true", help="Save images without opening napari")
    args = parser.parse_args()
    image, save_options, scale, unit = read_image(args.input)
    output_dir = args.output_dir or args.input.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for name, footprint in FOOTPRINTS.items():
        # Replace each pixel with the middle value of its selected neighbors.
        # Reflect the image at its edges so every pixel has a full neighborhood.
        filtered = median_filter(image, footprint=footprint, mode="reflect")
        output = output_dir / f"{args.input.stem}_median_{name}.tif"
        tifffile.imwrite(output, filtered, **save_options)
        results[name] = filtered
        print(f"Saved {output}")

    if not args.no_view:
        import napari

        viewer = napari.Viewer(title=f"Median filter comparison — scale in {unit}")
        limits = (float(image.min()), float(image.max()))
        if limits[0] == limits[1]:
            limits = (limits[0], limits[0] + 1)
        for name, data in {"Original": image, **results}.items():
            viewer.add_image(data, name=name, colormap="gray", scale=scale,
                             contrast_limits=limits)
        viewer.grid.enabled = True
        viewer.grid.shape = (2, 2)
        viewer.reset_view()
        napari.run()


if __name__ == "__main__":
    main()
