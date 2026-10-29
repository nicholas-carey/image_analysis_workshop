"""Save Cell_cropped_2.tif as JPEG and visualize compression differences.

Run: uv run src/Codex_scripts/compare_cell_jpeg.py
Try stronger compression: uv run src/Codex_scripts/compare_cell_jpeg.py --quality 30
Close napari to end the script. Outputs are saved beside the source TIFF.
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, TiffImagePlugin
import tifffile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality", type=int, default=90,
                        help="JPEG quality from 1 to 100 (default: 90).")
    args = parser.parse_args()
    if not 1 <= args.quality <= 100:
        parser.error("Quality must be between 1 and 100.")

    source = Path(__file__).resolve().parents[2] / "testImages" / "Cell_cropped_2.tif"
    jpeg_path = source.with_name(f"{source.stem}_jpeg_q{args.quality}.jpg")
    difference_path = source.with_name(f"{source.stem}_jpeg_q{args.quality}_difference.tif")
    for path in (jpeg_path, difference_path):
        if path.exists():
            parser.error(f"Output already exists: {path}. Move it or choose another quality.")

    with tifffile.TiffFile(source) as tif:
        original = tif.asarray()
        if original.ndim != 3 or original.shape[-1] != 3 or original.dtype != np.uint8:
            parser.error("This script expects an 8-bit RGB TIFF (rows, columns, 3).")
        tags = tif.pages[0].tags
        # ImageJ stores pixels per physical unit in these resolution tags.
        resolution = (tags["XResolution"].value, tags["YResolution"].value)
        resolution_unit = int(tags["ResolutionUnit"].value)
        imagej_metadata = tif.imagej_metadata or {}
        unit = imagej_metadata.get("unit", "")
        x_resolution = resolution[0][0] / resolution[0][1]
        y_resolution = resolution[1][0] / resolution[1][1]

    # JPEG EXIF uses inches or centimeters. Convert the source calibration to
    # pixels per centimeter; this keeps the physical pixel size unchanged.
    if resolution_unit == 2:
        x_per_cm, y_per_cm = x_resolution / 2.54, y_resolution / 2.54
    elif resolution_unit == 3:
        x_per_cm, y_per_cm = x_resolution, y_resolution
    elif unit.lower() in ("micron", "microns", "um", "µm", "micrometer"):
        x_per_cm, y_per_cm = x_resolution * 10000, y_resolution * 10000
    else:
        parser.error(f"Cannot interpret the source pixel calibration: {unit!r}.")
    exif = Image.Exif()
    exif[282] = TiffImagePlugin.IFDRational(x_per_cm)  # XResolution
    exif[283] = TiffImagePlugin.IFDRational(y_per_cm)  # YResolution
    exif[296] = 3  # centimeters
    Image.fromarray(original).save(jpeg_path, quality=args.quality,
                                   subsampling=2, exif=exif)

    # Read the saved JPEG so we measure the actual compression changes.
    with Image.open(jpeg_path) as saved:
        jpeg = np.array(saved.convert("RGB"))

    # Convert before subtraction: unsigned 8-bit numbers cannot hold negatives.
    rgb_difference = np.abs(original.astype(np.float32) - jpeg.astype(np.float32))
    # One value per pixel: average the absolute changes in R, G, and B.
    difference = rgb_difference.mean(axis=2)
    tifffile.imwrite(
        difference_path, difference, imagej=True,
        photometric="minisblack", resolution=resolution,
        resolutionunit=resolution_unit,
        metadata={"axes": "YX", "unit": unit} if unit else {"axes": "YX"},
    )
    print(f"Saved JPEG: {jpeg_path}")
    print(f"Saved mean absolute RGB difference: {difference_path}")
    print(f"Mean absolute channel difference: {rgb_difference.mean():.4f}")
    print(f"Largest absolute channel difference: {rgb_difference.max():.0f}")
    print("Difference values are in the original 0–255 intensity units.")

    import napari

    # Use the same physical spacing for all three images (in micrometers).
    spacing = (10000 / y_per_cm, 10000 / x_per_cm)
    viewer = napari.Viewer()
    viewer.add_image(original, rgb=True, name="Original TIFF",
                     scale=spacing, units=("um", "um"), contrast_limits=(0, 255))
    viewer.add_image(jpeg, rgb=True, name=f"JPEG (quality {args.quality})",
                     scale=spacing, units=("um", "um"), contrast_limits=(0, 255))
    viewer.add_image(difference, name="Mean absolute RGB difference",
                     colormap="inferno", scale=spacing, units=("um", "um"),
                     contrast_limits=(0, max(float(difference.max()), 1.0)))
    # Arrange the three layers side by side for comparison.
    viewer.grid.enabled = True
    viewer.grid.shape = (1, 3)
    viewer.dims.axis_labels = ("row", "column")
    viewer.scale_bar.visible = True
    napari.run()


if __name__ == "__main__":
    main()
