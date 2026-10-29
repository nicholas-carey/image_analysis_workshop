"""Recompress an image 100 times at JPEG quality 90 and compare with the TIFF.

Run: uv run src/Codex_scripts/compare_cell_jpeg_repeated.py
Each save uses the decoded result of the previous save. Close napari to finish.
Also estimate a single-save quality with the closest mean absolute RGB error.
"""

import argparse
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, TiffImagePlugin
import tifffile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality", type=int, default=90,
                        help="JPEG quality from 1 to 100 (default: 90).")
    parser.add_argument("--generations", type=int, default=100,
                        help="Number of successive saves (default: 100).")
    args = parser.parse_args()
    if args.generations < 1:
        parser.error("Generations must be at least 1.")
    if not 1 <= args.quality <= 100:
        parser.error("Quality must be between 1 and 100.")

    source = Path(__file__).resolve().parents[2] / "testImages" / "Cell_cropped_2.tif"
    jpeg_path = source.with_name(f"{source.stem}_jpeg_q{args.quality}_repeated{args.generations}.jpg")
    difference_path = source.with_name(f"{source.stem}_jpeg_q{args.quality}_repeated{args.generations}_difference.tif")
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
    jpeg = original.copy()
    # Reuse this new output file for each generation; the final one remains.
    for generation in range(1, args.generations + 1):
        Image.fromarray(jpeg).save(jpeg_path, quality=args.quality,
                                  subsampling=2, exif=exif)
        with Image.open(jpeg_path) as saved:
            jpeg = np.array(saved.convert("RGB"))
        if generation == 1 or generation % 10 == 0 or generation == args.generations:
            error = np.abs(original.astype(np.float32) - jpeg.astype(np.float32)).mean()
            print(f"Generation {generation}: mean absolute RGB error = {error:.4f}")

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

    # A metric-based comparison, not a universal conversion of JPEG quality.
    # Each candidate below starts from the ORIGINAL, with the same encoder
    # and subsampling as compare_cell_jpeg.py. Temporary JPEGs stay in memory.
    target_error = float(rgb_difference.mean())
    candidates = []
    for quality in range(1, 101):
        with BytesIO() as buffer:
            Image.fromarray(original).save(buffer, format="JPEG", quality=quality,
                                           subsampling=2, exif=exif)
            buffer.seek(0)
            with Image.open(buffer) as saved:
                single = np.array(saved.convert("RGB"))
        error = float(np.abs(original.astype(np.float32) - single.astype(np.float32)).mean())
        candidates.append((abs(error - target_error), quality, error))
    _, closest_quality, closest_error = min(candidates)
    print(f"\nRepeated-save mean absolute RGB error: {target_error:.4f}")
    print(f"Closest single-save quality by this metric: {closest_quality}")
    print(f"Single-save mean absolute RGB error: {closest_error:.4f}")
    print("This matches overall error approximately, not individual pixels or appearance.")
    print("Try in the original script:")
    print(f"uv run src/Codex_scripts/compare_cell_jpeg.py --quality {closest_quality}")

    import napari

    # Use the same physical spacing for all three images (in micrometers).
    spacing = (10000 / y_per_cm, 10000 / x_per_cm)
    viewer = napari.Viewer()
    viewer.add_image(original, rgb=True, name="Original TIFF",
                     scale=spacing, units=("um", "um"), contrast_limits=(0, 255))
    viewer.add_image(jpeg, rgb=True, name=f"JPEG ({args.generations} saves, quality {args.quality})",
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
