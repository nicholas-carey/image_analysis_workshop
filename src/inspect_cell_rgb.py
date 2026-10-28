"""Print RGB values for about 3% of pixels and display the color image.

Run: uv run src/inspect_cell_rgb.py
Optional input: uv run src/inspect_cell_rgb.py testImages/Cell_cropped_2.tif
Close napari to end the script.
"""

import argparse
from pathlib import Path

import numpy as np
import tifffile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "image", nargs="?", type=Path,
        default=Path(__file__).resolve().parent.parent / "testImages" / "Cell_cropped_2.tif",
        help="Path to an RGB TIFF image.",
    )
    image_path = parser.parse_args().image.expanduser()
    if not image_path.is_file():
        parser.error(f"Input file not found: {image_path}")

    with tifffile.TiffFile(image_path) as tif:
        series = tif.series[0]
        if tif.pages[0].photometric.name != "RGB":
            parser.error("Expected a color TIFF with RGB pixels.")
        pixels = series.asarray()
        # Some TIFFs store channels first; put them last for RGB display.
        if series.axes == "SYX":
            pixels = np.moveaxis(pixels, 0, -1)
        elif series.axes != "YXS":
            parser.error(f"Expected one 2D RGB image; found axes {series.axes}.")

    if pixels.ndim != 3 or pixels.shape[-1] not in (3, 4):
        parser.error(f"Expected RGB or RGBA pixels; found shape {pixels.shape}.")

    rows, columns = pixels.shape[:2]
    total = rows * columns
    # Round to the nearest whole pixel: exactly 3% may not be a whole number.
    count = max(1, int(total * 0.03 + 0.5))
    # Imagine reading the pixels row by row. Sample at evenly spaced positions
    # in that list. Rounding gives gaps of 33 or 34 pixels (about 3%).
    indices = np.floor((np.arange(count) + 0.5) * total / count).astype(int)
    sample_rows, sample_columns = np.unravel_index(indices, (rows, columns))

    print(f"Image: {image_path.name}")
    print(f"Rows: {rows}; columns: {columns}; number type: {pixels.dtype}")
    print(f"Sample: {count} of {total} pixels ({100 * count / total:.3f}%)")
    print("Regular spacing in row-by-row order; row and column indices start at 0.\n")
    print(f"{'Row':>7} {'Column':>7} {'R':>12} {'G':>12} {'B':>12}")
    print("-" * 54)
    for row, column in zip(sample_rows, sample_columns):
        red, green, blue = pixels[row, column, :3]
        print(f"{row:7d} {column:7d} {red:>12} {green:>12} {blue:>12}")

    import napari

    viewer = napari.Viewer()
    viewer.add_image(pixels, rgb=True, name=image_path.stem)
    viewer.dims.axis_labels = ("row", "column")
    napari.run()


if __name__ == "__main__":
    main()
