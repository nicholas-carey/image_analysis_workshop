"""Print the nucleus image's pixels and dimensions, then open it in napari.

Run from the workshop folder: uv run src/inspect_nucleus.py
Close the napari window to end the script.
"""

from pathlib import Path

import numpy as np
import tifffile


def main():
    # Find the image relative to this script, regardless of the working folder.
    image_path = (
        Path(__file__).resolve().parent.parent
        / "testImages"
        / "nucleus_cropped_1.tif"
    )
    pixels = tifffile.imread(image_path)

    if pixels.ndim != 2:
        raise ValueError(f"Expected a 2D image; found shape {pixels.shape}.")

    # A 2D image is a table of numbers: rows first, then columns.
    rows, columns = pixels.shape
    print(f"Image: {image_path.name}")
    print(f"Rows: {rows}")
    print(f"Columns: {columns}")
    print(f"Number type (dtype): {pixels.dtype}")
    print("\nTop-left 10 × 10 pixel matrix (rows 0–9, columns 0–9):")
    # :10 selects the first ten entries. Keep all values visible in the output.
    with np.printoptions(threshold=np.inf, linewidth=200):
        print(pixels[:10, :10])

    import napari

    viewer = napari.Viewer()
    viewer.add_image(pixels, name=image_path.stem, colormap="gray")
    viewer.dims.axis_labels = ("row", "column")
    napari.run()


if __name__ == "__main__":
    main()
