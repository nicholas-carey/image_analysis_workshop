"""Print a spaced 10 x 10 pixel sample, then display the full image in napari.

Run from the workshop folder: uv run src/inspect_nucleus_skipping.py [input.tif]
"""

import argparse
from pathlib import Path

import numpy as np
import tifffile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "image",
        nargs="?",
        type=Path,
        default=(Path(__file__).resolve().parent.parent
                 / "testImages" / "nucleus_cropped_1.tif"),
        help="Input TIFF path (default: the workshop's nucleus_cropped_1.tif).",
    )
    args = parser.parse_args()
    # Supplied relative paths are relative to the folder you run the command in.
    image_path = args.image.expanduser()
    if not image_path.is_file():
        parser.error(f"Input file not found: {image_path}")
    pixels = tifffile.imread(image_path)
    if pixels.ndim != 2:
        raise ValueError(f"Expected a 2D image; found shape {pixels.shape}.")
    if min(pixels.shape) < 28:
        raise ValueError("The image needs at least 28 rows and columns for this sample.")

    # Python counts from zero. Step 3 selects one pixel, then skips two:
    # indices 0, 3, 6, 9, 12, 15, 18, 21, 24, 27 along each axis.
    sample = pixels[:30:3, :30:3]
    print(f"Image: {image_path.name}")
    print(f"Number type (dtype): {pixels.dtype}")
    print("\n10 x 10 matrix, skipping two rows and columns between selections:")
    print("Row and column indices: 0, 3, 6, 9, 12, 15, 18, 21, 24, 27")
    with np.printoptions(threshold=np.inf, linewidth=200):
        print(sample)

    import napari

    # Display the original image; the sample above is only for terminal output.
    viewer = napari.Viewer()
    viewer.add_image(pixels, name=image_path.stem, colormap="gray")
    viewer.dims.axis_labels = ("row", "column")
    napari.run()


if __name__ == "__main__":
    main()
