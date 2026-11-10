"""Measure every nucleus in a label image and summarize the results.

Input: the watershed label image (from otsu_open_watershed.py) and the original
OME-TIFF (for the DAPI pixel values and the pixel size in micrometers).

Per nucleus (saved as a CSV table, one row per nucleus):
  - area in square micrometers
  - long and short axis length (µm) and their ratio (1 = round; 2 = twice as long as wide)
  - sum, mean and standard deviation of the DAPI pixel values under the mask
  - flags: too small, touching the image border, possibly two unseparated nuclei

Which objects count as nuclei?
  - Objects smaller than --min-area (µm²) are removed: specks, not nuclei.
  - Objects larger than --large-factor x the median nucleus area are FLAGGED as
    "possibly unseparated" (two nuclei the watershed did not split). They are
    only excluded with --exclude-large, because a big nucleus may be real
    (e.g. a cell in G2, or a polyploid hepatocyte) - check them in napari first.
  - Objects touching the image edge are cut off, so their area and intensity
    sum are too small. They are FLAGGED, and excluded with --exclude-border.

Run (from the workshop folder):
    uv run src/measure_nuclei.py testImages/3Channel_FF_Corrected_1/3Channel_FF_Corrected_1_MMStack_Pos0_DAPI_otsu_open3_watershed_md10_labels.tif testImages/3Channel_FF_Corrected_1/3Channel_FF_Corrected_1_MMStack_Pos0.ome.tif
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import tifffile
from skimage.measure import regionprops_table

# Reuse the DAPI reader (also returns the pixel size) from the earlier script.
from otsu_open_label_save import read_dapi


def intensity_sum(mask, intensity):
    """Sum of the pixel values inside one object (extra property for regionprops)."""
    return intensity[mask].sum(dtype=np.int64)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("labels", type=Path, help="Label image TIFF (0 = background).")
    parser.add_argument("image", type=Path, help="Original OME-TIFF with a DAPI channel.")
    parser.add_argument("--min-area", type=float, default=30,
                        help="Smallest area (µm²) that counts as a nucleus (default 30).")
    parser.add_argument("--large-factor", type=float, default=1.75,
                        help="Flag objects larger than this x the median area (default 1.75).")
    parser.add_argument("--exclude-large", action="store_true",
                        help="Exclude objects flagged as possibly unseparated.")
    parser.add_argument("--exclude-border", action="store_true",
                        help="Exclude objects that touch the image border.")
    args = parser.parse_args()
    for path in (args.labels, args.image):
        if not path.is_file():
            parser.error(f"File not found: {path}")

    labels = tifffile.imread(args.labels)
    dapi, (size_y, size_x) = read_dapi(args.image)  # pixel size in µm
    if labels.shape != dapi.shape:
        parser.error(f"Label image {labels.shape} and DAPI {dapi.shape} differ in size.")
    if abs(size_y - size_x) > 1e-6:
        parser.error("Pixels are not square; axis lengths in µm would be wrong.")
    um_per_pixel = size_x
    um2_per_pixel = size_y * size_x

    # --- Measure every object at once ----------------------------------------
    # regionprops_table looks at each label number separately. With
    # intensity_image=dapi it also reads the DAPI values under each object.
    props = regionprops_table(
        labels, intensity_image=dapi,
        properties=("label", "area", "axis_major_length", "axis_minor_length",
                    "intensity_mean", "intensity_std", "bbox"),
        extra_properties=(intensity_sum,))

    area_um2 = props["area"] * um2_per_pixel
    major_um = props["axis_major_length"] * um_per_pixel
    minor_um = props["axis_minor_length"] * um_per_pixel
    # Long axis / short axis. A 1-pixel object has a short axis of 0: no ratio.
    with np.errstate(divide="ignore", invalid="ignore"):
        aspect_ratio = np.where(minor_um > 0, major_um / minor_um, np.nan)

    # --- Flags -----------------------------------------------------------------
    too_small = area_um2 < args.min_area
    rows, cols = labels.shape
    touches_border = ((props["bbox-0"] == 0) | (props["bbox-1"] == 0) |
                      (props["bbox-2"] == rows) | (props["bbox-3"] == cols))
    # "Large" is relative to the median of the real (not too small) nuclei.
    median_area = np.median(area_um2[~too_small])
    large = ~too_small & (area_um2 > args.large_factor * median_area)

    keep = ~too_small
    if args.exclude_large:
        keep &= ~large
    if args.exclude_border:
        keep &= ~touches_border

    # --- Per-nucleus table (all objects, with flags and an 'included' column) --
    stem = args.labels.name.removesuffix(".tif")
    table_path = args.labels.with_name(f"{stem}_measurements.csv")
    columns = {
        "label": props["label"],
        "area_um2": area_um2,
        "major_axis_um": major_um,
        "minor_axis_um": minor_um,
        "aspect_ratio": aspect_ratio,
        "dapi_sum": props["intensity_sum"],
        "dapi_mean": props["intensity_mean"],
        "dapi_std": props["intensity_std"],
        "too_small": too_small,
        "touches_border": touches_border,
        "possibly_unseparated": large,
        "included": keep,
    }
    with open(table_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for row in zip(*columns.values()):
            writer.writerow([f"{v:.4g}" if isinstance(v, float) else
                             int(v) if isinstance(v, (bool, np.bool_)) else v for v in row])

    # --- Summary over the included nuclei -------------------------------------
    print(f"Pixel size: {um_per_pixel:g} µm  ({um2_per_pixel:.4f} µm² per pixel)")
    print(f"Objects in label image:          {len(area_um2)}")
    print(f"  smaller than {args.min_area:g} µm² (removed): {too_small.sum()}")
    print(f"  touching the border:           {touches_border.sum()}"
          f"{' (excluded)' if args.exclude_border else ' (kept; --exclude-border to remove)'}")
    print(f"  possibly unseparated (> {args.large_factor:g} x median "
          f"{median_area:.0f} µm²): {large.sum()}"
          f"{' (excluded)' if args.exclude_large else ' (kept; --exclude-large to remove)'}"
          f"  labels: {props['label'][large].tolist()}")
    print(f"NUCLEI COUNTED: {keep.sum()}")
    print()

    summary = {
        "Area (µm²)": area_um2,
        "Long/short axis ratio": aspect_ratio,
        "DAPI sum": props["intensity_sum"].astype(float),
        "DAPI mean": props["intensity_mean"],
        "DAPI std": props["intensity_std"],
    }
    print(f"{'(included nuclei)':<24}{'mean':>12}{'median':>12}{'min':>12}{'max':>12}")
    for name, values in summary.items():
        v = values[keep]
        print(f"{name:<24}" + "".join(f"{x:>12.4g}" for x in
                                      (v.mean(), np.median(v), v.min(), v.max())))
    print()
    print(f"Per-nucleus table: {table_path}")


if __name__ == "__main__":
    main()
