"""Compare nuclear DAPI intensities with and without flatfield correction.

Data: the 25 tiles (5 x 5 grid) of 3Channel_Raw_25Pos_1, taken with a 20x objective
(Plan Apo VC 20x NA 0.75, 0.3238 µm pixels). Flatfield images: Flatfield/20x/DAPI-20x.tif
and the camera dark image Flatfield/Dark.tif.

Steps, for every tile:
  1. Read the DAPI channel (the measured image, MI).
  2. Flatfield correction (see Session6 README):  CI = (MI - BI) / FI
       BI = dark image (camera offset, ~100 counts per pixel)
       FI = (DAPI flatfield - dark), divided by its own mean so it averages 1.
            Values > 1: brighter-than-average illumination; < 1: dimmer (edges).
  3. Segment nuclei ONCE, on the corrected image (Otsu threshold, opening,
     watershed - the same recipe as otsu_open_watershed.py). The SAME nuclei are
     then measured in both images, so any difference comes from the correction
     and not from a slightly different segmentation.
  4. Per nucleus: summed DAPI intensity minus local background.
       local background = median of a ring of pixels around the nucleus
                          (--ring-gap to --ring-width pixels away, other nuclei excluded)
       background-corrected sum = sum of pixels - background x area (pixels)
     This is done separately in the raw and in the corrected image.
  5. Nuclei touching the tile edge (cut off), smaller than --min-area µm², or
     with low solidity (likely two nuclei the watershed did not split) are dropped.

Outputs (in --outdir, default 3Channel_Raw_25Pos_1/flatfield_analysis):
  nuclei_dapi_raw_vs_flatfield.csv   one row per nucleus, both measurements
  dapi_sum_histogram.png             histogram of the background-corrected sums
                                     + intensity vs. distance from the tile center

Run (from the workshop folder):
    uv run src/flatfield_compare_nuclei.py

Look at the segmentation of one tile in napari (raw, corrected, flatfield, nuclei):
    uv run src/flatfield_compare_nuclei.py --show Pos-1-002_002
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops_table
from skimage.morphology import dilation, erosion
from skimage.segmentation import clear_border, expand_labels, relabel_sequential, watershed

FOOTPRINT = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
RAW_COLOR, FF_COLOR = "#2a78d6", "#eb6834"  # blue = raw, orange = flatfield corrected


def read_tile_dapi(path):
    """DAPI plane and pixel size (µm) of one Micro-Manager tile file.

    Each file holds the 3 channels of one position. Channel names are in the
    sidecar *_metadata.txt (Summary -> ChNames); pixel size is in each page's tags.
    """
    meta = json.loads(path.with_name(path.name.replace(".ome.tif", "_metadata.txt")).read_text())
    names = meta["Summary"]["ChNames"]
    if "DAPI" not in names:
        raise SystemExit(f"{path.name}: no DAPI channel, found {names}")
    with tifffile.TiffFile(path) as tif:
        page = tif.pages[names.index("DAPI")]
        pixel_um = float(page.tags["MicroManagerMetadata"].value["PixelSizeUm"])
        return page.asarray().astype(np.float64), pixel_um


def segment(image, rounds, min_distance):
    """Otsu threshold -> opening -> distance-transform watershed (as in session 5)."""
    binary = image > threshold_otsu(image)
    for _ in range(rounds):
        binary = erosion(binary, FOOTPRINT)
    for _ in range(rounds):
        binary = dilation(binary, FOOTPRINT)
    objects = label(binary)
    distance = ndi.distance_transform_edt(binary)
    peaks = peak_local_max(distance, min_distance=min_distance, labels=objects,
                           exclude_border=False)
    markers = np.zeros(image.shape, dtype=np.int32)
    markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
    # objects without a peak still get one seed (their most central pixel)
    seeded = set(objects[tuple(peaks.T)])
    next_id = len(peaks) + 1
    for i, sl in enumerate(ndi.find_objects(objects), start=1):
        if i not in seeded and sl is not None:
            d = np.where(objects[sl] == i, distance[sl], 0)
            r, c = np.unravel_index(np.argmax(d), d.shape)
            markers[sl[0].start + r, sl[1].start + c] = next_id
            next_id += 1
    nuclei = watershed(-distance, markers, mask=binary)
    nuclei = clear_border(nuclei)            # drop nuclei cut off by the tile edge
    return relabel_sequential(nuclei)[0]


def local_background(image, nuclei, ids, gap, width):
    """Median of a ring around each nucleus, ignoring pixels near any nucleus.

    Pixels between `gap` and `width` px from a nucleus belong to its ring
    (expand_labels gives each pixel to the nearest nucleus, so rings of
    neighbors do not overlap). Falls back to the tile-wide background
    when a ring is (almost) empty, e.g. in a crowded area.
    """
    near_any = expand_labels(nuclei, gap) > 0
    grown = expand_labels(nuclei, width)
    ring = np.where(near_any, 0, grown)
    tile_bg = float(np.median(image[~near_any]))
    counts = ndi.sum(np.ones_like(image), ring, ids)
    med = np.array(ndi.median(image, ring, ids), dtype=float)
    return np.where(counts >= 20, med, tile_bg)


def measure(image, nuclei, ids, gap, width):
    """Background-corrected summed intensity and the background value per nucleus."""
    total = np.asarray(ndi.sum(image, nuclei, ids))
    area = np.asarray(ndi.sum(np.ones_like(image), nuclei, ids))
    bg = local_background(image, nuclei, ids, gap, width)
    return total - bg * area, bg


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tiles", type=Path, default=Path("3Channel_Raw_25Pos_1"))
    p.add_argument("--flatfield", type=Path, default=Path("Flatfield/20x/DAPI-20x.tif"))
    p.add_argument("--dark", type=Path, default=Path("Flatfield/Dark.tif"))
    p.add_argument("--outdir", type=Path, help="Output folder (default: <tiles>/flatfield_analysis)")
    p.add_argument("--rounds", type=int, default=3, help="Erosion/dilation rounds (default 3).")
    p.add_argument("--min-distance", type=int, default=10, help="Watershed seed distance, px.")
    p.add_argument("--min-area", type=float, default=30, help="Smallest nucleus, µm² (default 30).")
    p.add_argument("--min-solidity", type=float, default=0.9,
                   help="Drop objects less convex than this (likely merged nuclei; default 0.9).")
    p.add_argument("--ring-gap", type=int, default=4, help="Background ring starts this many px out.")
    p.add_argument("--ring-width", type=int, default=15, help="Background ring ends this many px out.")
    p.add_argument("--show", metavar="POSITION", help="Open this tile in napari, e.g. Pos-1-002_002.")
    args = p.parse_args()

    tiles = sorted(args.tiles.glob("*.ome.tif"))
    if not tiles:
        p.error(f"No .ome.tif files in {args.tiles}")
    outdir = args.outdir or args.tiles / "flatfield_analysis"
    outdir.mkdir(parents=True, exist_ok=True)

    # --- Flatfield image FI, normalized to mean 1 ------------------------------
    dark = tifffile.imread(args.dark).astype(np.float64)
    ff = tifffile.imread(args.flatfield).astype(np.float64) - dark
    fi = ff / ff.mean()
    print(f"Flatfield {args.flatfield.name}: dimmest/brightest pixel = "
          f"{fi.min():.2f} / {fi.max():.2f} of the mean "
          f"(corner-to-center illumination drop ~{(1 - fi.min() / fi.max()) * 100:.0f}%)")

    rows = []
    for path in tiles:
        position = path.name.split("MMStack_")[1].removesuffix(".ome.tif")
        mi, pixel_um = read_tile_dapi(path)
        if mi.shape != fi.shape:
            p.error(f"{path.name}: image {mi.shape} and flatfield {fi.shape} differ in size.")
        ci = (mi - dark) / fi                                   # flatfield corrected

        nuclei = segment(ci, args.rounds, args.min_distance)
        props = regionprops_table(nuclei, properties=("label", "area", "solidity", "centroid"))
        area_um2 = props["area"] * pixel_um**2
        keep = (area_um2 >= args.min_area) & (props["solidity"] >= args.min_solidity)
        ids = props["label"][keep]

        raw_sum, raw_bg = measure(mi, nuclei, ids, args.ring_gap, args.ring_width)
        ff_sum, ff_bg = measure(ci, nuclei, ids, args.ring_gap, args.ring_width)
        cy, cx = props["centroid-0"][keep], props["centroid-1"][keep]
        r_um = np.hypot(cy - (mi.shape[0] - 1) / 2, cx - (mi.shape[1] - 1) / 2) * pixel_um
        for k, i in enumerate(ids):
            rows.append(dict(position=position, nucleus=int(i),
                             area_um2=round(float(area_um2[keep][k]), 2),
                             dist_from_center_um=round(float(r_um[k]), 1),
                             flatfield_value=round(float(fi[int(cy[k]), int(cx[k])]), 4),
                             raw_background=round(float(raw_bg[k]), 1),
                             raw_bgcorr_sum=round(float(raw_sum[k]), 1),
                             ff_background=round(float(ff_bg[k]), 1),
                             ff_bgcorr_sum=round(float(ff_sum[k]), 1)))
        print(f"{position}: {len(ids)} nuclei kept (of {nuclei.max()} not touching the edge)",
              flush=True)

        if args.show == position:
            show_in_napari(mi, ci, fi, nuclei, pixel_um)

    csv_path = outdir / "nuclei_dapi_raw_vs_flatfield.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {len(rows)} nuclei: {csv_path}")

    plot(rows, outdir / "dapi_sum_histogram.png")


def robust_cv(x):
    """Spread relative to the median: (IQR / 1.349) / median, in %.
    For a bell curve this equals the usual CV, but outliers barely affect it."""
    q1, q2, q3 = np.percentile(x, [25, 50, 75])
    return (q3 - q1) / 1.349 / q2 * 100


def plot(rows, out_png):
    import matplotlib.pyplot as plt

    raw = np.array([r["raw_bgcorr_sum"] for r in rows])
    ffc = np.array([r["ff_bgcorr_sum"] for r in rows])
    dist = np.array([r["dist_from_center_um"] for r in rows])

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b",
                         "xtick.color": "#52514e", "ytick.color": "#52514e"})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6), facecolor="#fcfcfb",
                                   gridspec_kw=dict(width_ratios=[1.4, 1]))

    # --- Left: histogram -----------------------------------------------------
    # Both measurements are scaled to "x median", so the raw numbers (counts)
    # and the corrected numbers (counts / FI) share one x-axis.
    bins = np.linspace(0, 3, 61)
    for values, color, name in ((raw, RAW_COLOR, "Without flatfield"),
                                (ffc, FF_COLOR, "With flatfield")):
        rel = values / np.median(values)
        edge = np.median(rel[dist > 350])
        ax1.hist(rel, bins=bins, histtype="step", linewidth=2, color=color,
                 label=f"{name}\n  edge nuclei (>350 µm): {edge:.2f} × median")
    ax1.set_xlabel("Background-corrected summed DAPI intensity (× median of all nuclei)")
    ax1.set_ylabel("Number of nuclei")
    ax1.set_title(f"Summed DAPI per nucleus, {len(rows)} nuclei in 25 tiles",
                  loc="left", fontsize=11)
    ax1.legend(frameon=False, loc="upper right")
    ax1.grid(axis="y", color="#e6e5e0", linewidth=0.8)
    ax1.set_axisbelow(True)

    # --- Right: why - intensity vs. position in the field of view ------------
    edges = np.arange(0, dist.max() + 25, 25)
    centers = (edges[:-1] + edges[1:]) / 2
    for values, color, name in ((raw, RAW_COLOR, "Without flatfield"),
                                (ffc, FF_COLOR, "With flatfield")):
        rel = values / np.median(values)
        med = [np.median(rel[(dist >= a) & (dist < b)]) if ((dist >= a) & (dist < b)).sum() >= 10
               else np.nan for a, b in zip(edges[:-1], edges[1:])]
        ax2.plot(centers, med, color=color, linewidth=2, marker="o", markersize=5,
                 markeredgecolor="#fcfcfb", markeredgewidth=1.5, label=name)
    ax2.axhline(1, color="#9c9b96", linewidth=1, linestyle="--")
    ax2.set_xlabel("Distance of nucleus from the image center (µm)")
    ax2.set_ylabel("Median summed DAPI (× overall median)")
    ax2.set_title("Without correction, nuclei at the edge look dimmer", loc="left", fontsize=11)
    ax2.legend(frameon=False, loc="lower left")
    ax2.grid(axis="y", color="#e6e5e0", linewidth=0.8)
    ax2.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(out_png, dpi=150, facecolor=fig.get_facecolor())
    print(f"Saved plot: {out_png}")
    print("\nMedian summed DAPI by distance from the image center (× median of all nuclei):")
    print(f"{'distance (µm)':>14} {'nuclei':>7} {'without FF':>11} {'with FF':>8}")
    for a, b in ((0, 150), (150, 250), (250, 350), (350, 420), (420, 1000)):
        m = (dist >= a) & (dist < b)
        if m.any():
            print(f"{f'{a}-{b}':>14} {m.sum():>7} {np.median(raw[m]) / np.median(raw):>11.2f} "
                  f"{np.median(ffc[m]) / np.median(ffc):>8.2f}")
    print(f"Robust CV of all nuclei - without flatfield: {robust_cv(raw):.1f}%, "
          f"with flatfield: {robust_cv(ffc):.1f}% (mostly cell-cycle: G1 vs. G2 nuclei)")
    plt.show()


def show_in_napari(mi, ci, fi, nuclei, pixel_um):
    import napari

    common = dict(scale=(pixel_um, pixel_um), units=("µm", "µm"))
    viewer = napari.Viewer(title="Flatfield comparison")
    viewer.add_image(mi, name="DAPI raw", colormap="gray",
                     contrast_limits=tuple(np.percentile(mi, [0.5, 99.8])), **common)
    viewer.add_image(ci, name="DAPI flatfield corrected", colormap="gray",
                     contrast_limits=tuple(np.percentile(ci, [0.5, 99.8])), **common)
    viewer.add_image(fi, name="Flatfield (mean = 1)", colormap="magma", **common)
    outlines = viewer.add_labels(nuclei, name="Nuclei", **common)
    outlines.contour = 2
    viewer.scale_bar.visible = True
    napari.run()


if __name__ == "__main__":
    main()
