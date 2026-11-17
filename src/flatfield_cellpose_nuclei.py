"""Flatfield-correct all 25 tiles, segment nuclei with Cellpose-SAM, measure them,
and compare with the same analysis on the uncorrected (raw) images.

Data: 3Channel_Raw_25Pos_1 (5 x 5 tiles), 20x objective, channels Cy3, FITC, DAPI.
Flatfield images: Flatfield/20x/<channel>-20x.tif; camera dark image: Flatfield/Dark.tif.

Steps, for every tile:
  1. Flatfield correction of all 3 channels (see Session6 README):
         CI = (MI - BI) / FI
       MI = measured image, BI = dark image (camera offset),
       FI = (flatfield - dark) divided by its own mean, so it averages 1.
     The corrected tile is saved as an OME-TIFF (same channels and pixel size).
  2. Segment nuclei with Cellpose-SAM on the DAPI channel - TWICE:
     once in the raw image and once in the corrected image. Each segmentation
     is independent, so we see whether the correction changes which nuclei are
     found as well as how bright they are.
     Nuclei touching the tile edge (cut off) are removed.
  3. Per nucleus, in the image it was found in:
       area (µm²)
       local background = median of a ring of pixels around the nucleus
       background-corrected summed DAPI = sum of pixels - background x area (pixels)

Outputs (in --outdir, default 3Channel_Raw_25Pos_1/flatfield_cellpose):
  corrected/<tile>_flatfield.ome.tif     flatfield-corrected tiles (3 channels)
  masks/<tile>_{raw,flatfield}_nuclei.tif  Cellpose nucleus masks (reused with --reuse-masks)
  nuclei_cellpose_raw_and_flatfield.csv  one row per nucleus; "condition" column = raw / flatfield
  nuclei_cellpose_comparison.png         histogram of summed DAPI + raw-vs-corrected comparison

Run (from the workshop folder; ~10 minutes on the Mac GPU, 50 Cellpose runs):
    uv run src/flatfield_cellpose_nuclei.py

Re-make the plots and table without running Cellpose again:
    uv run src/flatfield_cellpose_nuclei.py --reuse-masks

Look at one tile in napari (raw, corrected, flatfield, both sets of nuclei):
    uv run src/flatfield_cellpose_nuclei.py --reuse-masks --show Pos-1-002_002
"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage as ndi
from skimage.measure import regionprops_table
from skimage.segmentation import clear_border, relabel_sequential

# Reuse the background-ring measurement from the earlier (Otsu) flatfield script.
from flatfield_compare_nuclei import measure, robust_cv

RAW_COLOR, FF_COLOR = "#2a78d6", "#eb6834"  # blue = raw, orange = flatfield corrected


def read_tile(path):
    """All channels of one Micro-Manager tile: {name: 2D image}, and the pixel size (µm).

    Each file holds the channels of one position as separate pages. Channel names
    are in the sidecar *_metadata.txt (Summary -> ChNames); pixel size is in the page tags.
    """
    meta = json.loads(path.with_name(path.name.replace(".ome.tif", "_metadata.txt")).read_text())
    names = meta["Summary"]["ChNames"]
    with tifffile.TiffFile(path) as tif:
        channels = {name: tif.pages[i].asarray().astype(np.float64) for i, name in enumerate(names)}
        pixel_um = float(tif.pages[0].tags["MicroManagerMetadata"].value["PixelSizeUm"])
    return channels, pixel_um


def save_corrected(path, channels, pixel_um):
    """Save corrected channels as a 16-bit OME-TIFF with channel names and pixel size.
    Values are rounded and clipped to 0-65535 (the corrected image keeps the raw
    image's mean, so it fits the same 16-bit range)."""
    stack = np.stack([np.clip(np.round(img), 0, 65535) for img in channels.values()]).astype(np.uint16)
    tifffile.imwrite(path, stack, ome=True, compression="zlib", metadata={
        "axes": "CYX", "Channel": {"Name": list(channels)},
        "PhysicalSizeX": pixel_um, "PhysicalSizeXUnit": "µm",
        "PhysicalSizeY": pixel_um, "PhysicalSizeYUnit": "µm"})


def save_mask(path, nuclei, pixel_um):
    pixels_per_cm = (1e4 / pixel_um, 1e4 / pixel_um)
    tifffile.imwrite(path, nuclei.astype(np.uint16), compression="zlib",
                     resolution=pixels_per_cm, resolutionunit="CENTIMETER", metadata=None)


def segment(model, dapi):
    """Cellpose-SAM nuclei, without those cut off by the tile edge, numbered 1, 2, 3, ..."""
    nuclei, _, _ = model.eval(dapi)
    return relabel_sequential(clear_border(nuclei))[0]


def measure_nuclei(image, nuclei, pixel_um, fi, args):
    """One dict per nucleus: area, position, background and background-corrected sum."""
    props = regionprops_table(nuclei, properties=("label", "area", "centroid"))
    ids = props["label"]
    area_um2 = props["area"] * pixel_um**2
    keep = area_um2 >= args.min_area
    ids, area_um2 = ids[keep], area_um2[keep]
    cy, cx = props["centroid-0"][keep], props["centroid-1"][keep]
    bg_sum, bg = measure(image, nuclei, ids, args.ring_gap, args.ring_width)
    dist_um = np.hypot(cy - (image.shape[0] - 1) / 2, cx - (image.shape[1] - 1) / 2) * pixel_um
    return [dict(nucleus=int(i), area_um2=round(float(a), 2),
                 centroid_x_um=round(float(x * pixel_um), 1), centroid_y_um=round(float(y * pixel_um), 1),
                 dist_from_center_um=round(float(d), 1),
                 flatfield_value=round(float(fi[int(y), int(x)]), 4),
                 background=round(float(b), 1), bgcorr_sum_dapi=round(float(s), 1))
            for i, a, x, y, d, b, s in zip(ids, area_um2, cx, cy, dist_um, bg, bg_sum)]


def match_fraction(nuclei_a, nuclei_b, iou=0.5):
    """Fraction of nuclei in A that have a partner in B overlapping by IoU >= 0.5."""
    if nuclei_a.max() == 0:
        return np.nan
    a, b = nuclei_a.ravel(), nuclei_b.ravel()
    both = (a > 0) & (b > 0)
    # Overlap (pixels) of every A-B pair, and the size of each nucleus
    pairs, overlap = np.unique(np.stack([a[both], b[both]]), axis=1, return_counts=True)
    size_a, size_b = np.bincount(a), np.bincount(b)
    pair_iou = overlap / (size_a[pairs[0]] + size_b[pairs[1]] - overlap)
    return len(set(pairs[0][pair_iou >= iou])) / nuclei_a.max()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tiles", type=Path, default=Path("3Channel_Raw_25Pos_1"))
    p.add_argument("--flatfield-dir", type=Path, default=Path("Flatfield/20x"),
                   help="Folder with <channel>-20x.tif flatfield images (default Flatfield/20x).")
    p.add_argument("--dark", type=Path, default=Path("Flatfield/Dark.tif"))
    p.add_argument("--outdir", type=Path, help="Output folder (default: <tiles>/flatfield_cellpose)")
    p.add_argument("--min-area", type=float, default=20, help="Smallest nucleus, µm² (default 20).")
    p.add_argument("--ring-gap", type=int, default=4, help="Background ring starts this many px out.")
    p.add_argument("--ring-width", type=int, default=15, help="Background ring ends this many px out.")
    p.add_argument("--reuse-masks", action="store_true",
                   help="Load nucleus masks saved by an earlier run instead of running Cellpose.")
    p.add_argument("--cpu", action="store_true", help="Run Cellpose on the CPU (much slower).")
    p.add_argument("--show", metavar="POSITION", help="Open this tile in napari, e.g. Pos-1-002_002.")
    args = p.parse_args()

    tiles = sorted(args.tiles.glob("*.ome.tif"))
    if not tiles:
        p.error(f"No .ome.tif files in {args.tiles}")
    outdir = args.outdir or args.tiles / "flatfield_cellpose"
    for sub in ("corrected", "masks"):
        (outdir / sub).mkdir(parents=True, exist_ok=True)

    # --- Flatfield images FI (one per channel), normalized to mean 1 ------------
    dark = tifffile.imread(args.dark).astype(np.float64)
    channel_names = list(read_tile(tiles[0])[0])
    flatfields = {}
    for name in channel_names:
        ff_path = args.flatfield_dir / f"{name}-20x.tif"
        if not ff_path.is_file():
            p.error(f"No flatfield image for channel {name}: {ff_path}")
        ff = tifffile.imread(ff_path).astype(np.float64) - dark
        flatfields[name] = ff / ff.mean()
        print(f"Flatfield {name}: edge-to-center illumination drop "
              f"~{(1 - flatfields[name].min() / flatfields[name].max()) * 100:.0f}%")

    model = None
    if not args.reuse_masks:
        from cellpose import models  # slow import, so only when needed
        model = models.CellposeModel(gpu=not args.cpu)
        print(f"Cellpose-SAM running on: {model.device}\n", flush=True)

    rows, tile_stats = [], []
    for path in tiles:
        position = path.name.split("MMStack_")[1].removesuffix(".ome.tif")
        start = time.time()
        raw, pixel_um = read_tile(path)

        # --- 1. Flatfield correction of every channel --------------------------
        corrected = {name: (img - dark) / flatfields[name] for name, img in raw.items()}
        save_corrected(outdir / "corrected" / f"{path.name.removesuffix('.ome.tif')}_flatfield.ome.tif",
                       corrected, pixel_um)

        # --- 2. Cellpose nuclei, in raw and corrected DAPI ----------------------
        masks = {}
        for condition, image in (("raw", raw["DAPI"]), ("flatfield", corrected["DAPI"])):
            mask_path = outdir / "masks" / f"{position}_{condition}_nuclei.tif"
            if args.reuse_masks:
                if not mask_path.is_file():
                    p.error(f"{mask_path} not found; run once without --reuse-masks.")
                masks[condition] = tifffile.imread(mask_path)
            else:
                masks[condition] = segment(model, image)
                save_mask(mask_path, masks[condition], pixel_um)

        # --- 3. Measure, each in the image its nuclei were found in --------------
        # The raw image still contains the camera offset; the local background
        # subtraction removes it, so raw and corrected sums are comparable.
        fi = flatfields["DAPI"]
        counts = {}
        for condition, image in (("raw", raw["DAPI"]), ("flatfield", corrected["DAPI"])):
            measured = measure_nuclei(image, masks[condition], pixel_um, fi, args)
            counts[condition] = len(measured)
            rows += [dict(condition=condition, position=position, **r) for r in measured]
        matched = match_fraction(masks["flatfield"], masks["raw"])
        tile_stats.append(dict(position=position, raw=counts["raw"],
                               flatfield=counts["flatfield"], matched=matched))
        print(f"{position}: {counts['raw']:>4} nuclei raw, {counts['flatfield']:>4} flatfield; "
              f"{matched * 100:.0f}% of flatfield nuclei also found in raw  "
              f"({time.time() - start:.0f} s)", flush=True)

        if args.show == position:
            show_in_napari(raw["DAPI"], corrected["DAPI"], fi, masks, pixel_um)

    csv_path = outdir / "nuclei_cellpose_raw_and_flatfield.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {len(rows)} rows: {csv_path}")
    print(f"Saved corrected tiles: {outdir / 'corrected'}")

    summarize(rows, tile_stats)
    plot(rows, tile_stats, outdir / "nuclei_cellpose_comparison.png")


def column(rows, condition, key):
    return np.array([r[key] for r in rows if r["condition"] == condition])


def summarize(rows, tile_stats):
    print("\n=== Raw vs. flatfield-corrected (Cellpose-SAM nuclei) ===")
    print(f"{'':32}{'raw':>12}{'flatfield':>12}")
    stats = {}
    for cond in ("raw", "flatfield"):
        s, a, d = (column(rows, cond, k) for k in ("bgcorr_sum_dapi", "area_um2", "dist_from_center_um"))
        stats[cond] = [len(s), np.median(a), np.median(s), robust_cv(s),
                       np.median(s[d > 350]) / np.median(s[d < 150])]
    labels = ["Nuclei (not touching edge)", "Median area (µm²)", "Median summed DAPI",
              "Robust CV of summed DAPI (%)", "Edge / center summed DAPI"]
    formats = ["{:>12.0f}", "{:>12.1f}", "{:>12.4g}", "{:>12.1f}", "{:>12.2f}"]
    for k, (label, fmt) in enumerate(zip(labels, formats)):
        print(f"{label:32}" + fmt.format(stats["raw"][k]) + fmt.format(stats["flatfield"][k]))
    print(f"{'Nuclei found in both (IoU≥0.5)':32}"
          f"{np.nanmean([t['matched'] for t in tile_stats]) * 100:>23.1f}%")
    print("  'Edge / center': median of nuclei >350 µm from the image center divided by "
          "nuclei <150 µm; 1.00 = no position effect.")


def plot(rows, tile_stats, out_png):
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b",
                         "xtick.color": "#52514e", "ytick.color": "#52514e"})
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), facecolor="#fcfcfb")
    (ax_hist, ax_dist), (ax_area, ax_count) = axes
    conditions = (("raw", RAW_COLOR, "Without flatfield"), ("flatfield", FF_COLOR, "With flatfield"))

    # --- Histogram of background-corrected summed DAPI (flatfield) ---------------
    # Main result: filled histogram of the corrected data, raw data as an outline.
    ff_sum = column(rows, "flatfield", "bgcorr_sum_dapi")
    bins = np.linspace(0, np.percentile(ff_sum, 99.5) * 1.1, 70)
    ax_hist.hist(ff_sum, bins=bins, color=FF_COLOR, alpha=0.75, label=f"With flatfield (n={len(ff_sum)})")
    raw_sum = column(rows, "raw", "bgcorr_sum_dapi")
    ax_hist.hist(raw_sum, bins=bins, histtype="step", linewidth=1.8, color=RAW_COLOR,
                 label=f"Without flatfield (n={len(raw_sum)})")
    ax_hist.set_xlabel("Background-corrected summed DAPI intensity per nucleus")
    ax_hist.set_ylabel("Number of nuclei")
    ax_hist.set_title("Summed DAPI per nucleus (Cellpose-SAM, 25 tiles)", loc="left", fontsize=11)
    ax_hist.legend(frameon=False)

    # --- Intensity vs. distance from the image center ----------------------------
    for cond, color, name in conditions:
        s, d = column(rows, cond, "bgcorr_sum_dapi"), column(rows, cond, "dist_from_center_um")
        edges = np.arange(0, d.max() + 25, 25)
        med = [np.median(s[(d >= a) & (d < b)]) / np.median(s) if ((d >= a) & (d < b)).sum() >= 10
               else np.nan for a, b in zip(edges[:-1], edges[1:])]
        ax_dist.plot((edges[:-1] + edges[1:]) / 2, med, color=color, linewidth=2, marker="o",
                     markersize=5, markeredgecolor="#fcfcfb", label=name)
    ax_dist.axhline(1, color="#9c9b96", linewidth=1, linestyle="--")
    ax_dist.set_xlabel("Distance of nucleus from the image center (µm)")
    ax_dist.set_ylabel("Median summed DAPI (× overall median)")
    ax_dist.set_title("Position effect: edge nuclei look dimmer without correction",
                      loc="left", fontsize=11)
    ax_dist.legend(frameon=False, loc="lower left")

    # --- Nucleus area ---------------------------------------------------------------
    area_bins = np.linspace(0, np.percentile(column(rows, "flatfield", "area_um2"), 99.5) * 1.1, 60)
    for cond, color, name in conditions:
        ax_area.hist(column(rows, cond, "area_um2"), bins=area_bins, histtype="step",
                     linewidth=1.8, color=color, label=name)
    ax_area.set_xlabel("Nucleus area (µm²)")
    ax_area.set_ylabel("Number of nuclei")
    ax_area.set_title("Nucleus size: segmentation barely changes", loc="left", fontsize=11)
    ax_area.legend(frameon=False)

    # --- Nuclei per tile: raw vs. corrected ------------------------------------------
    n_raw = np.array([t["raw"] for t in tile_stats])
    n_ff = np.array([t["flatfield"] for t in tile_stats])
    lim = (0, max(n_raw.max(), n_ff.max()) * 1.08)
    ax_count.plot(lim, lim, color="#9c9b96", linewidth=1, linestyle="--")
    ax_count.scatter(n_raw, n_ff, s=40, color=FF_COLOR, edgecolor="#fcfcfb")
    ax_count.set_xlim(lim)
    ax_count.set_ylim(lim)
    ax_count.set_xlabel("Nuclei per tile, without flatfield")
    ax_count.set_ylabel("Nuclei per tile, with flatfield")
    ax_count.set_title("Nuclei found per tile (dashed: equal)", loc="left", fontsize=11)

    for ax in axes.ravel():
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#e6e5e0", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, facecolor=fig.get_facecolor())
    print(f"\nSaved plot: {out_png}")
    plt.show()


def show_in_napari(raw_dapi, ff_dapi, fi, masks, pixel_um):
    import napari

    common = dict(scale=(pixel_um, pixel_um), units=("µm", "µm"))
    viewer = napari.Viewer(title="Flatfield + Cellpose comparison")
    viewer.add_image(raw_dapi, name="DAPI raw", colormap="gray",
                     contrast_limits=tuple(np.percentile(raw_dapi, [0.5, 99.8])), **common)
    viewer.add_image(ff_dapi, name="DAPI flatfield corrected", colormap="gray",
                     contrast_limits=tuple(np.percentile(ff_dapi, [0.5, 99.8])), **common)
    viewer.add_image(fi, name="DAPI flatfield (mean = 1)", colormap="magma", visible=False, **common)
    for condition in ("raw", "flatfield"):
        layer = viewer.add_labels(masks[condition], name=f"Nuclei ({condition})", **common)
        layer.contour = 2
    viewer.canvas.overlays.scale_bar.visible = True
    napari.run()


if __name__ == "__main__":
    main()
