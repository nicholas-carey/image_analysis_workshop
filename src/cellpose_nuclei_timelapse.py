"""Segment the nuclei in every frame of a time-lapse movie with Cellpose-SAM.

Data: Tracking/P31-crop-crop.tif - one channel, 86 frames (1 frame per minute), 269 x 258 pixels.

What the script does:
  1. Reads the movie (frames x height x width) and its metadata (pixel size, time between frames).
  2. Runs Cellpose-SAM on each frame separately. Every frame gets its own label image:
     background = 0, nuclei = 1, 2, 3, ...
     NOTE: the numbers are NOT linked between frames yet - nucleus 5 in frame 1 is not
     necessarily nucleus 5 in frame 2. Linking them over time is the tracking step.
     Nuclei touching the image edge are kept, because moving cells enter and leave the field.
  3. Saves all label images as one stack, with the same pixel size and frame interval as the
     movie, plus a table with the number of nuclei per frame.

Outputs (next to the movie, unless --outdir is given):
  <movie>_cellpose_nuclei.tif          label stack (frames x height x width), open in napari or Fiji
  <movie>_cellpose_nuclei_counts.csv   number of nuclei and median nucleus area per frame

Run (from the workshop folder):
    uv run src/cellpose_nuclei_timelapse.py

Look at the result in napari (movie + nuclei; use the slider at the bottom to go through time):
    uv run src/cellpose_nuclei_timelapse.py --show

Re-open napari without running Cellpose again:
    uv run src/cellpose_nuclei_timelapse.py --reuse-masks --show

Useful options if the segmentation is not good:
    --diameter 30          typical nucleus diameter in pixels (default: let Cellpose decide)
    --cellprob-threshold   lower (e.g. -1) finds more / larger nuclei, higher (e.g. 1) fewer / smaller
    --flow-threshold       higher (e.g. 0.6) accepts more irregular shapes (default 0.4)
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import tifffile


def read_movie(path):
    """The movie as a (frames, height, width) array, plus its pixel size and frame interval.

    ImageJ stores the pixel size as a resolution (pixels per unit) in the TIFF tags, and the
    unit and time between frames in its own metadata.
    """
    with tifffile.TiffFile(path) as tif:
        movie = tif.asarray()
        ij = tif.imagej_metadata or {}
        page = tif.pages[0]
        x_res = page.tags["XResolution"].value if "XResolution" in page.tags else (1, 1)
        y_res = page.tags["YResolution"].value if "YResolution" in page.tags else (1, 1)
    if movie.ndim == 2:  # a single frame
        movie = movie[np.newaxis]
    if movie.ndim != 3:
        raise ValueError(f"Expected a single-channel movie (frames, y, x), got shape {movie.shape}")

    unit = ij.get("unit", "pixel")
    if unit in ("micron", "um", "\\u00B5m"):
        unit = "µm"
    pixel_size = (y_res[1] / y_res[0], x_res[1] / x_res[0])  # (y, x), in `unit` per pixel
    if unit == "pixel" or unit == "":
        pixel_size, unit = (1.0, 1.0), "pixel"  # no calibration in the file
    info = dict(pixel_size=pixel_size, unit=unit,
                finterval=float(ij.get("finterval", 1.0)), tunit=ij.get("tunit", "frame"))
    return movie, info


def save_labels(path, labels, info):
    """Save the label stack as an ImageJ TIFF with the movie's pixel size and frame interval.
    16-bit is enough: each frame holds far fewer than 65535 nuclei."""
    py, px = info["pixel_size"]
    metadata = {"axes": "TYX", "finterval": info["finterval"], "tunit": info["tunit"]}
    if info["unit"] != "pixel":
        metadata["unit"] = "um" if info["unit"] == "µm" else info["unit"]
    tifffile.imwrite(path, labels.astype(np.uint16), imagej=True, compression="zlib",
                     resolution=(1 / px, 1 / py), metadata=metadata)


def segment_movie(movie, args):
    """Run Cellpose-SAM on each frame; returns a label stack of the same shape as the movie."""
    from cellpose import models  # slow import, so only when needed

    model = models.CellposeModel(gpu=not args.cpu)
    print(f"Cellpose-SAM running on: {model.device}", flush=True)

    labels = np.zeros(movie.shape, dtype=np.uint16)
    start = time.time()
    for t, frame in enumerate(movie):
        masks, _, _ = model.eval(frame, diameter=args.diameter,
                                 flow_threshold=args.flow_threshold,
                                 cellprob_threshold=args.cellprob_threshold)
        labels[t] = masks
        print(f"  frame {t + 1:>3}/{len(movie)}: {masks.max():>4} nuclei", flush=True)
    print(f"Done in {time.time() - start:.0f} s")
    return labels


def count_per_frame(labels, info):
    """One row per frame: time, number of nuclei, median nucleus area."""
    py, px = info["pixel_size"]
    area_unit = "pixels" if info["unit"] == "pixel" else f"{info['unit']}²"
    rows = []
    for t, frame in enumerate(labels):
        sizes = np.bincount(frame.ravel())[1:]  # pixels per nucleus (skip background 0)
        sizes = sizes[sizes > 0]
        rows.append({"frame": t + 1,
                     f"time_{info['tunit']}": round(t * info["finterval"], 3),
                     "n_nuclei": len(sizes),
                     f"median_area_{area_unit}": round(float(np.median(sizes)) * py * px, 2)
                     if len(sizes) else 0})
    return rows


def show_in_napari(movie, labels, info):
    import napari

    time_unit = {"min": "minute", "s": "second", "sec": "second", "h": "hour"}.get(info["tunit"], "pixel")
    common = dict(scale=(info["finterval"], *info["pixel_size"]),
                  units=(time_unit, info["unit"], info["unit"]))
    viewer = napari.Viewer(title="Cellpose nuclei over time")
    viewer.add_image(movie, name="movie", colormap="gray",
                     contrast_limits=tuple(np.percentile(movie, [0.5, 99.8])), **common)
    layer = viewer.add_labels(labels, name="nuclei (Cellpose)", **common)
    layer.contour = 1  # outlines only, so you can still see the nuclei underneath
    viewer.dims.axis_labels = ("time", "y", "x")
    viewer.canvas.overlays.scale_bar.visible = info["unit"] != "pixel"
    napari.run()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("movie", type=Path, nargs="?", default=Path("Tracking/P31-crop-crop.tif"),
                   help="Time-lapse TIFF (default Tracking/P31-crop-crop.tif).")
    p.add_argument("--outdir", type=Path, help="Output folder (default: the movie's folder).")
    p.add_argument("--diameter", type=float, default=None,
                   help="Nucleus diameter in pixels (default: Cellpose decides).")
    p.add_argument("--flow-threshold", type=float, default=0.4)
    p.add_argument("--cellprob-threshold", type=float, default=0.0)
    p.add_argument("--cpu", action="store_true", help="Run Cellpose on the CPU (slower).")
    p.add_argument("--reuse-masks", action="store_true",
                   help="Load the label stack saved by an earlier run instead of running Cellpose.")
    p.add_argument("--show", action="store_true", help="Open movie and nuclei in napari.")
    args = p.parse_args()

    if not args.movie.is_file():
        p.error(f"{args.movie} not found (run this from the workshop folder).")
    outdir = args.outdir or args.movie.parent
    outdir.mkdir(parents=True, exist_ok=True)
    labels_path = outdir / f"{args.movie.stem}_cellpose_nuclei.tif"
    csv_path = outdir / f"{args.movie.stem}_cellpose_nuclei_counts.csv"

    movie, info = read_movie(args.movie)
    print(f"{args.movie}: {movie.shape[0]} frames of {movie.shape[1]} x {movie.shape[2]} pixels, "
          f"pixel size {info['pixel_size'][1]:.4g} {info['unit']}, "
          f"{info['finterval']:g} {info['tunit']} between frames")

    if args.reuse_masks:
        if not labels_path.is_file():
            p.error(f"{labels_path} not found; run once without --reuse-masks.")
        labels = tifffile.imread(labels_path)
    else:
        labels = segment_movie(movie, args)
        save_labels(labels_path, labels, info)
        print(f"Saved nuclei labels: {labels_path}")

    rows = count_per_frame(labels, info)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    counts = [r["n_nuclei"] for r in rows]
    print(f"Saved counts per frame: {csv_path}")
    print(f"Nuclei per frame: min {min(counts)}, median {np.median(counts):.0f}, max {max(counts)}")

    if args.show:
        show_in_napari(movie, labels, info)


if __name__ == "__main__":
    main()
