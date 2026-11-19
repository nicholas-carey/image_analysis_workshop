"""Track Cellpose-segmented nuclei through a time-lapse movie with btrack.

Input: the movie and the Cellpose label stack made by src/cellpose_nuclei_timelapse.py
       (Tracking/P31-crop-crop.tif and Tracking/P31-crop-crop_cellpose_nuclei.tif).

How btrack works, in two steps:
  1. Frame-to-frame linking. Each nucleus becomes a point (its centroid). A motion model
     (a Kalman filter) predicts where each nucleus will be in the next frame and links it
     to the most likely nucleus there. This gives short pieces of track ("tracklets").
  2. Global optimisation. btrack then looks at all tracklets together and decides, for every
     tracklet end, which explanation is most likely: the nucleus left the image, it was
     missed by the segmentation for a few frames (link to a later tracklet), or it divided
     (link to TWO later tracklets = a mitosis). Two settings limit which tracklets it will
     even consider joining:
       dist_thresh  maximum distance (pixels) between the end of one tracklet and the start
                    of the next
       time_thresh  maximum number of frames between them (the nucleus may be "missing"
                    this long, e.g. during mitosis when the chromatin is hard to segment)

We start from btrack's example "cell_config" settings and then try many combinations of
dist_thresh and time_thresh. There is no hand-made "true" answer for this movie, so each
combination gets a problem score (lower = better):
  + track starts in the middle of the image, after the first frame, without being the
    daughter of a division   (a nucleus cannot appear from nowhere -> probably a broken track)
  + track ends in the middle of the image, before the last frame, without dividing
    (a nucleus cannot vanish -> probably a broken track)
  + jumps: a step within a track, or from a mother to a daughter, longer than --max-jump
    (nuclei move ~2 pixels per frame here, so a big jump means two different nuclei
    were joined - this is what goes wrong when the thresholds are too generous)
Starts/ends within one nucleus diameter of the image edge are fine: nuclei move in and out.
The best (lowest-score) combination is used for the final tracking. If two combinations
score the same, the smaller (more cautious) thresholds win.

Outputs (in the movie's folder, unless --outdir is given):
  <movie>_tracks.csv              one row per nucleus per frame: track_id, frame, time, x, y,
                                  area, parent/root track (for divisions), ...
  <movie>_tracked_nuclei.tif      the Cellpose labels renumbered so each nucleus keeps the
                                  same number (= track_id) in every frame
  <movie>_btrack_optimisation.csv problem score for every dist_thresh / time_thresh combination
  <movie>_btrack_optimisation.png the same as a heatmap

Run (from the workshop folder; needs internet the first time, to download cell_config):
    uv run src/btrack_nuclei_timelapse.py --show

Skip the optimisation and use your own values:
    uv run src/btrack_nuclei_timelapse.py --dist-thresh 30 --time-thresh 5 --show
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import tifffile

# Reuse the movie reader / label writer from the Cellpose script (keeps the pixel size and
# frame interval in the output files).
from cellpose_nuclei_timelapse import read_movie, save_labels

DIST_GRID = (5, 10, 15, 20, 25, 30, 40, 50, 60, 80)  # pixels
TIME_GRID = (1, 2, 3, 4, 5, 6, 8, 10)                # frames


def run_btrack(objects, config_file, image_shape, dist_thresh, time_thresh, search_radius):
    """Track all objects with the cell_config settings, except for the two thresholds.
    Returns the tracks and the same tracks in the format of a napari Tracks layer."""
    import btrack
    import btrack.config

    config = btrack.config.load_config(config_file)
    config.hypothesis_model.dist_thresh = dist_thresh
    config.hypothesis_model.time_thresh = time_thresh

    height, width = image_shape
    with btrack.BayesianTracker(verbose=False) as tracker:
        tracker.configure(config)
        tracker.max_search_radius = search_radius
        tracker.volume = ((0, width), (0, height))  # btrack wants x first, then y
        tracker.append(objects)
        tracker.track()     # step 1: frame-to-frame linking
        # step 2: join tracklets, find divisions (msg_lev: don't print the solver's messages)
        tracker.optimize(tm_lim=60_000, msg_lev="GLP_MSG_OFF")
        tracks = tracker.tracks
        napari_tracks = tracker.to_napari()
    return tracks, napari_tracks


def real_points(track):
    """t, x, y of the frames where the nucleus was actually segmented. btrack also adds
    'dummy' points: its guess of where the nucleus was in frames where it was missing."""
    real = ~np.asarray(track.dummy, dtype=bool)
    return (np.asarray(track.t)[real], np.asarray(track.x)[real], np.asarray(track.y)[real])


def score_tracks(tracks, n_frames, image_shape, edge, max_jump):
    """Count the signs of tracking mistakes (see the explanation at the top)."""
    height, width = image_shape

    def near_edge(x, y):
        return x < edge or y < edge or x > width - 1 - edge or y > height - 1 - edge

    by_id = {trk.ID: trk for trk in tracks}
    broken_starts = broken_ends = jumps = divisions = 0
    for trk in tracks:
        t, x, y = real_points(trk)
        if len(t) == 0:
            continue
        has_mother = not trk.is_root and trk.parent in by_id
        if t[0] > 0 and not has_mother and not near_edge(x[0], y[0]):
            broken_starts += 1
        if t[-1] < n_frames - 1 and not trk.children and not near_edge(x[-1], y[-1]):
            broken_ends += 1
        jumps += int(np.sum(np.hypot(np.diff(x), np.diff(y)) > max_jump))
        if has_mother:  # distance from where the mother was last seen to the daughter
            _, mx, my = real_points(by_id[trk.parent])
            if len(mx) and np.hypot(x[0] - mx[-1], y[0] - my[-1]) > max_jump:
                jumps += 1
        if trk.children:
            divisions += 1
    return dict(n_tracks=len(tracks), divisions=divisions, broken_starts=broken_starts,
                broken_ends=broken_ends, jumps=jumps, score=broken_starts + broken_ends + jumps)


def optimise_thresholds(objects, config_file, image_shape, n_frames, args):
    """Run btrack for every dist_thresh x time_thresh combination; return all results."""
    results = []
    start = time.time()
    for time_thresh in args.time_grid:
        for dist_thresh in args.dist_grid:
            tracks, _ = run_btrack(objects, config_file, image_shape, dist_thresh, time_thresh,
                                   args.search_radius)
            results.append(dict(dist_thresh=dist_thresh, time_thresh=time_thresh,
                                **score_tracks(tracks, n_frames, image_shape, args.edge,
                                               args.max_jump)))
        row = [r for r in results if r["time_thresh"] == time_thresh]
        print(f"  time_thresh {time_thresh:>2}: scores "
              + " ".join(f"{r['score']:>3}" for r in row)
              + f"   (dist_thresh {' '.join(str(d) for d in args.dist_grid)})", flush=True)
    print(f"Optimisation done in {time.time() - start:.0f} s")
    return results


def plot_scores(results, args, best, out_png):
    """Heatmap of the problem score; the chosen combination is outlined in black,
    the cell_config default (dist 40, time 2) with a dashed line."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    dists, times = list(args.dist_grid), list(args.time_grid)
    grid = np.full((len(times), len(dists)), np.nan)
    for r in results:
        grid[times.index(r["time_thresh"]), dists.index(r["dist_thresh"])] = r["score"]

    fig, ax = plt.subplots(figsize=(1.0 + 0.75 * len(dists), 1.4 + 0.6 * len(times)),
                           facecolor="#fcfcfb")
    image = ax.imshow(grid, cmap="Blues", origin="lower", aspect="auto")
    for i in range(len(times)):
        for j in range(len(dists)):
            dark = grid[i, j] > np.nanmin(grid) + 0.6 * (np.nanmax(grid) - np.nanmin(grid))
            ax.text(j, i, f"{grid[i, j]:.0f}", ha="center", va="center", fontsize=9,
                    color="#fcfcfb" if dark else "#0b0b0b")

    def outline(dist, tthresh, **style):
        if dist in dists and tthresh in times:
            ax.add_patch(Rectangle((dists.index(dist) - 0.5, times.index(tthresh) - 0.5), 1, 1,
                                   fill=False, **style))

    outline(40, 2, edgecolor="#52514e", linewidth=1.5, linestyle="--")
    outline(best["dist_thresh"], best["time_thresh"], edgecolor="#0b0b0b", linewidth=2.5)
    ax.set_xticks(range(len(dists)), [str(d) for d in dists])
    ax.set_yticks(range(len(times)), [str(t) for t in times])
    ax.set_xlabel("dist_thresh (pixels)")
    ax.set_ylabel("time_thresh (frames)")
    ax.set_title("Tracking problems (broken starts + broken ends + jumps); lower is better\n"
                 "solid box = chosen, dashed box = cell_config default", loc="left", fontsize=10)
    fig.colorbar(image, ax=ax, label="problem score")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def write_tracks_csv(path, tracks, info):
    """One row per track per frame. Gap frames (nucleus missing, position guessed by btrack)
    are kept and marked with gap_filled = 1."""
    py, px = info["pixel_size"]
    calibrated = info["unit"] != "pixel"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["track_id", "frame", f"time_{info['tunit']}", "x_px", "y_px"]
        if calibrated:
            header += [f"x_{info['unit']}", f"y_{info['unit']}"]
        header += ["area_px", "gap_filled", "mother_id", "root_id", "generation",
                   "daughter_ids", "fate"]
        writer.writerow(header)
        for trk in sorted(tracks, key=lambda trk: trk.ID):
            area = trk.properties.get("area", np.full(len(trk), np.nan))
            mother = 0 if trk.is_root else trk.parent
            daughters = " ".join(str(c) for c in trk.children)
            for t, x, y, a, dummy in zip(trk.t, trk.x, trk.y, area, trk.dummy):
                row = [trk.ID, int(t), round(t * info["finterval"], 3), round(x, 2), round(y, 2)]
                if calibrated:
                    row += [round(x * px, 3), round(y * py, 3)]
                row += ["" if dummy or np.isnan(a) else int(a), int(dummy), mother, trk.root,
                        trk.generation, daughters, trk.fate.name]
                writer.writerow(row)


def show_in_napari(movie, labels, tracked_labels, napari_tracks, info):
    import napari

    time_unit = {"min": "minute", "s": "second", "sec": "second", "h": "hour"}.get(info["tunit"], "pixel")
    common = dict(scale=(info["finterval"], *info["pixel_size"]),
                  units=(time_unit, info["unit"], info["unit"]))
    viewer = napari.Viewer(title="btrack nuclei tracks")
    viewer.add_image(movie, name="movie", colormap="gray",
                     contrast_limits=tuple(np.percentile(movie, [0.5, 99.8])), **common)
    cellpose = viewer.add_labels(labels, name="nuclei (Cellpose, numbered per frame)",
                                 visible=False, **common)
    cellpose.contour = 1
    tracked = viewer.add_labels(tracked_labels, name="nuclei (colored by track)", opacity=0.4,
                                **common)
    tracked.contour = 0
    data, properties, graph = napari_tracks
    # Tracks are colored by their "root": a mother and her daughters share a color.
    viewer.add_tracks(data, properties=properties, graph=graph, name="tracks",
                      color_by="root", tail_length=20, tail_width=2, **common)
    viewer.dims.axis_labels = ("time", "y", "x")
    viewer.canvas.overlays.scale_bar.visible = info["unit"] != "pixel"
    napari.run()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("movie", type=Path, nargs="?", default=Path("Tracking/P31-crop-crop.tif"))
    p.add_argument("--labels", type=Path,
                   help="Cellpose label stack (default: <movie>_cellpose_nuclei.tif).")
    p.add_argument("--outdir", type=Path, help="Output folder (default: the movie's folder).")
    p.add_argument("--dist-thresh", type=float,
                   help="Use this dist_thresh (pixels) instead of optimising.")
    p.add_argument("--time-thresh", type=float,
                   help="Use this time_thresh (frames) instead of optimising.")
    p.add_argument("--dist-grid", type=float, nargs="+", default=DIST_GRID,
                   help=f"dist_thresh values to try (default {' '.join(map(str, DIST_GRID))}).")
    p.add_argument("--time-grid", type=float, nargs="+", default=TIME_GRID,
                   help=f"time_thresh values to try (default {' '.join(map(str, TIME_GRID))}).")
    p.add_argument("--search-radius", type=float, default=50,
                   help="Frame-to-frame linking only looks this far (pixels) (default 50).")
    p.add_argument("--edge", type=float,
                   help="Starts/ends this close to the image edge (pixels) are OK "
                        "(default: one nucleus diameter).")
    p.add_argument("--max-jump", type=float,
                   help="A step longer than this (pixels) counts as a wrong link "
                        "(default: one nucleus diameter).")
    p.add_argument("--show", action="store_true", help="Open movie, nuclei and tracks in napari.")
    args = p.parse_args()

    labels_path = args.labels or args.movie.with_name(f"{args.movie.stem}_cellpose_nuclei.tif")
    for path in (args.movie, labels_path):
        if not path.is_file():
            p.error(f"{path} not found. Run this from the workshop folder, after "
                    "src/cellpose_nuclei_timelapse.py.")
    outdir = args.outdir or args.movie.parent
    outdir.mkdir(parents=True, exist_ok=True)

    movie, info = read_movie(args.movie)
    labels = tifffile.imread(labels_path)
    n_frames, image_shape = labels.shape[0], labels.shape[1:]

    import logging

    import btrack  # slow import, so after the quick checks above
    import btrack.datasets
    import btrack.utils

    # btrack prints many messages for every run; only show warnings and errors.
    logging.getLogger("btrack").setLevel(logging.WARNING)

    # One point per nucleus per frame, with its area (btrack does not look at the pixels).
    objects = btrack.utils.segmentation_to_objects(labels, properties=("area",))
    # Nucleus areas, frame by frame (Cellpose numbers the nuclei anew in every frame).
    areas = np.concatenate([np.bincount(frame.ravel())[1:] for frame in labels])
    diameter = 2 * np.sqrt(np.median(areas[areas > 0]) / np.pi)
    args.edge = args.edge if args.edge is not None else round(diameter, 1)
    args.max_jump = args.max_jump if args.max_jump is not None else round(diameter, 1)
    print(f"{len(objects)} nuclei in {n_frames} frames; typical nucleus diameter "
          f"{diameter:.0f} pixels (edge margin {args.edge}, max jump {args.max_jump} pixels)")

    config_file = btrack.datasets.cell_config()  # btrack's example settings for cells
    stem = args.movie.stem

    if args.dist_thresh is not None or args.time_thresh is not None:
        import btrack.config

        cell_config = btrack.config.load_config(config_file).hypothesis_model
        best = dict(dist_thresh=args.dist_thresh or cell_config.dist_thresh,
                    time_thresh=args.time_thresh or cell_config.time_thresh)
        print(f"Using dist_thresh {best['dist_thresh']:g}, time_thresh {best['time_thresh']:g}")
    else:
        print(f"Trying {len(args.dist_grid) * len(args.time_grid)} combinations of "
              "dist_thresh and time_thresh ...")
        results = optimise_thresholds(objects, config_file, image_shape, n_frames, args)
        best = min(results, key=lambda r: (r["score"], r["dist_thresh"], r["time_thresh"]))
        default = next((r for r in results if r["dist_thresh"] == 40 and r["time_thresh"] == 2),
                       None)

        with open(outdir / f"{stem}_btrack_optimisation.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        plot_scores(results, args, best, outdir / f"{stem}_btrack_optimisation.png")
        print(f"Saved optimisation table and heatmap: {outdir / (stem + '_btrack_optimisation')}"
              ".csv / .png")

        print(f"\n{'':24}{'tracks':>8}{'divisions':>11}{'broken starts':>15}"
              f"{'broken ends':>13}{'jumps':>7}{'score':>7}")
        for name, r in (("cell_config (40, 2)", default),
                        (f"best ({best['dist_thresh']:g}, {best['time_thresh']:g})", best)):
            if r:
                print(f"{name:24}{r['n_tracks']:>8}{r['divisions']:>11}{r['broken_starts']:>15}"
                      f"{r['broken_ends']:>13}{r['jumps']:>7}{r['score']:>7}")

    # --- Final tracking with the chosen thresholds -----------------------------------------
    tracks, napari_tracks = run_btrack(objects, config_file, image_shape, best["dist_thresh"],
                                       best["time_thresh"], args.search_radius)
    final = score_tracks(tracks, n_frames, image_shape, args.edge, args.max_jump)
    print(f"\nFinal tracking: {final['n_tracks']} tracks, {final['divisions']} divisions")

    tracks_csv = outdir / f"{stem}_tracks.csv"
    write_tracks_csv(tracks_csv, tracks, info)
    print(f"Saved tracks: {tracks_csv}")

    # Labels renumbered by track ID: the same nucleus has the same number in every frame.
    tracked_labels = btrack.utils.update_segmentation(labels, tracks, color_by="ID")
    tracked_path = outdir / f"{stem}_tracked_nuclei.tif"
    save_labels(tracked_path, tracked_labels, info)
    print(f"Saved labels numbered by track: {tracked_path}")

    if args.show:
        show_in_napari(movie, labels, tracked_labels, napari_tracks, info)


if __name__ == "__main__":
    main()
