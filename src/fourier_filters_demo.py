"""Low-pass and high-pass filters in Fourier space, with sharp and with soft
("apodized") edges, applied to the noisy nucleus image.

Run it from the top folder of the workshop repository with:

    uv run src/fourier_filters_demo.py

The pictures are saved in Pictures/fourier2d/:

    fourier_filters.png   the four filters: mask, kernel, Fourier image, result
    ringing.png           sharp versus soft low-pass on a sharp-edged test disk
    lowpass_profile.png   a line through the nucleus: sharp versus soft low-pass
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw into files only, no window on screen
import matplotlib.pyplot as plt
import numpy as np
import tifffile

REPO = Path(__file__).resolve().parent.parent
INPUT = REPO / "testImages" / "Nucleus_cropped_noise.tif"
OUT_DIR = REPO / "Pictures" / "fourier2d"

# The same circle as in fourier_lowpass_nucleus.py: 10% of the image size.
DIAMETER_FRACTION = 0.1
# Width of the soft edge, as a fraction of the radius.  The mask goes smoothly
# from 1 to 0 between (1 - EDGE/2) x radius and (1 + EDGE/2) x radius.
EDGE = 1.0

ORIGINAL_COLOR = "#999999"
SHARP_COLOR = "#eb6834"
SOFT_COLOR = "#2a78d6"


def distance_to_center(shape):
    """For every pixel of the (shifted) Fourier image: its distance to the center."""
    rows, cols = shape
    r, c = np.ogrid[:rows, :cols]
    return np.hypot(r - rows // 2, c - cols // 2)


def sharp_lowpass(distance, radius):
    """1 inside the circle, 0 outside."""
    return (distance <= radius).astype(float)


def soft_lowpass(distance, radius):
    """1 well inside the circle, 0 well outside, and in between a smooth step
    shaped like half a cosine wave ("apodization", Greek for "removing the feet")."""
    inner, outer = radius * (1 - EDGE / 2), radius * (1 + EDGE / 2)
    step = np.clip((distance - inner) / (outer - inner), 0, 1)  # 0 ... 1
    return 0.5 * (1 + np.cos(np.pi * step))


def apply(image, mask):
    """Multiply the (shifted) Fourier image by the mask and transform back."""
    fourier = np.fft.fftshift(np.fft.fft2(image)) * mask
    return np.real(np.fft.ifft2(np.fft.ifftshift(fourier))), fourier


def kernel_of(mask, half_size=60):
    """The real-space kernel that does the same as the mask (convolution
    theorem): the inverse Fourier transform of the mask.  Return a small area
    around its center."""
    kernel = np.fft.fftshift(np.real(np.fft.ifft2(np.fft.ifftshift(mask))))
    rows, cols = kernel.shape
    return kernel[rows // 2 - half_size:rows // 2 + half_size + 1,
                  cols // 2 - half_size:cols // 2 + half_size + 1]


def show(ax, data, title, **kwargs):
    ax.imshow(data, interpolation="nearest", **kwargs)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])


def main():
    image = tifffile.imread(INPUT).astype(float)
    distance = distance_to_center(image.shape)
    radius = DIAMETER_FRACTION * min(image.shape) / 2

    low_sharp = sharp_lowpass(distance, radius)
    low_soft = soft_lowpass(distance, radius)
    filters = [
        ("low-pass, sharp edge", low_sharp),
        ("low-pass, soft edge", low_soft),
        ("high-pass, sharp edge", 1 - low_sharp),  # everything the low-pass removes
        ("high-pass, soft edge", 1 - low_soft),
    ]
    results = {name: apply(image, mask) for name, mask in filters}

    # --- Picture 1: one row per filter ----------------------------------------
    full_amp = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(image))))
    amp_range = dict(vmin=0, vmax=full_amp.max())
    fig, axes = plt.subplots(len(filters), 4, figsize=(12, 12.5))
    for row, (name, mask) in enumerate(filters):
        filtered, fourier = results[name]
        kernel = kernel_of(mask)
        # The center of a high-pass kernel is a very large spike; scale the
        # colors to the rest of the kernel, and exaggerate them (x4), so that
        # the weak rings around the center are visible.
        off_center = kernel.copy()
        off_center[kernel.shape[0] // 2, kernel.shape[1] // 2] = 0
        k_max = np.abs(off_center).max() / 4
        show(axes[row, 0], mask, "mask (white = keep)", cmap="gray", vmin=0, vmax=1)
        show(axes[row, 1], kernel, "the same filter as a kernel\n(red +, blue -, colors x4)",
             cmap="RdBu_r", vmin=-k_max, vmax=k_max)
        show(axes[row, 2], np.log1p(np.abs(fourier)), "Fourier image x mask (log)",
             cmap="gray", **amp_range)
        if name.startswith("low"):
            show(axes[row, 3], filtered, "result", cmap="gray",
                 vmin=image.min(), vmax=np.percentile(image, 99.9))
        else:
            # High-pass results are around 0, with positive and negative values.
            limit = np.percentile(np.abs(filtered), 99.5)
            show(axes[row, 3], filtered, "result", cmap="gray", vmin=-limit, vmax=limit)
        axes[row, 0].set_ylabel(name, fontsize=12)
    fig.suptitle(f"Filters in Fourier space (circle diameter: {2 * radius:.0f} pixels)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fourier_filters.png", dpi=90)
    plt.close(fig)
    print("Saved fourier_filters.png")

    # --- Picture 2: ringing at a perfectly sharp edge ---------------------------
    # A bright disk on a dark background, with a razor-sharp edge, the same size
    # as the image.  Real microscope images never have edges this sharp.
    disk = (distance_to_center(image.shape) <= 50).astype(float)
    disk_sharp, _ = apply(disk, low_sharp)
    disk_soft, _ = apply(disk, low_soft)
    fig = plt.figure(figsize=(12, 7.5))
    grid = fig.add_gridspec(2, 3, height_ratios=[1, 0.9])
    for col, (data, title) in enumerate([(disk, "test image: a sharp-edged disk"),
                                         (disk_sharp, "low-pass, sharp edge"),
                                         (disk_soft, "low-pass, soft edge")]):
        show(fig.add_subplot(grid[0, col]), data, title, cmap="gray",
             vmin=-0.15, vmax=1.15)
    ax = fig.add_subplot(grid[1, :])
    center = image.shape[0] // 2
    ax.plot(disk[center], color=ORIGINAL_COLOR, lw=1.5, label="test image")
    ax.plot(disk_sharp[center], color=SHARP_COLOR, lw=2, label="low-pass, sharp edge")
    ax.plot(disk_soft[center], color=SOFT_COLOR, lw=2, label="low-pass, soft edge")
    ax.axhline(0, color="#999999", lw=0.8)
    ax.set_xlim(0, image.shape[1] - 1)
    ax.set_xlabel("column (pixels)")
    ax.set_ylabel("pixel value")
    ax.set_title("Pixel values along the middle row", loc="left", fontsize=11)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, loc="upper right")
    fig.suptitle("Ringing: a sharp-edged mask makes ripples next to sharp edges",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "ringing.png", dpi=90)
    plt.close(fig)
    print("Saved ringing.png")

    # --- Picture 3: a line through the nucleus ---------------------------------
    # Use the row through the brightest part of the sharp low-pass result.
    sharp, _ = results["low-pass, sharp edge"]
    soft, _ = results["low-pass, soft edge"]
    row = np.unravel_index(np.argmax(sharp), sharp.shape)[0]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(image[row], color=ORIGINAL_COLOR, lw=1, label="noisy image")
    ax.plot(sharp[row], color=SHARP_COLOR, lw=2, label="low-pass, sharp edge")
    ax.plot(soft[row], color=SOFT_COLOR, lw=2, label="low-pass, soft edge")
    ax.set_xlim(0, image.shape[1] - 1)
    ax.set_ylim(0, 1.15 * max(sharp[row].max(), soft[row].max()))
    ax.set_xlabel("column (pixels)")
    ax.set_ylabel("pixel value")
    ax.set_title(f"Pixel values along row {row}, through the nucleus", loc="left")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "lowpass_profile.png", dpi=100)
    plt.close(fig)
    print(f"Saved lowpass_profile.png (row {row})")


if __name__ == "__main__":
    main()
