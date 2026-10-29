"""Pictures that introduce the (Discrete) Fourier Transform of 2D images.

Run it from the top folder of the workshop repository with:

    uv run src/fourier_2d_images_demo.py

The pictures are saved in Pictures/fourier2d/:

    stripes.png                  simple stripe images and their Fourier images
    actin_spectrum.png           the actin image, its amplitudes and its phases
    image_build.gif / .png       the actin image rebuilt from more and more waves
    amplitude_phase_swap.png     amplitudes of one image + phases of another
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw into files only, no window on screen
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from matplotlib.animation import FuncAnimation, PillowWriter

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "Pictures" / "fourier2d"
ACTIN = REPO / "testImages" / "Actin_Cropped.tif"
MITO = REPO / "testImages" / "Mito_Cropped.tif"


def fourier(image):
    """The 2D Discrete Fourier Transform of an image, with the zero frequency
    moved from the top left corner to the center ("fftshift"), which is how
    Fourier images are usually shown.  The result is one complex number per
    pixel: its length is the amplitude, its angle the phase of one 2D wave."""
    return np.fft.fftshift(np.fft.fft2(image))


def inverse_fourier(spectrum):
    """Turn a (shifted) Fourier image back into a normal image."""
    return np.real(np.fft.ifft2(np.fft.ifftshift(spectrum)))


def log_amplitude(spectrum):
    """The amplitudes span a huge range (the center is often a million times
    brighter than the rest), so show their logarithm, like a strong gamma."""
    return np.log1p(np.abs(spectrum))


def show(ax, image, title, cmap="gray", **kwargs):
    ax.imshow(image, cmap=cmap, interpolation="nearest", **kwargs)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])


# --- 1. Stripes ---------------------------------------------------------------

def stripes():
    """Images made of one or two 2D waves ("stripes"), and their Fourier images.

    A 2D wave is set by how many times it repeats across the image from left to
    right (kx) and from top to bottom (ky).  Whole numbers of repeats make each
    wave land exactly on one spot of the Fourier image.
    """
    size = 64
    y, x = np.mgrid[0:size, 0:size] / size  # 0 ... 1 across the image

    def wave(kx, ky, phase=0.0):
        return np.cos(2 * np.pi * (kx * x + ky * y) + phase)

    examples = [
        ("3 stripes, left to right", 0.5 + 0.5 * wave(3, 0)),
        ("10 stripes: finer", 0.5 + 0.5 * wave(10, 0)),
        ("tilted stripes", 0.5 + 0.5 * wave(6, 6)),
        ("3 stripes, moved", 0.5 + 0.5 * wave(3, 0, np.pi / 2)),
        ("3 stripes + tilted", 0.5 + 0.25 * wave(3, 0) + 0.25 * wave(6, 6)),
    ]
    fig, axes = plt.subplots(2, len(examples), figsize=(13, 5.6))
    for col, (title, image) in enumerate(examples):
        show(axes[0, col], image, title, vmin=0, vmax=1)
        show(axes[1, col], log_amplitude(fourier(image)), "")
        # Mark the center: the zero frequency.
        axes[1, col].plot(size // 2, size // 2, "+", color="#eb6834", ms=12, mew=1)
    axes[0, 0].set_ylabel("image", fontsize=11)
    axes[1, 0].set_ylabel("Fourier image", fontsize=11)
    fig.suptitle("Each 2D wave (stripe pattern) is a pair of spots in the Fourier "
                 "image (orange + = center = zero frequency)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "stripes.png", dpi=100)
    plt.close(fig)
    print("Saved stripes.png")


# --- 2. The actin image -------------------------------------------------------

def actin_spectrum(actin):
    spectrum = fourier(actin)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8))
    show(axes[0], actin, "Actin_Cropped.tif")
    show(axes[1], log_amplitude(spectrum), "amplitudes (log)")
    show(axes[2], np.degrees(np.angle(spectrum)), "phases (-180 to 180 degrees)",
         cmap="twilight", vmin=-180, vmax=180)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "actin_spectrum.png", dpi=100)
    plt.close(fig)
    print("Saved actin_spectrum.png")


# --- 3. Rebuilding the actin image from its strongest waves --------------------

def image_build(actin):
    """Keep only the N waves with the largest amplitude, set all others to 0,
    and transform back.  N grows from 1 to all of them."""
    spectrum = fourier(actin)
    order = np.argsort(np.abs(spectrum), axis=None)[::-1]  # strongest first
    total = spectrum.size
    counts = [1, 3, 10, 30, 100, 300, 1000, 3000, 10000, 30000, total]
    vmin, vmax = actin.min(), actin.max()

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8))

    def draw(frame):
        n = counts[frame]
        kept = np.zeros(spectrum.size, dtype=bool)
        kept[order[:n]] = True
        kept = kept.reshape(spectrum.shape)
        rebuilt = inverse_fourier(np.where(kept, spectrum, 0))
        for ax in axes:
            ax.clear()
        show(axes[0], np.where(kept, log_amplitude(spectrum), 0),
             f"the {n} strongest waves ({100 * n / total:.2g}% of all)")
        show(axes[1], rebuilt, "image rebuilt from these waves", vmin=vmin, vmax=vmax)
        show(axes[2], actin, "original image", vmin=vmin, vmax=vmax)
        fig.tight_layout()

    anim = FuncAnimation(fig, draw, frames=len(counts))
    anim.save(OUT_DIR / "image_build.gif", writer=PillowWriter(fps=0.8), dpi=80)
    draw(counts.index(1000))
    fig.savefig(OUT_DIR / "image_build.png", dpi=100)
    plt.close(fig)
    print("Saved image_build.gif and image_build.png")


# --- 4. Amplitudes of one image, phases of the other ---------------------------

def amplitude_phase_swap(actin, mito):
    actin_f, mito_f = fourier(actin), fourier(mito)
    # A complex number with a given amplitude and phase: amplitude x e^(i phase).
    actin_amp_mito_phase = inverse_fourier(np.abs(actin_f) * np.exp(1j * np.angle(mito_f)))
    mito_amp_actin_phase = inverse_fourier(np.abs(mito_f) * np.exp(1j * np.angle(actin_f)))

    fig, axes = plt.subplots(1, 4, figsize=(15, 4.4))
    show(axes[0], actin, "actin")
    show(axes[1], mito, "mitochondria")
    show(axes[2], actin_amp_mito_phase, "amplitudes of actin,\nphases of mitochondria")
    show(axes[3], mito_amp_actin_phase, "amplitudes of mitochondria,\nphases of actin")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "amplitude_phase_swap.png", dpi=100)
    plt.close(fig)
    print("Saved amplitude_phase_swap.png")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    actin = tifffile.imread(ACTIN).astype(float)
    mito = tifffile.imread(MITO).astype(float)
    stripes()
    actin_spectrum(actin)
    image_build(actin)
    amplitude_phase_swap(actin, mito)
    print(f"Saved pictures to {OUT_DIR}")


if __name__ == "__main__":
    main()
