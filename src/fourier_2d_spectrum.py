"""Show the "recipe" of a square wave as one picture: frequency versus phase,
with the brightness of each spot showing the amplitude of that sine wave.

Run it from the top folder of the workshop repository with:

    uv run src/fourier_2d_spectrum.py

Two pictures are saved in Pictures/fourier/:

    spectrum_2d_square.png          the square wave from fourier_series_demo.py
    spectrum_2d_square_shifted.png  the same square wave, a quarter second later
"""

import matplotlib

matplotlib.use("Agg")  # draw into files only, no window on screen
import matplotlib.pyplot as plt
import numpy as np

# Re-use the square wave and its recipe from the animation script (in the same
# "src" folder), so both always describe exactly the same signal.
from fourier_series_demo import (OUT_DIR, PERIOD, SUM_COLOR, TARGET_COLOR, TIME,
                                 sine, square_wave, square_wave_recipe,
                                 style_signal_axes)

N_WAVES = 10
MAX_FREQ = 20  # Hz, the highest frequency shown
PHASE_STEP = 15  # degrees per row of the picture


def recipe_image(freqs, amps, phases):
    """Make an image with one column per frequency (0, 1, 2, ... Hz) and one row
    per phase (-180 to 180 degrees, in steps of PHASE_STEP).  Each sine wave
    lights up one pixel, as bright as its amplitude; all other pixels are 0."""
    phase_rows = np.arange(-180, 180 + PHASE_STEP, PHASE_STEP)
    image = np.zeros((len(phase_rows), MAX_FREQ + 1))
    for f, a, p in zip(freqs, amps, phases):
        row = int(np.round((np.degrees(p) + 180) / PHASE_STEP))
        image[row, int(round(f))] = a
    return image


def draw(shift, title, filename):
    freqs, amps, phases = square_wave_recipe(N_WAVES, shift)
    fig, (ax_sig, ax_img) = plt.subplots(
        2, 1, figsize=(8, 8), gridspec_kw={"height_ratios": [1, 2.2]})

    # Top: the signal itself (the sum of the waves) and the ideal square wave.
    total = sum(sine(TIME, f, a, p) for f, a, p in zip(freqs, amps, phases))
    style_signal_axes(ax_sig, f"Signal: sum of {N_WAVES} sine waves")
    ax_sig.plot(TIME, square_wave(TIME, shift), color=TARGET_COLOR, lw=1.5,
                ls="--", label="square wave")
    ax_sig.plot(TIME, total, color=SUM_COLOR, lw=2, label="sum")
    ax_sig.legend(loc="lower right", bbox_to_anchor=(1, 1), ncol=2,
                  frameon=False, fontsize=9)

    # Bottom: the recipe as an image.  "extent" puts the pixel centers at the
    # right frequency and phase on the axes.
    image = recipe_image(freqs, amps, phases)
    half_row = PHASE_STEP / 2
    shown = ax_img.imshow(image, cmap="gray", origin="lower", aspect="auto",
                          interpolation="nearest", vmin=0, vmax=amps.max(),
                          extent=(-0.5, MAX_FREQ + 0.5,
                                  -180 - half_row, 180 + half_row))
    ax_img.set_xlabel("frequency (Hz)")
    ax_img.set_ylabel("phase (degrees)")
    ax_img.set_yticks([-180, -90, 0, 90, 180])
    ax_img.set_xticks(range(0, MAX_FREQ + 1, 2))
    ax_img.set_title("Recipe: one spot per sine wave, brightness = amplitude",
                     loc="left", fontsize=11)
    fig.colorbar(shown, ax=ax_img, label="amplitude", shrink=0.9)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_DIR / filename, dpi=100)
    plt.close(fig)
    print(f"Saved {filename}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    draw(0.0, "Square wave", "spectrum_2d_square.png")
    draw(PERIOD / 4, "The same square wave, moved 0.25 s later in time",
         "spectrum_2d_square_shifted.png")


if __name__ == "__main__":
    main()
