"""Build a rising sawtooth from sine waves, using the Session 4 Step 2 layout.

Run from the workshop folder:
    uv run src/sawtooth_wave_demo.py
    uv run src/sawtooth_wave_demo.py --waves 10 --output-dir ~/fourier_demo_output

Writes sawtooth_wave_build.gif and a PNG showing the final sum.
"""

import argparse
from pathlib import Path

# Reuse the plotting style from the square-wave lesson (in the same src folder).
from fourier_series_demo import (
    FADED_COLOR, NEW_COLOR, SUM_COLOR, TARGET_COLOR, PERIOD,
    draw_spectrum,
)
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter


def sawtooth_recipe(n_waves):
    """Return frequency, positive amplitude, and sine phase for each wave.

    For a ramp from -1 to +1 over one period, the sine coefficients are
    -2 / (pi * k). A phase of pi (180 degrees) supplies the minus sign
    while keeping the plotted amplitudes positive.
    """
    k = np.arange(1, n_waves + 1)
    return k / PERIOD, 2 / (np.pi * k), np.full(n_waves, np.pi)


def build_sawtooth(n_waves, output_dir):
    """Add one sine wave per animation step and save the final picture."""
    if n_waves < 1:
        raise ValueError("Use at least one sine wave.")
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    freqs, amps, phases = sawtooth_recipe(n_waves)
    # Keep even the fastest sine wave smooth when the wave count is increased.
    time = np.linspace(0, 2 * PERIOD, max(1001, 40 * n_waves + 1))
    target = 2 * ((time / PERIOD) % 1) - 1
    parts = amps[:, None] * np.sin(
        2 * np.pi * freqs[:, None] * time + phases[:, None])
    totals = np.cumsum(parts, axis=0)

    fig = plt.figure(figsize=(12, 6.5))
    grid = fig.add_gridspec(2, 2, width_ratios=[2.2, 1], hspace=0.5, wspace=0.25)
    ax_sum = fig.add_subplot(grid[0, 0])
    ax_parts = fig.add_subplot(grid[1, 0])
    ax_amp = fig.add_subplot(grid[0, 1])
    ax_phase = fig.add_subplot(grid[1, 1])

    def signal_axes(ax, title):
        ax.clear()
        ax.set(xlim=(0, 2 * PERIOD), ylim=(-1.6, 1.6), xlabel="time (s)")
        ax.set_title(title, loc="left", fontsize=11)
        ax.axhline(0, color="#999999", lw=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    def draw(n):
        signal_axes(ax_sum, f"Sum of {n} sine wave{'s' if n != 1 else ''}")
        ax_sum.plot(time, target, color=TARGET_COLOR, lw=1.5, ls="--",
                    label="ideal sawtooth")
        ax_sum.plot(time, totals[n - 1], color=SUM_COLOR, lw=2, label="sum")
        ax_sum.legend(loc="upper left", frameon=False, fontsize=9)
        signal_axes(ax_parts, f"Waves added so far (newest: {freqs[n - 1]:g} Hz)")
        for wave in parts[:n - 1]:
            ax_parts.plot(time, wave, color=FADED_COLOR, lw=1)
        ax_parts.plot(time, parts[n - 1], color=NEW_COLOR, lw=2)
        draw_spectrum(ax_amp, ax_phase, freqs[:n], amps[:n], phases[:n],
                      freqs[-1], [FADED_COLOR] * (n - 1) + [NEW_COLOR])
        ax_amp.set_ylim(0, 0.7)
        ax_phase.set_title("Phase of each sine wave", loc="left", fontsize=11)
        fig.suptitle("Building a rising sawtooth from sine waves", fontsize=13)

    # One second per addition, then hold the complete sum for three seconds.
    frames = list(range(1, n_waves + 1)) + [n_waves] * 2
    anim = FuncAnimation(fig, draw, frames=frames, interval=1000, repeat=True)
    try:
        print(f"Building {n_waves} waves; saving to {output_dir}", flush=True)
        anim.save(output_dir / "sawtooth_wave_build.gif",
                  writer=PillowWriter(fps=1), dpi=80)
        draw(n_waves)
        fig.savefig(output_dir / "sawtooth_wave_build.png", dpi=100)
    finally:
        plt.close(fig)
    print("Saved sawtooth_wave_build.gif and sawtooth_wave_build.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--waves", type=int, default=50,
                        help="Number of sine waves to add (default: 50).")
    parser.add_argument("--output-dir", type=Path,
                        default=Path.home() / "fourier_demo_output",
                        help="Output folder (default: ~/fourier_demo_output).")
    args = parser.parse_args()
    if args.waves < 1:
        parser.error("--waves must be at least 1")
    build_sawtooth(args.waves, args.output_dir)


if __name__ == "__main__":
    main()
