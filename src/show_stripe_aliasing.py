"""Show sampled vertical stripes and their Fourier amplitudes in napari.

Run from the project folder: uv run src/show_stripe_aliasing.py
A stripe cycle means one bright band plus one dark band.
"""

import numpy as np

SIZE = 64
CYCLES = (30, 40)


def make_stripes(cycles):
    """Sample a horizontal cosine, then repeat it down every row."""
    x = np.arange(SIZE)
    row = 0.5 + 0.5 * np.cos(2 * np.pi * cycles * x / SIZE)
    return np.tile(row, (SIZE, 1))


def fourier_amplitude(pixels):
    """Center zero frequency and use a logarithm to reveal weaker amplitudes."""
    return np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(pixels))))


def main():
    import napari

    viewer = napari.Viewer(title="64 x 64 stripes: sampling and Fourier amplitude")
    # Grid mode separates the four layers into individual panels.
    images = [make_stripes(cycles) for cycles in CYCLES]
    spectra = [fourier_amplitude(pixels) for pixels in images]
    spectrum_max = max(float(spectrum.max()) for spectrum in spectra)
    for cycles, pixels, spectrum in zip(CYCLES, images, spectra):
        metadata = {
            "requested_cycles_across_image": cycles,
            "pixel_size_yx": (1, 1),
            "pixel_size_unit": "pixel",
            "pattern": "0.5 + 0.5*cos(2*pi*cycles*x/64)",
        }
        viewer.add_image(
            pixels, name=f"{cycles} cycles / 64 pixels — vertical stripes",
            colormap="gray", contrast_limits=(0, 1),
            interpolation2d="nearest", metadata=metadata,
        )
        viewer.add_image(
            spectrum, name=f"{cycles} cycles — log Fourier amplitude",
            colormap="gray", contrast_limits=(0, spectrum_max),
            interpolation2d="nearest",
            metadata={
                "source_metadata": metadata,
                "frequency_step_yx": (1 / SIZE, 1 / SIZE),
                "frequency_unit": "cycles/pixel",
                "zero_frequency_index_yx": (SIZE // 2, SIZE // 2),
                "amplitude_transform": "ln(1 + abs(fftshift(fft2(image))))",
            },
        )
    viewer.grid.enabled = True
    viewer.grid.shape = (2, 2)
    viewer.reset_view()
    print("Each image is 64 x 64 pixels; one cycle = one bright/dark pair.")
    print("30 cycles: Fourier peaks at +/-30 cycles per image, plus the mean at zero.")
    print("40 cycles: aliases to 24 cycles, with Fourier peaks at +/-24.")
    print("64 pixels cannot resolve more than 32 cycles per image (Nyquist limit).")
    napari.run()


if __name__ == "__main__":
    main()
