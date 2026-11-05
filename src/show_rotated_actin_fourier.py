"""Compare original actin, a 45-degree rotation, and its Fourier amplitude.

Run from the project folder:
    uv run src/show_rotated_actin_fourier.py
"""

from pathlib import Path

import numpy as np
from scipy.ndimage import rotate

# Reuse the TIFF calibration reader and Fourier calculation from our first demo.
from show_mito_fourier import load_image_and_metadata, log_fourier_amplitude

IMAGE = Path(__file__).resolve().parents[1] / "testImages" / "Actin_cropped.tif"


def rotate_actin(pixels):
    """Keep the whole image, using linear interpolation and a black border."""
    # Float data avoids rounding interpolated intensities to whole numbers.
    return rotate(
        pixels.astype(np.float64), angle=45, reshape=True,
        order=1, mode="constant", cval=0.0, prefilter=False,
    )


def main():
    import napari

    original, metadata = load_image_and_metadata(IMAGE)
    dy, dx = metadata["pixel_size_yx"]
    # An array rotation represents a physical 45-degree rotation for square pixels.
    if not np.isclose(dy, dx):
        raise ValueError("This demo requires square pixels for a physical 45-degree rotation.")
    rotated = rotate_actin(original)
    spectrum = log_fourier_amplitude(rotated)
    rotated_metadata = {
        **metadata,
        "rotation_degrees": 45,
        "interpolation": "linear",
        "border_fill": 0,
    }
    height, width = rotated.shape
    spectrum_metadata = {
        "source_metadata": rotated_metadata,
        "frequency_step_yx": (1 / (height * dy), 1 / (width * dx)),
        "frequency_unit": f"cycles/{metadata['pixel_size_unit']}",
        "zero_frequency_index_yx": (height // 2, width // 2),
        "amplitude_transform": "ln(1 + abs(fftshift(fft2(rotated_image))))",
    }

    viewer = napari.Viewer(title="Actin: original, 45-degree rotation, and Fourier amplitude")
    # Match brightness settings for a fair comparison of the two spatial images.
    limits = (float(original.min()), float(original.max()))
    viewer.add_image(original, name="Original actin", colormap="gray",
                     contrast_limits=limits, metadata=metadata)
    viewer.add_image(rotated, name="Actin rotated 45 degrees", colormap="gray",
                     contrast_limits=limits, metadata=rotated_metadata)
    viewer.add_image(spectrum, name="Log Fourier amplitude of rotated actin",
                     colormap="gray", metadata=spectrum_metadata)
    # Use array coordinates for comparison. Calibration is retained in metadata;
    # Fourier coordinates measure frequency, rather than distance in the sample.
    viewer.grid.enabled = True
    viewer.grid.shape = (1, 3)
    viewer.reset_view()
    print(f"Original shape: {original.shape}; rotated shape: {rotated.shape}")
    print(f"Spatial pixel size retained: {dy:g} x {dx:g} {metadata['pixel_size_unit']}")
    print("Fourier panel: ln(1 + amplitude), with zero frequency at the center.")
    print("The zero-filled rotation border also contributes to the spectrum.")
    napari.run()


if __name__ == "__main__":
    main()
