"""Display mitochondria and their log Fourier amplitude side by side.

Run from the project folder: uv run src/show_mito_fourier.py
"""

from pathlib import Path

import numpy as np
import tifffile

IMAGE = Path(__file__).resolve().parents[1] / "testImages" / "Mito_Cropped.tif"


def load_image_and_metadata(path):
    """Read this ImageJ TIFF, including its physical pixel sizes."""
    with tifffile.TiffFile(path) as tif:
        pixels = tif.asarray()
        tags = tif.pages[0].tags
        imagej = dict(tif.imagej_metadata or {})
        # TIFF resolutions are pixels per unit; invert to get units per pixel.
        sizes = []
        for name in ("YResolution", "XResolution"):
            numerator, denominator = tags[name].value
            sizes.append(denominator / numerator)
    if pixels.ndim != 2:
        raise ValueError(f"Expected a single 2D image, got shape {pixels.shape}.")
    return pixels, {
        "source": str(path),
        "pixel_size_yx": tuple(sizes),
        "pixel_size_unit": imagej.get("unit", "unknown"),
        "imagej_metadata": imagej,
    }


def log_fourier_amplitude(pixels):
    """Move zero frequency to the center and compress the amplitude range."""
    fourier = np.fft.fftshift(np.fft.fft2(pixels))
    # Adding 1 avoids log(0); natural logarithms reveal weaker frequencies.
    return np.log1p(np.abs(fourier))


def main():
    import napari

    pixels, metadata = load_image_and_metadata(IMAGE)
    spectrum = log_fourier_amplitude(pixels)
    dy, dx = metadata["pixel_size_yx"]
    height, width = pixels.shape
    frequency_metadata = {
        "source_metadata": metadata,
        "frequency_step_yx": (1 / (height * dy), 1 / (width * dx)),
        "frequency_unit": f"cycles/{metadata['pixel_size_unit']}",
        "zero_frequency_index_yx": (height // 2, width // 2),
        "amplitude_transform": "ln(1 + abs(fftshift(fft2(image))))",
    }

    viewer = napari.Viewer(title="Mitochondria: image and log Fourier amplitude")
    viewer.add_image(pixels, name="Original mitochondria",
                     colormap="gray", metadata=metadata)
    viewer.add_image(spectrum, name="Log Fourier amplitude",
                     colormap="gray", metadata=frequency_metadata)
    # Both panels use array coordinates so they have matching display sizes.
    # Physical pixel sizes and reciprocal frequency spacing remain in metadata;
    # a spatial scale bar would not apply to the Fourier panel.
    viewer.grid.enabled = True
    viewer.grid.shape = (1, 2)
    viewer.reset_view()
    print(f"Source pixel size (Y, X): {dy:g}, {dx:g} {metadata['pixel_size_unit']}")
    print("Panels: original image and ln(1 + Fourier amplitude).")
    print("Center = zero frequency; farther from center = finer spatial detail.")
    print("Display coordinates are pixels/bins; calibration is in layer metadata.")
    napari.run()


if __name__ == "__main__":
    main()
