"""Compare Gaussian blurs, detail masks, and adjustable unsharp sharpening.

Run from the project folder:
    uv run src/Codex_scripts/gaussian_unsharp_masks.py
    uv run src/Codex_scripts/gaussian_unsharp_masks.py --sigma-5 1 --sigma-21 5
    uv run src/Codex_scripts/gaussian_unsharp_masks.py --amount-5 0.5 --amount-21 1.5

Keep median_filters.py beside this script for its pixel-size metadata reader.
The saved unsharp_masked images are the signed differences requested here;
sharpened = original + amount * unsharp_masked. Each amount has its own live
slider in napari. Click "Save current sharpened images" to save slider changes.
Initial sharpened results are saved automatically, including with --no-view.
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
import tifffile

from median_filters import read_image


def positive_sigma(value):
    sigma = float(value)
    if not np.isfinite(sigma) or sigma <= 0:
        raise argparse.ArgumentTypeError("Sigma must be a finite number greater than zero.")
    return sigma


def nonnegative_amount(value):
    amount = float(value)
    if not np.isfinite(amount) or amount < 0:
        raise argparse.ArgumentTypeError("Amount must be a finite number at least zero.")
    return amount


def sharpen_image(original, mask, amount):
    """Amount 0 returns the original; amount 1 adds the full detail mask."""
    if not np.isfinite(amount) or amount < 0:
        raise ValueError("Amount must be finite and at least zero.")
    return (original.astype(np.float64) + amount * mask.astype(np.float64)).astype(np.float32)


def save_sharpened(input_path, images, amounts, sigmas, save_options):
    """Include the chosen settings in filenames; retain spatial calibration."""
    paths = []
    for size in (5, 21):
        output = input_path.with_name(
            f"{input_path.stem}_sharpened_{size}x{size}"
            f"_sigma_{sigmas[size]}_amount_{amounts[size]}.tif")
        tifffile.imwrite(output, images[size], **save_options)
        print(f"Saved {output}")
        paths.append(output)
    return paths


def filter_images(image, sigma_5=1.0, sigma_21=5.0):
    """Use exact kernel sizes; retain negative differences and fractional values."""
    if any(not np.isfinite(s) or s <= 0 for s in (sigma_5, sigma_21)):
        raise ValueError("Sigma values must be finite and greater than zero.")
    original = image.astype(np.float64)
    results = {}
    for size, sigma in ((5, sigma_5), (21, sigma_21)):
        # Radius fixes the width to 2 * radius + 1, independent of sigma.
        # Gaussian weights are normalized to sum to one. Reflect at the edges.
        blurred = gaussian_filter(original, sigma=sigma, radius=size // 2,
                                  mode="reflect")
        results[f"gaussian_{size}x{size}"] = blurred.astype(np.float32)
        # Subtract in floating point so negative values do not wrap or clip.
        results[f"unsharp_masked_{size}x{size}"] = (original - blurred).astype(np.float32)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path,
                        default=Path(__file__).resolve().parents[2] / "testImages/Actin_cropped.tif")
    parser.add_argument("--sigma-5", type=positive_sigma, default=1.0,
                        help="Gaussian width in pixels for the 5x5 kernel (default: 1)")
    parser.add_argument("--sigma-21", type=positive_sigma, default=5.0,
                        help="Gaussian width in pixels for the 21x21 kernel (default: 5)")
    parser.add_argument("--amount-5", type=nonnegative_amount, default=1.0,
                        help="Fraction of the 5x5 detail mask to add back (default: 1)")
    parser.add_argument("--amount-21", type=nonnegative_amount, default=1.0,
                        help="Fraction of the 21x21 detail mask to add back (default: 1)")
    parser.add_argument("--no-view", action="store_true", help="Save without opening napari")
    args = parser.parse_args()

    original, save_options, scale, unit = read_image(args.input)
    results = filter_images(original, args.sigma_5, args.sigma_21)
    for name, image in results.items():
        output = args.input.with_name(f"{args.input.stem}_{name}.tif")
        tifffile.imwrite(output, image, **save_options)
        print(f"Saved {output}")
    print(f"Gaussian sigma: 5x5 = {args.sigma_5} pixels; 21x21 = {args.sigma_21} pixels")
    print("unsharp_masked = original minus blur (signed difference)")

    amounts = {5: args.amount_5, 21: args.amount_21}
    sigmas = {5: args.sigma_5, 21: args.sigma_21}
    sharpened = {size: sharpen_image(original, results[f"unsharp_masked_{size}x{size}"], amount)
                 for size, amount in amounts.items()}
    save_sharpened(args.input, sharpened, amounts, sigmas, save_options)

    if not args.no_view:
        import napari
        from magicgui.widgets import Container, FloatSlider, Label, PushButton

        viewer = napari.Viewer(title=f"Gaussian blur and unsharp masks — scale in {unit}")
        intensity_limits = (float(original.min()), float(original.max()))
        if intensity_limits[0] == intensity_limits[1]:
            intensity_limits = (intensity_limits[0], intensity_limits[0] + 1)
        # Differences have a separate, shared symmetric range: zero is gray,
        # positive detail is bright, and negative detail is dark.
        amplitude = max(float(np.max(np.abs(data))) for name, data in results.items()
                        if name.startswith("unsharp_masked"))
        amplitude = amplitude or 1.0
        layers = {"Original": original, **results}
        for name, data in layers.items():
            limits = (-amplitude, amplitude) if name.startswith("unsharp_masked") else intensity_limits
            viewer.add_image(data, name=name, colormap="gray", scale=scale,
                             contrast_limits=limits)
        sharpened_layers = {}
        for size in (5, 21):
            sharpened_layers[size] = viewer.add_image(
                sharpened[size], name=f"Sharpened {size}x{size} (amount {amounts[size]:g})",
                colormap="gray", scale=scale, contrast_limits=intensity_limits)

        sliders = {
            size: FloatSlider(label=f"{size}x{size} amount", value=amounts[size],
                              min=0, max=max(5.0, amounts[size]), step=0.05)
            for size in (5, 21)
        }
        save_button = PushButton(text="Save current sharpened images")
        status = Label(value="Initial images saved. Move sliders to preview changes.")
        controls = Container(widgets=[
            Label(value="Sharpened = original + amount × (original − blur)"),
            Label(value="0 = original; 0.5 = half mask; 1 = full mask; >1 = stronger"),
            sliders[5], sliders[21], save_button, status,
        ])

        def update_preview(*_):
            for size in (5, 21):
                amounts[size] = float(sliders[size].value)
                sharpened[size] = sharpen_image(
                    original, results[f"unsharp_masked_{size}x{size}"], amounts[size])
                sharpened_layers[size].data = sharpened[size]
                sharpened_layers[size].name = f"Sharpened {size}x{size} (amount {amounts[size]:g})"
            status.value = "Preview changed. Click Save to write these amounts."

        def save_current(*_):
            paths = save_sharpened(args.input, sharpened, amounts, sigmas, save_options)
            status.value = "Saved: " + "; ".join(path.name for path in paths)

        for slider in sliders.values():
            slider.changed.connect(update_preview)
        save_button.clicked.connect(save_current)
        viewer.window.add_dock_widget(controls, name="Unsharp mask amounts", area="right")
        # Fixed display limits make comparisons consistent while adjusting amounts.
        # Highlights can saturate on screen, but saved float images are not clipped.
        viewer.grid.enabled = True
        viewer.grid.shape = (2, 4)
        viewer.reset_view()
        napari.run()


if __name__ == "__main__":
    main()
