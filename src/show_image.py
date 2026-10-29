"""Your first script: open a 3-channel microscopy image in Napari.

Run it from the top folder of the workshop repository with:

    uv run src/show_image.py

Napari opens in its own window.  Close that window to end the script.
"""

import json
from pathlib import Path

import napari
from bioio import BioImage
from napari.utils import Colormap

# Where the image lives.  Path(__file__) is this script, ".parent.parent" walks
# up two folders to the top of the workshop repository.  Building the path this
# way means the script works no matter which folder you started it from.
REPO = Path(__file__).resolve().parent.parent
IMAGE_DIR = REPO / "testImages" / "3Channel_FF_Corrected_1"
IMAGE = IMAGE_DIR / "3Channel_FF_Corrected_1_MMStack_Pos0.ome.tif"

# BioImage opens many microscopy formats and gives access to the metadata that
# the microscope stored alongside the pixels.
img = BioImage(IMAGE)

# Ask for the pixels in channel-row-column order ("CYX").  The result is a
# NumPy array of shape (3, 2048, 2048): 3 channels, each 2048 x 2048 pixels.
pixels = img.get_image_data("CYX")

# The names of the three filters used, read from the image file itself.
channel_names = [str(name) for name in img.channel_names]

# How large one pixel is in the sample, in micrometers.
pixel_size = img.physical_pixel_sizes

# The colors the microscope software used to display each channel are *not*
# stored in the image file; Micro-Manager keeps them in a separate
# "DisplaySettings.json" file next to it.  Read them from there, so the colors
# on your screen match what was seen at the microscope.  Each color is given as
# red, green and blue fractions between 0 and 1.
display_file = IMAGE_DIR / "DisplaySettings.json"
with open(display_file) as f:
    display_settings = json.load(f)

file_colors = {}
for channel in display_settings["map"]["ChannelSettings"]["array"]:
    name = channel["Channel"]["scalar"]
    file_colors[name] = channel["Color"]["scalar"]["Components"]

# Napari needs a "colormap": a recipe for turning pixel values into colors.
# A dark pixel should be black and the brightest pixel the channel's color, so
# each colormap runs from black to that color.  The fourth number is opacity,
# which stays at 1 (fully opaque).
colormaps = []
for name in channel_names:
    red, green, blue = file_colors.get(name, [1.0, 1.0, 1.0])
    colormaps.append(
        Colormap(colors=[[0.0, 0.0, 0.0, 1.0], [red, green, blue, 1.0]], name=name)
    )

print(f"Loaded {IMAGE.name}")
print(f"  shape      : {pixels.shape}  (channels, height, width)")
print(f"  data type  : {pixels.dtype}  (16-bit, so values run 0 - 65535)")
print(f"  value range: {pixels.min()} - {pixels.max()}")
print(f"  pixel size : {pixel_size.X} x {pixel_size.Y} micrometer")
print("  channels   :")
for name in channel_names:
    print(f"    {name:6s} displayed as RGB {file_colors.get(name)}")

# Hand the array to Napari.  "channel_axis=0" tells Napari that the first
# dimension holds the channels, so it makes one layer per channel instead of
# treating the stack as a 3-slice 3D volume.  "blending='additive'" overlays
# the colors the way a fluorescence microscope does: where channels overlap,
# their colors add up.  "scale" and "units" make Napari aware of the real pixel
# size, so the scale bar is in micrometers.
viewer = napari.Viewer()
viewer.add_image(
    pixels,
    channel_axis=0,
    name=channel_names,
    colormap=colormaps,
    blending="additive",
    scale=(pixel_size.Y, pixel_size.X),
    units=("um", "um"),
)
viewer.scale_bar.visible = True

# Start the Napari window and wait until you close it.
napari.run()
