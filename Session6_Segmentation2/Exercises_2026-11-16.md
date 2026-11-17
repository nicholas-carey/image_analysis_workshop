# Session 6 exercises: Cellpose-SAM and flatfield correction

*Worked through on 16 November 2026.*

All commands are run from the `image_analysis_workshop` folder. All scripts are in `src/`.

---

## 1. Does Cellpose-SAM run on the Mac GPU?

**Question:** Can Cellpose use GPU acceleration on a Mac, and do we need to install anything?

**Answer:** Yes, and no extra installation is needed. Macs do not use NVIDIA's CUDA. On Apple Silicon (M1–M4), PyTorch uses Apple's own GPU backend, called **MPS** (Metal Performance Shaders). The standard `torch` installed by `uv` already includes it.

Check it yourself:

```bash
uv run python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
```

In your own code, ask for the GPU when you load the model; Cellpose picks MPS automatically:

```python
from cellpose import models
model = models.CellposeModel(gpu=True)   # model.device -> mps
```

In the Cellpose GUI, tick **use GPU**.

**Speed test** (one 512 × 512 image of nuclei, M4 Max):

| Setting | Device | Time | Nuclei found |
|---|---|---|---|
| `gpu=True` | mps | ~1 s | 309 |
| `gpu=False` | cpu | ~88 s | 331 |

**Things to notice**
- The GPU is about 90× faster here.
- GPU and CPU give slightly different results (309 vs. 331 nuclei), because MPS does some of the maths a little differently. Pick one for a whole analysis, so your numbers stay comparable.
- The first run downloads the Cellpose-SAM model (~1.15 GB) to `~/.cellpose/models`. Do this before the workshop, not on the workshop Wi-Fi.

---

## 2. Segment nuclei and cells in one image

**Script:** `src/cellpose_nuclei_cytoplasm.py`

**Image:** `testImages/3Channel_FF_Corrected_1/3Channel_FF_Corrected_1_MMStack_Pos0.ome.tif`. This image has 3 channels: Cy3, FITC and DAPI. The pixel size is 0.3238 µm.

```bash
uv run src/cellpose_nuclei_cytoplasm.py testImages/3Channel_FF_Corrected_1/3Channel_FF_Corrected_1_MMStack_Pos0.ome.tif
```

**What the script does**
1. It reads the DAPI and FITC channels by name and reads the pixel size.
2. **Nuclei:** it runs Cellpose-SAM on the DAPI channel alone.
3. **Cells:** it runs Cellpose-SAM on FITC (the cytoplasm) **plus** DAPI. The nuclei tell Cellpose where each cell is, which helps it separate neighbouring cells whose cytoplasm touches.
4. It saves both masks as label images. In a label image, every pixel holds the number of the object it belongs to, and 0 is background. The files keep the original pixel size:
   - `..._DAPI_cellpose_nuclei_labels.tif`
   - `..._FITC_cellpose_cells_labels.tif`
5. It shows the channels with both masks as outlines in napari. To see filled masks, set **contour** to 0 in a mask layer's controls. Close napari to end the script.

**Result:** 54 nuclei and 61 cells, about 10 s per mask on the GPU.

**Things to try**
- Why are there more cells than nuclei? 11 cells have no nucleus. Most of them touch the image edge, so their nucleus is outside the field of view. Look at the others in napari: are they pieces of cytoplasm that were split off?
- Run with `--cells-from-fitc-only` to segment cells from FITC alone. What does the DAPI channel add?
- Run with `--cpu` and compare the time.

**Warnings you can ignore:** *"Sparse invariant checks are implicitly disabled"* comes from inside Cellpose and does not affect the results.

---

## 3. Flatfield correction: does it matter for segmentation and for intensities?

**Script:** `src/flatfield_cellpose_nuclei.py`

**Data:** `3Channel_Raw_25Pos_1`, a 5 × 5 grid of tiles taken with the 20x objective. The channels are Cy3, FITC and DAPI.
**Flatfield images:** `Flatfield/20x/<channel>-20x.tif`. **Camera dark image:** `Flatfield/Dark.tif`.

```bash
uv run src/flatfield_cellpose_nuclei.py                    # full run, ~12 min on the GPU
uv run src/flatfield_cellpose_nuclei.py --reuse-masks      # redo table + plots without Cellpose
uv run src/flatfield_cellpose_nuclei.py --reuse-masks --show Pos-1-002_002   # one tile in napari
```

**What the script does, for every tile**
1. **Flatfield correction** of all 3 channels (see the README):

   > CI = (MI − BI) / FI

   MI is the measured image and BI is the dark image (the camera offset, ~100 counts). FI is the flatfield minus the dark image, divided by its own mean so that it averages 1. The corrected tiles are saved as 16-bit OME-TIFFs that keep the channel names and pixel size.
2. **Cellpose-SAM nuclei** in the DAPI channel, run **twice**: once on the raw image and once on the corrected image. Nuclei touching the tile edge are dropped because they are cut off.
3. **Measurements** for every nucleus, in the image it was found in:
   - **Area** (µm²).
   - **Local background:** the median of a ring of pixels 4–15 px around the nucleus. Pixels close to other nuclei are excluded.
   - **Background-corrected summed DAPI:** the sum of the nucleus's pixels minus (background × area in pixels).

**Outputs** (in `3Channel_Raw_25Pos_1/flatfield_cellpose/`)
- `nuclei_cellpose_raw_and_flatfield.csv` has one row per nucleus. The `condition` column says `raw` or `flatfield`.
- `nuclei_cellpose_comparison.png` has four panels: the histogram of summed DAPI, intensity vs. distance from the image center, nucleus area, and nuclei per tile.
- `corrected/` holds the corrected tiles and `masks/` the Cellpose masks.

**Results**

| | Raw | Flatfield corrected |
|---|---|---|
| Nuclei found (not touching the edge) | 1305 | 1299 |
| Median area (µm²) | 198.6 | 198.3 |
| Robust CV of summed DAPI | 52.6 % | 48.5 % |
| Summed DAPI, edge nuclei / center nuclei | **0.51** | **0.81** |
| Same nucleus found in both (overlap ≥ 50%) | 96.8 % | |

"Edge" means nuclei more than 350 µm from the image center; "center" means nuclei less than 150 µm from it.

**What we learned**
- **Segmentation hardly changes.** Cellpose rescales the brightness of each image itself, so the uneven illumination barely affects which nuclei it finds or how large they are.
- **Intensities change a lot.** Without correction, nuclei at the edge of a tile look about **half as bright** as nuclei in the center. If you quantify DNA content or any other intensity, flatfield correction is essential.
- **Correction is not perfect.** After correction, edge nuclei are still at 0.81 of the center value, and the far corners stay dimmer. Possible causes: biology (cell density and cell-cycle stage vary across the dish), or a flatfield image that does not fully match the sample.

**Things to look at / try**
- **Small objects:** about 30 objects have a summed DAPI near zero. Are they debris? Try `--reuse-masks --min-area 50` to drop small objects without re-running Cellpose.
- **Clumps:** objects of 500–2,700 µm² (the median nucleus is ~200 µm²) are probably clumps that Cellpose did not split. How do they change the histogram?
- **Cell cycle:** the summed-DAPI histogram should show G1 and G2 peaks, with G2 at twice the DNA of G1. Can you see them? Are they sharper with or without correction?
- **Comparison with Session 5:** the earlier script `src/flatfield_compare_nuclei.py` does the same comparison with Otsu + watershed segmentation. How do its nucleus counts and histograms compare with Cellpose's?
