# Automatic threshold methods in scikit-image

All functions below live in `skimage.filters`. They fall into two groups.

## Global methods: one threshold for the whole image

Most of these look only at the **histogram**, which counts how many pixels have each brightness value. Each one has its own rule for where to cut that histogram into "background" and "foreground".

| Function | How it picks the threshold | Works best when |
|---|---|---|
| `threshold_otsu` | Tries every possible threshold. It keeps the one where the pixels on each side are as similar to each other as possible (lowest spread within each group). | The histogram has two clear peaks of roughly similar size. This is the most common starting point. |
| `threshold_multiotsu` | The same idea, but it splits the image into 3 or more classes, for example background, dim nuclei and bright nuclei. | You have more than two populations. |
| `threshold_li` | Repeatedly adjusts the threshold so the two-level (0/1) image loses as little information as possible compared with the original ("minimum cross-entropy"). | Fluorescence images; it often gives a lower threshold than Otsu and keeps dim objects. |
| `threshold_yen` | Chooses the threshold that makes the foreground and background histograms each as "orderly" as possible (maximum entropy). | Small bright objects on a dark background. |
| `threshold_isodata` | Starts with a guess, then repeats: average the mean of the pixels below and the mean of the pixels above, and use that as the new threshold, until it stops changing. | Gives results very close to Otsu. |
| `threshold_triangle` | Draws a line from the histogram's main peak to its far end. The threshold is the histogram point farthest from that line. | The foreground is a **small** part of the image, so its peak is tiny next to the background peak. |
| `threshold_minimum` | Smooths the histogram until only two peaks remain, then puts the threshold in the valley between them. | Two well-separated peaks. It fails if the histogram never settles into two peaks. |
| `threshold_mean` | Uses the average pixel value. | Crude; mostly useful as a baseline for comparison. |

## Local (adaptive) methods: a different threshold for every pixel

These compute a threshold from each pixel's surroundings, for example a 51×51 pixel window around it. That handles uneven illumination or staining, where one value can't fit the whole image.

| Function | How it works |
|---|---|
| `threshold_local` | Each pixel is compared with the mean (or Gaussian-weighted mean, or median) of its window, minus an `offset` you choose. |
| `threshold_niblack` | Local mean + k × local standard deviation. |
| `threshold_sauvola` | A version of Niblack that copes better with low-contrast background. It was developed for scanned documents. |

## Two related tools

- **`apply_hysteresis_threshold(image, low, high)`** uses two thresholds. A pixel is foreground if it is above `high`, or if it is above `low` and connected to a pixel above `high`. This keeps the dim edges of real objects but drops isolated noise.
- **`try_all_threshold(image)`** shows the result of 7 global methods (Isodata, Li, Mean, Minimum, Otsu, Triangle, Yen) side by side in one figure. It's a quick way to compare them.

## What this means for the workshop DAPI image

In `3Channel_FF_Corrected_1_MMStack_Pos0.ome.tif`, a manual threshold of 1500 put only about 2.5% of DAPI pixels in the foreground. So the nucleus peak in the histogram is tiny compared with the background peak. Under those conditions:

- Otsu may still work, but it can be pulled toward the background.
- Triangle, Li or Yen often suit fluorescent nuclei better.

## What this means for CellDIVE liver data

- A global threshold may fail where staining varies across a large tissue piece. A local method, or a global method run tile by tile, is more robust there.
- `threshold_multiotsu` could help separate dim small immune-cell nuclei from bright hepatocyte nuclei.
