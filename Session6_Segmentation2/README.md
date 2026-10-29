# Segmentation (again)

## Illumination correction
Something that we glossed over in the previous section is illumination (or flatfield) correction.  Even in the best adjusted microscope, the illumination intensity (and the dection efficiency) is different in different parts of the image.  Most often, things are bright in the center and dimmer at the edges.  You can often see this by taking an image with your object of interest in the center, estimating (or measuring) its intenisty, the move it to a corner and repeat the estimate.  

This is obviously disastrous when you quantify things in your image.  They better have the same intensity independent of where in the image they are.  

To improve things we often apply "flatfield correction".  This relies on having perfect knowlegde of the combination of illumination and detection efficiency at each pixel.  In the CALM, we create such "flatfield" image using the best possible approximation of a uniform, thin, fluorescent sample.  These consist of highly concentrated fluorescent dye solution, so concentrated that the absorb most of the excitation light within the first micron or less of the sample, hence only a thin film is fluoresceing.  This approach was first suggested by [Michael Model](http://www.ncbi.nlm.nih.gov/pubmed/11500847). For a full protocol, see [the CALM website](https://calm.ucsf.edu/how-acquire-flat-field-correction-images).

When we consider this flatfield image (FI), the background image (BI the "dark" image of the camera), the measured image ('MI") and the "ideal", corrected image (CI) that we would like to have, we can write the following equation:

> MI = (CI * FI) + BI

Rewriting this to solve for CI, we get:

> CI = (MI - BI) / FI


When doing this calculation, it is nice to normalize the FI image to 1 (i.e. convert it to a floating point image and divide the image by its mean).  That way the mean intensity of the CI stays the same as that of the MI.  

* Exercise to compare results before and after flatfielding.

## Segmentation using trained networks
During the previous session you have seen how you can combine convolution filters to create an image that can be easily segmented into the objects that you are ineterested in.  Choosing the correct (sequencee) of filters and selecting a threshold is laborious and often not transferable from one image to the next.  Computer vision scientist have long worked on methods to automate this process so that a single operation can produce the desired segmentation.

To do so, researchers used annotated data sets, consisting of pairs of input images and the desired (segmented) result.  Initially, gigantic datasets were needed to produce the desired results.  Those algorithms were often fragile and did not work with images slightly different from what was in the training set.  A big breakthrough was the design of a ["U-Net"](https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28) architecture.  In a U-Net, the image is convoluted and downsamples in a couple of steps to a compressed, smaller image, and then convluted and upsampled.  A fuller and much better description of this architecture can be found [here](https://www.geeksforgeeks.org/machine-learning/u-net-architecture-explained/) and in the original paper.  A huge advantage of this architecture is that the size of the training set can be greatly reduced, and training sets of 10s of images can be enough to give useful results.

Many deep learning tools based on U-Net have been released.  Possibly the most widely used one is [Cellpose](https://www.biorxiv.org/content/10.1101/2020.02.02.931238v2.full) which not only excelled at segmenting cells from microscope images it had never seen before, but also was released with a nice user interface including tools to fine-train Cellpose for images on which it did not directly yield the desired results.
