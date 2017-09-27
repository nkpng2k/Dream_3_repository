"""
The purpose of this is to learn how to operate scikit-image and scikit-video

I will use these libraries in tandem with some classifiers to identify cards dealt
and count each card
"""
import skimage as ski
import numpy as np
import matplotlib.pyplot as plt
import PIL as pil

image = ski.data.coins()
edges = ski.filters.sobel(image)
ski.io.imshow(edges)
ski.io.show()

"""
train a model to classify cards
pass the model new image frames with a new card
have the model identify it and count
"""
%cd /Users/npng/galvanize/Dream_3_repository

#read in sample with grey scale -- Numpy values 0 = black, 1 = white
card_image = ski.io.imread('samples/IMG_1197.JPG', as_grey = True)

# filters.thresholding.threshold_minimum finds minimum value to separate edges
edges = ski.filters.thresholding.threshold_minimum(card_image)
filtered = card_image.copy()
filtered[filtered < edges] = 0

ski.io.imshow(filtered)
ski.io.show()

dbl_card = ski.io.imread('samples/IMG_1199.jpg', as_grey = True)

edges = ski.filters.thresholding.threshold_otsu(dbl_card)
dbl_filtered = dbl_card.copy()
dbl_filtered[dbl_filtered < edges] = 0

ski.io.imshow(dbl_filtered)
ski.io.show()

#NOTE: need to use contour lines to generate bounding box


"""
Bottom of Page
"""
