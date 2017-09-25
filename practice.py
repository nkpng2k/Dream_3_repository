"""
The purpose of this is to learn how to operate scikit-image and scikit-video

I will use these libraries in tandem with some classifiers to identify cards dealt
and count each card
"""
from skimage import data, io, filters
import numpy as np

image = data.coins()
edges = filters.sobel(image)
io.imshow(edges)
io.show()


"""
train a model to classify cards
pass the model new image frames with a new card
have the model identify it and count
"""
# %cd /Users/npng/galvanize/Dream_3_repository

#read in sample with grey scale -- Numpy values 0 = black, 1 = white
card_image = io.imread('samples/IMG_1197.JPG', as_grey = True)

# filters.thresholding.threshold_minimum finds minimum value to separate edges
edges = filters.thresholding.threshold_minimum(card_image)
filtered = card_image.copy()
filtered[filtered < edges] = 0

io.imshow(filtered)
io.show()

dbl_card = io.imread('samples/IMG_1199.jpg', as_grey = True)

edges = filters.thresholding.threshold_minimum(dbl_card)
dbl_filtered = dbl_card.copy()
dbl_filtered[dbl_filtered < edges] = 0

io.imshow(dbl_filtered)
io.show()
