"""
The purpose of this is to learn how to operate scikit-image and scikit-video

I will use these libraries in tandem with some classifiers to identify cards dealt
and count each card
"""
from skimage import filters, io
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
edges = filters.thresholding.threshold_minimum(card_image)
filtered = card_image.copy()
filtered[filtered < edges] = 0

io.imshow(filtered)
io.show()

dbl_card = ski.io.imread('samples/IMG_1199.jpg', as_grey = True)

edges = ski.filters.thresholding.threshold_otsu(dbl_card)
dbl_filtered = dbl_card.copy()
dbl_filtered[dbl_filtered < edges] = 0

ski.io.imshow(dbl_filtered)
ski.io.show()

#NOTE: need to use contour lines to generate bounding box
#NOTE: I now have cv2. Will this help?
#NOTE: will need to create your own training data, vectorize the data and label

import cv2

test = cv2.imread('card_images/6_d.bmp')
test_jpg = cv2.imread('samples/IMG_1197.jpg')

plt.imshow(cv2.cvtColor(test, cv2.COLOR_BGR2RGB))
plt.show()

plt.imshow(test_jpg, cmap = 'gray')
plt.show()

plt.imshow(cv2.cvtColor(test_jpg, cv2.COLOR_BGR2RGB))
plt.show()

"""
Bottom of Page
"""
