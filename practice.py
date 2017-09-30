"""
The purpose of this is to learn how to operate scikit-image and scikit-video

I will use these libraries in tandem with some classifiers to identify cards dealt
and count each card
"""
from skimage import data, filters, io, measure
import numpy as np
import matplotlib.pyplot as plt
import PIL as pil

image = data.coins()
edges = filters.sobel(image)
io.imshow(edges)
io.show()

"""
train a model to classify cards
pass the model new image frames with a new card
have the model identify it and count
"""
%cd /Users/npng/galvanize/Dream_3_repository

#read in sample with grey scale -- Numpy values 0 = black, 1 = white
card_image = io.imread('samples/IMG_1197.JPG', as_grey = True)

# filters.thresholding.threshold_minimum finds minimum value to separate edges
edges = filters.thresholding.threshold_minimum(card_image)
filtered = card_image.copy()
filtered[filtered < edges] = 0

contours = measure.find_contours(filtered, edges)
fig, ax = plt.subplots()
ax.imshow(filtered, interpolation='nearest', cmap=plt.cm.gray)

for n, contour in enumerate(contours):
    ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
plt.show()

dbl_card = io.imread('samples/IMG_1199.jpg', as_grey = True)

edges = filters.thresholding.threshold_otsu(dbl_card)
dbl_filtered = dbl_card.copy()
dbl_filtered[dbl_filtered < edges] = 0

io.imshow(dbl_filtered)
io.show()

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

test_grey = cv2.cvtColor(test_jpg, cv2.COLOR_BGR2GRAY)
thresh = cv2.adaptiveThreshold(test_grey,255,1,1,11,2)
im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(test_grey, contours, -1, (0,255,0), 3)
plt.imshow(thresh)
plt.show()




plt.imshow(test_grey[20:25], cmap = 'gray')
plt.show()


plt.imshow(cv2.cvtColor(test_jpg, cv2.COLOR_BGR2RGB))
plt.show()

"""
Bottom of Page
"""
