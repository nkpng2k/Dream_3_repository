"""
The purpose of this is to learn how to operate scikit-image and scikit-video

I will use these libraries in tandem with some classifiers to identify cards dealt
and count each card
"""
from skimage import data, filters, io, measure, transform, feature
import numpy as np
import matplotlib.pyplot as plt
import PIL as pil
import math

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
edges = filters.threshold_minimum(card_image)
filtered = card_image.copy()
filtered[filtered < edges] = 0

coords = np.argwhere(filtered > 0.9)

miny, minx = coords.min(axis = 0)
maxy, maxx = coords.max(axis = 0)

xs = [minx, minx, maxx, maxx]
ys = [miny, maxy, miny, maxy]

cropped = filtered[miny:maxy,minx:maxx]

#NOTE: use np.polyfit(), np.roots() of two polyfits will return the intersections

fig, ax = plt.subplots()
ax.imshow(img[2], cmap = plt.cm.gray)
# ax.scatter(xs, ys)
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

im = cv2.imread('samples/IMG_1197.jpg')
gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray,(1,1),1000)
flag, thresh = cv2.threshold(blur, 120, 255, cv2.THRESH_BINARY)

contours = cv2.findContours(thresh,1,2)
cnt = contours[0]
cv2.contourArea(cnt)
areas = []
for contour in contours:
    area = cv2.contourArea(contour)
    areas.append(area)


"""
Bottom of Page
"""
