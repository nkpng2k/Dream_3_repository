"""
The purpose of this is to learn how to operate scikit-image and scikit-video

I will use these libraries in tandem with some classifiers to identify cards dealt
and count each card
"""
from skimage import data, io, filters

image = data.coins()
edges = filters.sobel(image)
io.imshow(edges)
io.show()
