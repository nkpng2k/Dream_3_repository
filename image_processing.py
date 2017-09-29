import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob


class CardImageProcessing(object):
    """
    Class that will process card images within a file and return the processed images

    INPUT: directory with images
    OUTPUT: images processed into 1-D vectors as list of numpy arrays
    """

    def __init__(self, file_path):
        self.file_path = file_path






import glob
images = [str(file) for file in glob.glob("/Users/npng/galvanize/Dream_3_repository/card_images/*")]
images

from os import listdir
from os.path import isfile, join, splitext
import numpy
import cv2

mypath='/Users/npng/galvanize/Dream_3_repository/card_images'
onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]

images = numpy.empty(len(onlyfiles), dtype=object)
for n in range(0, len(onlyfiles)):
  images[n] = cv2.imread( join(mypath,onlyfiles[n]) )

for f in listdir(mypath):
    fname, extension = splitext(f)
    print fname, extension






"""
bottom of page
"""
