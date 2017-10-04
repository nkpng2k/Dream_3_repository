import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
from os import listdir
from os.path import isfile, join, splitext
from collections import Counter
from skimage import io, filters, color, transform
import math

class CardImageProcessing(object):
    """
    Class that will process card images within a file and return the processed images
    NOTE: must run file_info(self, file_path) method before any other preprocessing

    INPUT: directory with images
    ATTRIBUTES: self.raw_img - raw images read into list
                self.files - list of files
                self.file_names - list of files sans extensions
                self.file_ext - file extension used to parse files
    METHODS: label_images (returns: list of labels)
             vectorize_images (returns: array of 1-D vectors, raw images vectorized into 1-D array)
    """

    def __init__(self):
        self.file_path = None
        self.files = None
        self.file_names = None
        self.file_ext = None

    def _read_in_images(self):
        raw_list = []
        grey_list = []
        for f in self.files:
            img = io.imread(self.file_path+'/'+f)
            raw_list.append(img)
        for img in raw_list:
            grey = color.rgb2grey(img)
            grey_list.append(grey)
        return raw_list, grey_list

    # NOTE: all methods are written below this note

    def file_info(self, file_path):
        onlyfiles = [f for f in listdir(file_path) if isfile(join(file_path, f))]
        file_ext_count = Counter()
        for f in onlyfiles:
            fname, file_type = splitext(f)
            file_ext_count[file_type] += 1

        self.file_path = file_path
        self.file_ext = file_ext_count.most_common()[0][0]
        self.files = [f for f in onlyfiles if splitext(f)[1] == self.file_ext]
        self.file_names = [splitext(f)[0] for f in onlyfiles if splitext(f)[1] == self.file_ext]

        raw_imgs = self._read_in_images()
        return raw_imgs

    def generate_labels(self, delimiter = None,labels = None):
        """
        will manually assign labels for each of the images or if no manual labels are
        provided will pull the characters up until a specified delimiter as the label

        INPUT: labels --> (list or tuples) optional, assign manual labels for images
                          tuple will have this order: (card type, card suit)
               delimiter --> (string) delimiter that will is expected to separate the card type and
                             card suit. example: queen_heart.png - delimiter = '_'
        OUTPUT: 2 lists --> card type and card suit
        """
        card_type = []
        card_suit = []
        if labels == None:
            for name in self.file_names:
                card_type.append(name.split(delimiter)[0])
                card_suit.append(name.split(delimiter)[1])
        else:
            for tup in labels:
                card_type.append(tup[0])
                card_suit.append(tup[1])
        return card_type, card_suit

    def bounding_box_crop(self, images):
        """
        Detect edges, mask everything outside of edges to 0,
        determine coordinates for corners of card,
        crop box tangent to corners of card,
        return cropped images
        """
        cropped_list = []
        for img in images:
            edges = filters.thresholding.threshold_minimum(img)
            img[img < edges] = 0

            coords = np.argwhere(img > 0.9)

            miny, minx = coords.min(axis = 0)
            maxy, maxx = coords.max(axis = 0)

            cropped = img[miny:maxy,minx:maxx]

            cropped_list.append(cropped)

        return cropped_list


    #NOTE: Completed methods above this line
    
    def rotate_images(self, images):
        pass

    def vectorize_images(self, images):
        pass





if __name__ == "__main__":
    card_process = CardImageProcessing()
    raw_imgs, grey_imgs = card_process.file_info('/Users/npng/galvanize/Dream_3_repository/card_images')
    c_type, c_suit = card_process.generate_labels(delimiter = '_')
    cropped_imgs = card_process.bounding_box_crop(grey_imgs)

    io.imshow(cropped_imgs[5])
    io.show()

    # import glob
    # images = [file for file in glob.glob("/Users/npng/galvanize/Dream_3_repository/card_images/*")]
    #
    # mypath='/Users/npng/galvanize/Dream_3_repository/card_images'
    # onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
    #
    # images = numpy.empty(len(onlyfiles), dtype=object)
    # for n in range(0, len(onlyfiles)):
    #   images[n] = cv2.imread( join(mypath,onlyfiles[n]) )
    #
    # for f in listdir(mypath):
    #     fname, extension = splitext(f)
    #     print fname, extension






"""
bottom of page
"""
