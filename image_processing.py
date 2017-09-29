import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
from os import listdir
from os.path import isfile, join, splitext
from collections import Counter

class CardImageProcessing(object):
    """
    Class that will process card images within a file and return the processed images

    INPUT: directory with images
    ATTRIBUTES: raw images read into list
                images processed into 1-D vectors as list of numpy arrays
                labels for each vectorized image
    """

    def __init__(self, file_path):
        self.file_path = file_path
        files, f_ext, f_names = self._file_info()
        self.files = files
        self.file_names = f_names
        self.file_ext = f_ext

    def _file_info(self):
        onlyfiles = [f for f in listdir(self.file_path) if isfile(join(self.file_path, f))]
        file_ext_count = Counter()
        for f in onlyfiles:
            fname, file_type = splitext(f)
            file_ext_count[file_type] += 1

        file_ext = file_ext_count.most_common()[0][0]
        files = [f for f in onlyfiles if splitext(f)[1] == file_ext]
        file_names = [splitext(f)[0] for f in onlyfiles if splitext(f)[1] == file_ext]

        return files, file_ext, file_names

    def label_images(self, delimiter = None,labels = None):
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
                print name
                # card_type.append(name.split(delimiter)[0])
                # card_suit.append(name.split(delimiter)[1])
        else:
            for tup in labels:
                card_type.append(tup[0])
                card_suit.append(tup[1])
        return card_type, card_suit

    def _read_in_images(self):
        for f in self.files:
            pass



if __name__ == "__main__":
    card_process = CardImageProcessing('/Users/npng/galvanize/Dream_3_repository/card_images')
    c_type, c_suit = card_process.label_images(delimiter = '_')




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
