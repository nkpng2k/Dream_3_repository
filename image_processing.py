import numpy as np
import matplotlib.pyplot as plt
import glob
from os import listdir
from os.path import isfile, join, splitext
from collections import Counter
from skimage import data, color, filters, io, measure, transform, feature
import math
from scipy import stats

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

    def _calculate_intersections(self, cropped_img):
        edges = feature.canny(cropped_img, low_threshold = 0.2, high_threshold = 1)
        lines = transform.probabilistic_hough_line(edges, threshold=50, line_length=275,line_gap=10)

        set_slopes, set_lines = set(), set()
        pos_slope, neg_slope = [], []
        for line in lines:
            p0, p1 = line
            slope, intercept, _, _, _ = stats.linregress([p0[0], p1[0]], [p0[1], p1[1]])
            if True not in np.isclose(round(slope, 2), list(set_slopes), atol = 1e-02):
                set_slopes.add(round(slope, 2))
                set_lines.add(line)
                if slope > 0:
                    pos_slope.append((round(slope, 2), intercept))
                else:
                    neg_slope.append((round(slope, 2), intercept))

        coord_int = []
        for slope in pos_slope:
            coord1 = np.linalg.solve(np.array([[-slope[0], 1], [-neg_slope[0][0], 1]]), np.array([slope[1], neg_slope[0][1]]))
            coord2 = np.linalg.solve(np.array([[-slope[0], 1], [-neg_slope[1][0], 1]]), np.array([slope[1], neg_slope[1][1]]))
            coord_int.append(coord1)
            coord_int.append(coord2)

        return coord_int

    def _orient_intersection_coords(self, coord_int):
        xmin = coord_int[np.argmin(coord_int[:, 0]), :]
        xmax = coord_int[np.argmax(coord_int[:, 0]), :]
        ymin = coord_int[np.argmin(coord_int[:, 1]), :]
        ymax = coord_int[np.argmax(coord_int[:, 1]), :]

        if cropped.shape[0] < cropped.shape[1]:
            if coord_int[np.argmin(coord_int[:, 0]), :][1] > coord_int[np.argmax(coord_int[:, 0]), :][1]:
                tl, tr, bl, br = xmin, ymin, ymax, xmax
            else:
                tl, tr, bl, br = ymax, xmin, xmax, ymin
        else:
            if coord_int[np.argmin(coord_int[:, 0]), :][1] > coord_int[np.argmax(coord_int[:, 0]), :][1]:
                tl, tr, bl, br = ymin, xmax, xmin, ymax
            else:
                tl, tr, bl, br = xmin, ymin, ymax, xmax

        dst = np.array([tl, bl, br, tr])
        return dst

    # ------- NOTE: all public methods below this line --------

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

    def rotate_images(self, images):
        warped_images = []
        for img in images:
            intersect_coords = self._calculate_intersections(img)
            dst = self._orient_intersection_coords(intersect_coords)
            src = np.array([[0, 0], [0, 1000], [500, 1000], [500, 0]])
            persp_transform = transform.ProjectiveTransform()
            persp_transform.estimate(src, dst)
            warped = transform.warp(cropped, persp_transform, output_shape = (1000, 500))
            warped_images.append(warped)

        return warped_images

    def vectorize_images(self, images):
        pass





if __name__ == "__main__":
    card_process = CardImageProcessing()
    raw_imgs, grey_imgs = card_process.file_info('/Users/npng/galvanize/Dream_3_repository/card_images')
    c_type, c_suit = card_process.generate_labels(delimiter = '_')
    cropped_imgs = card_process.bounding_box_crop(grey_imgs)
    warped_imgs = card_process.rotate_images(cropped_imgs)
    io.imshow(cropped_imgs[0])
    len(cropped_imgs)
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
