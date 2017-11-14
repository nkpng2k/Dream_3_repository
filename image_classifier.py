import image_processing
import skimage
import numpy as np
from sklearn.cluster import KMeans

class ImageClassifer(object):

    def __init__(self, processor):
        self.corner_kmeans = None
        self.card_kmeans = None
        self.suit_classifier = None
        self.processor =  processor

    def _fit_classifers(self, X_train_cards, X_train_corners, y_train_type, y_train_suit):
        self.corner_kmeans = KMeans(n_clusters = 13)
        self.card_kmeans = KMeans(n_clusters = 13)
        self.suit_classifier = KMeans(n_clusters = 4)

    def predict_new(self, X_test):
        pass

    def fit(self, filepath):
        raw_imgs, grey_imgs = self.processor.file_info(filepath)
        c_type, c_suit = self.processor.generate_labels(delimiter = '_')
        cropped_imgs = self.processor.bounding_box_crop(grey_imgs)
        warped_imgs, tl_corner = self.processor.rotate_images(cropped_imgs)
        vectorized_imgs, hog_imgs = self.processor.vectorize_images(warped_imgs)
        vectorized_corner, hog_corner = self.processor.vectorize_images(tl_corner)
        self._fit_classifiers(self, vectorized_imgs, vectorized_corner, c_type, c_suit)



if __name__ == '__main__':
    training_filepath = '/Users/npng/galvanize/Dream_3_repository/card_images'
    card_process = image_processing.CardImageProcessing()
    card_classifier = ImageClassifer(card_process)
    card_classifier.fit(training_filepath)
