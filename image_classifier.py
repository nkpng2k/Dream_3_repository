import image_processing
import skimage
import numpy as np
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB

class ImageClassifer(object):

    def __init__(self, processor):
        self.corner_classifier = None
        self.card_classifier = None
        self.suit_classifier = None
        self.processor =  processor

    def _fit_classifiers(self, X_train_cards, X_train_corners, y_train_type, y_train_suit):
        self.corner_classifier = GaussianNB()
        self.card_classifier = GaussianNB()
        self.suit_classifier = GaussianNB()
        self.corner_classifier.fit(X_train_corners, y_train_type)
        self.card_classifier.fit(X_train_cards, y_train_type)
        self.suit_classifier.fit(X_train_corners, y_train_suit)

    def predict_one(self, X_test_corner, X_test_card):
        corner_predict = self.corner_classifier.predict([X_test_corner])
        card_predict = self.card_classifier.predict([X_test_card])
        suit_predict = self.suit_classifier.predict([X_test_corner])

        return corner_predict, card_predict, suit_predict

    def fit(self, filepath):
        raw_imgs, grey_imgs = self.processor.file_info(filepath)
        c_type, c_suit = self.processor.generate_labels(delimiter = '_')
        cropped_imgs = self.processor.bounding_box_crop(grey_imgs)
        warped_imgs, tl_corner = self.processor.training_images(cropped_imgs)
        vectorized_imgs, hog_imgs = self.processor.vectorize_images(warped_imgs)
        vectorized_corner, hog_corner = self.processor.vectorize_images(tl_corner)
        self._fit_classifiers(vectorized_imgs, vectorized_corner, c_type, c_suit)



if __name__ == '__main__':
    training_filepath = '/Users/npng/galvanize/Dream_3_repository/card_images'
    card_process = image_processing.CardImageProcessing()
    card_classifier = ImageClassifer(card_process)
    card_classifier.fit(training_filepath)
    test_filepath = '/Users/npng/galvanize/Dream_3_repository/samples'
    raw_imgs, grey_imgs = card_classifier.processor.file_info(test_filepath)
    cropped_imgs = card_classifier.processor.bounding_box_crop(grey_imgs)
    warped_imgs, tl_corner = card_classifier.processor.rotate_images(cropped_imgs)
    vectorized_imgs, hog_imgs = card_classifier.processor.vectorize_images(warped_imgs)
    vectorized_corners, hog_corner = card_classifier.processor.vectorize_images(tl_corner)

    for i in xrange(len(vectorized_imgs)):
        vect_card = vectorized_imgs[i]
        vect_corner = vectorized_corners[i]
        corner_predict, card_predict, suit_predict = card_classifier.predict_one(vect_corner, vect_card)
        print corner_predict, card_predict, suit_predict








"""
Bottom of Page
"""
