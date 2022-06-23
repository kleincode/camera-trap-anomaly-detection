# Functions related to approach 3 (local features).
# For training and evaluation scripts, see ./train_bow.py and ./eval_bow.py.

import cv2 as cv
import numpy as np
from tqdm import tqdm

from py.Session import SessionImage

def dense_keypoints(img, step=30, off=(15, 12)):
    """Generates a list of densely sampled keypoints on img.

    Args:
        img (_type_): Image to sample from. (only the shape is relevant)
        step (int, optional): Vertical and horizontal step size between and size of keypoints. Defaults to 30.
        off (tuple, optional): y and x offset of the first keypoint in the grid. Defaults to (15, 12).

    Returns:
        list[cv.KeyPoint]: List of keypoints
    """
    border_dist = (step + 1) // 2
    return [cv.KeyPoint(x, y, step) for y in range(border_dist + off[0], img.shape[0] - border_dist, step) 
                                    for x in range(border_dist + off[1], img.shape[1] - border_dist, step)]


def extract_descriptors(images: list[SessionImage]):
    """Extracts DSIFT descriptors from the provided images and returns them in a single array.

    Args:
        images (list[SessionImage]): List of images to read and compute descriptors from.

    Returns:
        np.array, shape=(len(images)*keypoints_per_image, 128): DSIFT descriptors.
    """
    sift = cv.SIFT_create()
    dscs = []
    for image in tqdm(images):
        img = image.read_opencv(gray=True)
        kp = dense_keypoints(img)
        kp, des = sift.compute(img, kp)
        dscs.append(des)
    return np.array(dscs)

def generate_dictionary_from_descriptors(dscs, dictionary_size: int):
    """Clusters the given (D)SIFT descriptors using k-means.
    This may take a while depending on the number of descriptors.

    Args:
        dscs (np.array, shape(-1, 128)): (D)SIFT descriptors for clustering.
        dictionary_size (int): Number of k-means clusters.

    Returns:
        np.array, shape=(dictionary_size, 128): BOW dictionary.
    """
    BOW = cv.BOWKMeansTrainer(dictionary_size)
    for dsc in dscs:
        BOW.add(dsc)
    dictionary = BOW.cluster()
    return dictionary

def generate_bow_features(images: list[SessionImage], dictionary):
    """Calculates the BOW features for the provided images using dictionary.
    Yields a feature vector for every image.

    Args:
        images (list[SessionImage]): List of images to read and compute feature vectors from.
        dictionary (np.array, shape=(-1, 128)): BOW dictionary.

    Yields:
        (str, np.array of shape=(dictionary.shape[0])): (filename, feature vector)
    """
    flann = cv.FlannBasedMatcher({"algorithm": 0, "trees": 5}, {"checks": 50})
    sift = cv.SIFT_create()
    bow_extractor = cv.BOWImgDescriptorExtractor(sift, flann) # or cv.BFMatcher(cv.NORM_L2)
    bow_extractor.setVocabulary(dictionary)
    
    for image in tqdm(images):
        img = image.read_opencv(gray=True)
        kp = dense_keypoints(img)
        feat = bow_extractor.compute(img, kp)
        yield image.filename, feat