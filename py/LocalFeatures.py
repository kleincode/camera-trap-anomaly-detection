# Functions related to approach 3 (local features).
# For training and evaluation scripts, see ./train_bow.py and ./eval_bow.py.

import cv2 as cv
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans

from py.Session import SessionImage

def dense_keypoints(img, step=30, size=60):
    """Generates a list of densely sampled keypoints on img. The keypoints are arranged tightly
    next to each other without spacing. The group of all keypoints is centered in the image.

    Args:
        img (_type_): Image to sample from. (only the shape is relevant)
        step (int, optional): Vertical and horizontal step size between keypoints. Defaults to 30.
        size (int, optional): Size of keypoints. Defaults to 60.

    Returns:
        list[cv.KeyPoint]: List of keypoints
    """
    # calculate offset to center keypoints
    off = ((img.shape[0] % step) // 2, (img.shape[1] % step) // 2)
    border_dist = (step + 1) // 2
    return [cv.KeyPoint(x, y, size) for y in range(border_dist + off[0], img.shape[0] - border_dist, step) 
                                    for x in range(border_dist + off[1], img.shape[1] - border_dist, step)]


def extract_descriptors(images: list[SessionImage], kp_step: int = 30, kp_size: int = 60):
    """Extracts DSIFT descriptors from the provided images and returns them in a single array.

    Args:
        images (list[SessionImage]): List of images to read and compute descriptors from.
        kp_step (int, optional): Keypoint step size, see dense_keypoints. Defaults to 30.
        kp_size (int, optional): Keypoint size, see dense_keypoints. Defaults to 60.

    Returns:
        np.array, shape=(len(images)*keypoints_per_image, 128): DSIFT descriptors.
    """
    sift = cv.SIFT_create()
    dscs = []
    output_kp = False
    for image in tqdm(images):
        img = image.read_opencv(gray=True)
        kp = dense_keypoints(img, kp_step, kp_size)
        # output number of keypoints once
        if not output_kp:
            print(f"{len(kp)} keypoints per image.")
            output_kp = True
        kp, des = sift.compute(img, kp)
        dscs.extend(des)
    return np.array(dscs).reshape(-1, 128)

def generate_dictionary_from_descriptors(dscs, dictionary_size: int):
    """Clusters the given (D)SIFT descriptors using k-means.
    This may take a while depending on the number of descriptors.

    Args:
        dscs (np.array, shape(-1, 128)): (D)SIFT descriptors for clustering.
        dictionary_size (int): Number of k-means clusters.

    Returns:
        np.array, shape=(dictionary_size, 128): BOW dictionary.
    """
    assert len(dscs.shape) == 2 and dscs.shape[1] == 128
    assert dictionary_size > 0 and dictionary_size <= dscs.shape[0]

    kmeans = KMeans(dictionary_size, verbose=1).fit(dscs)
    dictionary = kmeans.cluster_centers_
    assert dictionary.shape == (dictionary_size, 128)
    return dictionary

def pick_random_descriptors(dscs, dictionary_size: int):
    """Picks dictionary_size random descriptors to use as a vocabulary.
    Much faster but less accurate alternative to kmeans clustering.

    Args:
        dscs (np.array, shape(-1, 128)): (D)SIFT descriptors to pick from.
        dictionary_size (int): Number of clusters / vocabulary size.

    Returns:
        np.array, shape=(dictionary_size, 128): Randomly picked BOW dictionary.
    """
    assert len(dscs.shape) == 2 and dscs.shape[1] == 128
    assert dictionary_size > 0 and dictionary_size <= dscs.shape[0]

    return dscs[np.random.choice(len(dscs), size=dictionary_size, replace=False)]

def generate_bow_features(images: list[SessionImage], dictionary, kp_step: int = 30, kp_size: int = 60):
    """Calculates the BOW features for the provided images using dictionary.
    Yields a feature vector for every image.

    Args:
        images (list[SessionImage]): List of images to read and compute feature vectors from.
        dictionary (np.array, shape=(-1, 128)): BOW dictionary.
        kp_step (int, optional): Keypoint step size, see dense_keypoints. Must be identical to the step size used for vocabulary generation. Defaults to 30.
        kp_size (int, optional): Keypoint size, see dense_keypoints. Must be identical to the size used for vocabulary generation. Defaults to 60.

    Yields:
        (str, np.array of shape=(dictionary.shape[0])): (filename, feature vector)
    """
    assert len(dictionary.shape) == 2 and dictionary.shape[1] == 128
    assert kp_size > 0 and kp_step > 0

    flann = cv.FlannBasedMatcher({"algorithm": 0, "trees": 5}, {"checks": 50})
    sift = cv.SIFT_create()
    bow_extractor = cv.BOWImgDescriptorExtractor(sift, flann) # or cv.BFMatcher(cv.NORM_L2)
    bow_extractor.setVocabulary(dictionary)
    
    for image in tqdm(images):
        img = image.read_opencv(gray=True)
        kp = dense_keypoints(img, kp_step, kp_size)
        feat = bow_extractor.compute(img, kp)
        yield image.filename, feat