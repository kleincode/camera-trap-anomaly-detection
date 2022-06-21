import argparse
import os
import cv2 as cv
import numpy as np
from tqdm import tqdm

from py.Dataset import Dataset
from py.Session import SessionImage

def dense_keypoints(img, step=30, off=(15, 12)):
    border_dist = (step + 1) // 2
    return [cv.KeyPoint(x, y, step) for y in range(border_dist + off[0], img.shape[0] - border_dist, step) 
                                    for x in range(border_dist + off[1], img.shape[1] - border_dist, step)]

def extract_descriptors(images: list[SessionImage]):
    sift = cv.SIFT_create()
    dscs = []
    for image in tqdm(images):
        img = image.read_opencv(gray=True)
        kp = dense_keypoints(img)
        kp, des = sift.compute(img, kp)
        dscs.append(des)
    return np.array(dscs)

def generate_dictionary(dscs, dictionary_size):
    # dictionary size = number of clusters
    BOW = cv.BOWKMeansTrainer(dictionary_size)
    for dsc in dscs:
        BOW.add(dsc)
    dictionary = BOW.cluster()
    return dictionary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BOW train script")
    parser.add_argument("dataset_dir", type=str, help="Directory of the dataset containing all session folders")
    parser.add_argument("session_name", type=str, help="Name of the session to use for Lapse images (e.g. marten_01)")
    parser.add_argument("--clusters", type=int, help="Number of clusters / BOW vocabulary size", default=1024)

    args = parser.parse_args()

    ds = Dataset(args.dataset_dir)
    session = ds.create_session(args.session_name)
    save_dir = f"./bow_train_NoBackup/{session.name}"

    # Lapse DSIFT descriptors

    lapse_dscs_file = os.path.join(save_dir, "lapse_dscs.npy")
    if os.path.isfile(lapse_dscs_file):
        print(f"{lapse_dscs_file} already exists, loading lapse descriptor from file...")
        lapse_dscs = np.load(lapse_dscs_file)
    else:
        print("Extracting lapse descriptors...")
        lapse_dscs = extract_descriptors(list(session.generate_lapse_images()))
        os.makedirs(save_dir, exist_ok=True)
        np.save(lapse_dscs_file, lapse_dscs)

    # BOW dictionary

    dictionary_file = os.path.join(save_dir, f"bow_dict_{args.clusters}.npy")
    if os.path.isfile(dictionary_file):
        print(f"{dictionary_file} already exists, loading BOW dictionary from file...")
        dictionary = np.load(dictionary_file)
    else:
        print(f"Creating BOW vocabulary with {args.clusters} clusters...")
        dictionary = generate_dictionary(lapse_dscs, args.clusters)
        np.save(dictionary_file, dictionary)
    
    print("Complete!")