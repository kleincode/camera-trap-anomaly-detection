# Approach 3: Local features
# This script is used for generating a BOW vocabulary using
# densely sampeled SIFT features on Lapse images.
# See eval_bow.py for evaluation.

import argparse
import os
import numpy as np

from py.Dataset import Dataset
from py.LocalFeatures import extract_descriptors, generate_dictionary_from_descriptors, generate_bow_features

def main():
    parser = argparse.ArgumentParser(description="BOW train script")
    parser.add_argument("dataset_dir", type=str, help="Directory of the dataset containing all session folders")
    parser.add_argument("session_name", type=str, help="Name of the session to use for Lapse images (e.g. marten_01)")
    parser.add_argument("--clusters", type=int, help="Number of clusters / BOW vocabulary size", default=1024)
    parser.add_argument("--step_size", type=int, help="DSIFT keypoint step size. Smaller step size = more keypoints.", default=30)

    args = parser.parse_args()

    ds = Dataset(args.dataset_dir)
    session = ds.create_session(args.session_name)
    save_dir = f"./bow_train_NoBackup/{session.name}"

    # Lapse DSIFT descriptors

    lapse_dscs_file = os.path.join(save_dir, f"lapse_dscs_{args.step_size}.npy")
    dictionary_file = os.path.join(save_dir, f"bow_dict_{args.step_size}_{args.clusters}.npy")
    train_feat_file = os.path.join(save_dir, f"bow_train_{args.step_size}_{args.clusters}.npy")

    if os.path.isfile(lapse_dscs_file):
        if os.path.isfile(dictionary_file):
            # if dictionary file already exists, we don't need the lapse descriptors
            print(f"{lapse_dscs_file} already exists, skipping lapse descriptor extraction...")
        else:
            print(f"{lapse_dscs_file} already exists, loading lapse descriptor from file...")
            lapse_dscs = np.load(lapse_dscs_file)
    else:
        # Step 1 - extract dense SIFT descriptors
        print("Extracting lapse descriptors...")
        lapse_dscs = extract_descriptors(list(session.generate_lapse_images()), kp_step=args.step_size)
        os.makedirs(save_dir, exist_ok=True)
        np.save(lapse_dscs_file, lapse_dscs)

    # BOW dictionary

    if os.path.isfile(dictionary_file):
        print(f"{dictionary_file} already exists, loading BOW dictionary from file...")
        dictionary = np.load(dictionary_file)
    else:
        # Step 2 - create BOW dictionary from Lapse SIFT descriptors
        print(f"Creating BOW vocabulary with {args.clusters} clusters...")
        dictionary = generate_dictionary_from_descriptors(lapse_dscs, args.clusters)
        np.save(dictionary_file, dictionary)
    
    # Extract Lapse BOW features using vocabulary (train data)

    if os.path.isfile(train_feat_file):
        print(f"{train_feat_file} already exists, skipping lapse BOW feature extraction...")
    else:
        # Step 3 - calculate training data (BOW features of Lapse images)
        print(f"Extracting BOW features from Lapse images...")
        features = [feat for _, feat in generate_bow_features(list(session.generate_lapse_images()), dictionary, kp_step=args.step_size)]
        np.save(train_feat_file, features)
    
    print("Complete!")

if __name__ == "__main__":
    main()