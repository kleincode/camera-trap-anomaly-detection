# Approach 3: Local features
# This script is used for generating a BOW vocabulary using
# densely sampeled SIFT features on Lapse images.
# See eval_bow.py for evaluation.

import argparse
import os
import numpy as np
from timeit import default_timer as timer
from datetime import timedelta

from py.Dataset import Dataset
from py.LocalFeatures import extract_descriptors, generate_dictionary_from_descriptors, generate_bow_features, pick_random_descriptors

def main():
    parser = argparse.ArgumentParser(description="BOW train script")
    parser.add_argument("dataset_dir", type=str, help="Directory of the dataset containing all session folders")
    parser.add_argument("session_name", type=str, help="Name of the session to use for Lapse images (e.g. marten_01)")
    parser.add_argument("--clusters", type=int, help="Number of clusters / BOW vocabulary size", default=1024)
    parser.add_argument("--step_size", type=int, help="DSIFT keypoint step size. Smaller step size = more keypoints.", default=30)
    parser.add_argument("--keypoint_size", type=int, help="DSIFT keypoint size. Defaults to step_size.", default=-1)
    parser.add_argument("--include_motion", action="store_true", help="Include motion images for training.")
    parser.add_argument("--random_prototypes", action="store_true", help="Pick random prototype vectors instead of doing kmeans.")
    parser.add_argument("--num_vocabularies", type=int, help="Number of vocabularies to generate if random prototype choosing is enabled.", default=10)

    args = parser.parse_args()
    if args.keypoint_size <= 0:
        args.keypoint_size = args.step_size
    print(f"Using keypoint size {args.keypoint_size} with step size {args.step_size}.")

    ds = Dataset(args.dataset_dir)
    session = ds.create_session(args.session_name)
    save_dir = f"./bow_train_NoBackup/{session.name}"

    suffix = ""
    if args.include_motion:
        suffix += "_motion"
        print("Including motion data for prototype selection!")
    if args.random_prototypes:
        suffix += "_random"
        print("Picking random prototypes instead of using kmeans!")
    lapse_dscs_file = os.path.join(save_dir, f"lapse_dscs_{args.step_size}_{args.keypoint_size}.npy")
    motion_dscs_file = os.path.join(save_dir, f"motion_dscs_{args.step_size}_{args.keypoint_size}.npy")
    dictionary_file = os.path.join(save_dir, f"bow_dict_{args.step_size}_{args.keypoint_size}_{args.clusters}{suffix}.npy")
    train_feat_file = os.path.join(save_dir, f"bow_train_{args.step_size}_{args.keypoint_size}_{args.clusters}{suffix}.npy")

    # Lapse DSIFT descriptors

    if os.path.isfile(lapse_dscs_file):
        if os.path.isfile(dictionary_file):
            # if dictionary file already exists, we don't need the lapse descriptors
            print(f"{dictionary_file} already exists, skipping lapse descriptor extraction...")
        else:
            print(f"{lapse_dscs_file} already exists, loading lapse descriptors from file... ", end="")
            lapse_dscs = np.load(lapse_dscs_file)
            assert lapse_dscs.shape[-1] == 128
            lapse_dscs = lapse_dscs.reshape(-1, 128)
            print(f"Loaded {len(lapse_dscs)} lapse descriptors!")
    else:
        # Step 1 - extract dense SIFT descriptors
        print("Extracting lapse descriptors...")
        lapse_dscs = extract_descriptors(list(session.generate_lapse_images()), kp_step=args.step_size, kp_size=args.keypoint_size)
        os.makedirs(save_dir, exist_ok=True)
        np.save(lapse_dscs_file, lapse_dscs)
    
    # Motion DSIFT descriptors
    if args.include_motion:
        if os.path.isfile(motion_dscs_file):
            if os.path.isfile(dictionary_file):
                # if dictionary file already exists, we don't need the descriptors
                print(f"{dictionary_file} already exists, skipping motion descriptor extraction...")
            else:
                print(f"{motion_dscs_file} already exists, loading motion descriptors from file...", end="")
                motion_dscs = np.load(motion_dscs_file)
                assert motion_dscs.shape[-1] == 128
                motion_dscs = motion_dscs.reshape(-1, 128)
                print(f"Loaded {len(motion_dscs)} motion descriptors!")
                lapse_dscs = np.concatenate([lapse_dscs, motion_dscs])
        else:
            # Step 1b - extract dense SIFT descriptors from motion images
            print("Extracting motion descriptors...")
            motion_dscs = extract_descriptors(list(session.generate_motion_images()), kp_step=args.step_size, kp_size=args.keypoint_size)
            os.makedirs(save_dir, exist_ok=True)
            np.save(motion_dscs_file, motion_dscs)
            lapse_dscs = np.concatenate([lapse_dscs, motion_dscs])

    # BOW dictionary

    if os.path.isfile(dictionary_file):
        print(f"{dictionary_file} already exists, loading BOW dictionary from file...")
        dictionaries = np.load(dictionary_file)
    else:
        # Step 2 - create BOW dictionary from Lapse SIFT descriptors
        print(f"Creating BOW vocabulary with {args.clusters} clusters from {len(lapse_dscs)} descriptors...")
        start_time = timer()
        if args.random_prototypes:
            dictionaries = np.array([pick_random_descriptors(lapse_dscs, args.clusters) for i in range(args.num_vocabularies)])
        else:
            dictionaries = np.array([generate_dictionary_from_descriptors(lapse_dscs, args.clusters)])
        end_time = timer()
        delta_time = timedelta(seconds=end_time-start_time)
        print(f"Clustering took {delta_time}.")
        np.save(dictionary_file, dictionaries)
    
    # Extract Lapse BOW features using vocabulary (train data)

    if os.path.isfile(train_feat_file):
        print(f"{train_feat_file} already exists, skipping lapse BOW feature extraction...")
    else:
        # Step 3 - calculate training data (BOW features of Lapse images)
        print(f"Extracting BOW features from Lapse images...")
        features = [feat for _, feat in generate_bow_features(list(session.generate_lapse_images()), dictionaries, kp_step=args.step_size, kp_size=args.keypoint_size)]
        np.save(train_feat_file, features)
    
    print("Complete!")

if __name__ == "__main__":
    main()