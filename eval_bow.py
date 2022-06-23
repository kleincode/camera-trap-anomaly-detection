import argparse
import os
import numpy as np
from sklearn import svm

from py.Dataset import Dataset
from py.LocalFeatures import generate_bow_features

def main():
    parser = argparse.ArgumentParser(description="BOW train script")
    parser.add_argument("dataset_dir", type=str, help="Directory of the dataset containing all session folders")
    parser.add_argument("session_name", type=str, help="Name of the session to use for Lapse images (e.g. marten_01)")
    parser.add_argument("--clusters", type=int, help="Number of clusters / BOW vocabulary size", default=1024)

    args = parser.parse_args()

    ds = Dataset(args.dataset_dir)
    session = ds.create_session(args.session_name)
    save_dir = f"./bow_train_NoBackup/{session.name}"

    # Lapse DSIFT descriptors

    dictionary_file = os.path.join(save_dir, f"bow_dict_{args.clusters}.npy")
    train_feat_file = os.path.join(save_dir, f"bow_train_{args.clusters}.npy")
    eval_file = os.path.join(save_dir, f"bow_eval_{args.clusters}.csv")

    if not os.path.isfile(dictionary_file):
        print(f"ERROR: BOW dictionary missing! ({dictionary_file})")
    elif not os.path.isfile(train_feat_file):
        print(f"ERROR: Train data file missing! ({train_feat_file})")
    elif os.path.isfile(eval_file):
        print(f"ERROR: Eval file already exists! ({eval_file})")
    else:
        print(f"Loading dictionary from {dictionary_file}...")
        dictionary = np.load(dictionary_file)
        print(f"Loading training data from {train_feat_file}...")
        train_data = np.load(train_feat_file).squeeze()
        
        print(f"Fitting one-class SVM...")
        clf = svm.OneClassSVM().fit(train_data)

        print("Evaluating...")
        with open(eval_file, "a+") as f:
            for filename, feat in generate_bow_features(list(session.generate_motion_images()), dictionary):
                y = clf.decision_function(feat)[0]
                f.write(f"{filename},{y}\n")
                f.flush()

        print("Complete!")

if __name__ == "__main__":
    main()