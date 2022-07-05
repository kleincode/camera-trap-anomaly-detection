# Quick labeling script.
# The user is displayed every image and can then assign an image as "normal" (1-key) or "anomalous" (2-key).
# The list of all normal and anomalous images will be printed after every image to be copied to Labels.py.

import cv2
import argparse
import os

from py.Dataset import Dataset
from py.FileUtils import list_jpegs_recursive

def main():
    parser = argparse.ArgumentParser(description="BOW train script")
    parser.add_argument("dataset_dir", type=str, help="Directory of the dataset containing all session folders")
    parser.add_argument("session_name", type=str, help="Name of the session to use for Lapse images (e.g. marten_01)")
    parser.add_argument("--skip", type=int, help="Skip first n images", default=0)

    args = parser.parse_args()

    ds = Dataset(args.dataset_dir)
    session = ds.create_session(args.session_name)
    
    skip = args.skip
    if skip > 0:
        print(f"Skipping the first {skip} images...")
    normal = []
    anomalous = []
    motion_folder = session.get_motion_folder()
    quit = False
    # print(list_jpegs_recursive(motion_folder), motion_folder)
    for img_file in sorted(list_jpegs_recursive(motion_folder)):
        if skip > 0:
            skip -= 1
            continue
        img_nr = int(img_file[-9:-4])
        print(f"Labeling img #{img_nr} ({img_file})... ", end="")
        image = cv2.imread(os.path.join(motion_folder, img_file))
        cv2.imshow("labeler", image)

        # wait for user to press label or exit key
        while True:
            key = cv2.waitKey(0)
            if key == ord("1"):
                print("normal")
                normal.append(img_nr)
            elif key == ord("2"):
                print("anomalous")
                anomalous.append(img_nr)
            elif key == ord("x"):
                quit = True
            else:
                continue
            print(f"normal = {normal}")
            print(f"anomalous = {anomalous}")
            break
        if quit:
            break
    cv2.destroyAllWindows()
    print("Done.")
    print(f"normal = {normal}")
    print(f"anomalous = {anomalous}")

if __name__ == "__main__":
    main()