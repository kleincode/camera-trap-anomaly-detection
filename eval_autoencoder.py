# Approach 4: Autoencoder
# This script is used for evaluating an autoencoder on Motion and Lapse images.
# See train_autoencoder.py for training.

import argparse
import os
from glob import glob
from tqdm import tqdm
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from py.FileUtils import dump
from py.Dataset import Dataset
from py.PyTorchData import create_dataloader
from py.Autoencoder2 import Autoencoder
from py.Labels import LABELS

TRAIN_FOLDER = "./ae_train_NoBackup"

def load_autoencoder(train_name: str, device: str = "cpu", model_number: int = -1, latent_features: int = 32):
    if model_number < 0:
        model_path = sorted(glob(f"./ae_train_NoBackup/{train_name}/model_*.pth"))[-1]
    else:
        model_path = f"./ae_train_NoBackup/{train_name}/model_{model_number:03d}.pth"
    print(f"Loading model from {model_path}... ", end="")
    model = Autoencoder(latent_features=latent_features)
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    model.eval()
    print("Loaded!")
    return model

def eval_autoencoder(model: Autoencoder, data_loader: DataLoader, device: str = "cpu", include_images: bool = False):
    losses = [] # reconstruction errors
    encodings = [] # latent representations for KDE
    labels = []
    imgs = [] # input images (optional)

    with torch.no_grad():
        model = model.to(device)
        criterion = nn.MSELoss()

        for features, batch_labels in tqdm(data_loader):
            features = Variable(features).to(device)
            labels += batch_labels

            # forward
            encoded = model.encoder(features)
            output_batch = model.decoder(encoded)

            # Calculate and save encoded representation and loss
            encoded_flat = encoded.detach().cpu().numpy().reshape(encoded.size()[0], -1)
            for input, enc, output in zip(features, encoded_flat, output_batch):
                encodings.append(enc)
                losses.append(criterion(input, output).cpu().numpy())
                if include_images:
                    imgs.append(input.cpu().numpy())
    return np.array(losses), np.array(encodings), np.array(labels), np.array(imgs)

def main():
    parser = argparse.ArgumentParser(description="Autoencoder eval script - evaluates Motion and Lapse images of session")
    parser.add_argument("name", type=str, help="Name of the training session (name of the save folder)")
    parser.add_argument("dataset_folder", type=str, help="Path to dataset folder containing sessions")
    parser.add_argument("session", type=str, help="Session name")
    parser.add_argument("--device", type=str, help="PyTorch device to train on (cpu or cuda)", default="cpu")
    parser.add_argument("--batch_size", type=int, help="Batch size (>=1)", default=32)
    parser.add_argument("--latent", type=int, help="Number of latent features", default=512)
    parser.add_argument("--model_number", type=int, help="Load model save of specific epoch (default: use latest)", default=-1)
    parser.add_argument("--image_transforms", action="store_true", help="Truncate and resize images (only enable if the input images have not been truncated resized to the target size already)")
    parser.add_argument("--include_images", action="store_true", help="Include input images in Motion eval file")
    

    args = parser.parse_args()

    if args.image_transforms:
        print("Image transforms enabled: Images will be truncated and resized.")
    else:
        print("Image transforms disabled: Images are expected to be of the right size.")

    ds = Dataset(args.dataset_folder)
    session = ds.create_session(args.session)
    
    # Target file names
    train_dir = os.path.join(TRAIN_FOLDER, args.name)
    save_dir = os.path.join(train_dir, "eval")
    os.makedirs(save_dir, exist_ok=True)
    suffix = "_withimgs" if args.include_images else ""
    lapse_eval_file = os.path.join(save_dir, f"{session.name}_lapse.pickle")
    motion_eval_file = os.path.join(save_dir, f"{session.name}_motion{suffix}.pickle")

    # Load model
    model = load_autoencoder(args.name, args.device, args.model_number, latent_features=args.latent)
    
    # Check CUDA
    print("Is CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available() and args.device != "cuda":
        print("WARNING: CUDA is available but not activated! Use '--device cuda'.")
    print(f"Devices: ({torch.cuda.device_count()})")
    for i in range(torch.cuda.device_count()):
        print(torch.cuda.get_device_name(i))


    # Lapse eval
    if os.path.isfile(lapse_eval_file):
        print(f"Eval file for Lapse already exists ({lapse_eval_file}) Skipping Lapse evaluation...")
    else:
        print("Creating lapse data loader... ", end="")
        lapse_loader = create_dataloader(session.get_lapse_folder(), batch_size=args.batch_size, skip_transforms=not args.image_transforms, shuffle=False)
        results = eval_autoencoder(model, lapse_loader, args.device)
        dump(lapse_eval_file, results)
        print(f"Results saved to {lapse_eval_file}!")
    

    # Motion eval

    def is_labeled(filename: str) -> bool:
        img_nr = int(filename[-9:-4])
        return (img_nr <= LABELS[session.name]["max"]) and (img_nr not in LABELS[session.name]["not_annotated"])

    def labeler(filename: str) -> int:
        is_normal = (int(filename[-9:-4]) in LABELS[session.name]["normal"])
        return 0 if is_normal else 1
    
    if os.path.isfile(motion_eval_file):
        print(f"Eval file for Motion already exists ({motion_eval_file}) Skipping Motion evaluation...")
    else:
        print("Creating motion data loader... ", end="")
        motion_loader = create_dataloader(session.get_motion_folder(), batch_size=args.batch_size, skip_transforms=not args.image_transforms, shuffle=False, labeler=labeler, filter=is_labeled)
        results = eval_autoencoder(model, motion_loader, args.device, include_images=args.include_images)
        dump(motion_eval_file, results)
        print(f"Results saved to {motion_eval_file}!")
    print("Done.")

if __name__ == "__main__":
    main()
