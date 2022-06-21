import argparse
import os
from tqdm import tqdm
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchinfo import summary

from py.PyTorchData import create_dataloader, model_output_to_image
from py.Autoencoder2 import Autoencoder

def eval_autoencoder(model: Autoencoder, dataloader: DataLoader, name: str, set_name: str, device: str = "cpu", criterion = nn.MSELoss()):
    model = model.to(device)
    print(f"Using {device} device")

    print(f"Saving evaluation results to ./ae_train_NoBackup/{name}/eval")
    os.makedirs(f"./ae_train_NoBackup/{name}/eval", exist_ok=True)

    labels = []
    encodeds = []
    losses = []


    for img, labels in tqdm(dataloader):
        img_batch = Variable(img_batch).to(device)
        # ===================forward=====================
        encoded = model.encoder(img)
        encoded_flat = encoded.detach().numpy().reshape(encoded.size()[0], -1)
        output_batch = model.decoder(encoded)

        for input, output, label, enc_flat in zip(img, output_batch, labels, encoded_flat):
            losses.append(criterion(input, output))
            encodeds.append(enc_flat)
            labels.append(label)
    np.save(f"./ae_train_NoBackup/{name}/eval/{set_name}.npy")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autoencoder eval script")
    parser.add_argument("name", type=str, help="Name of the training session (name of the save folder)")
    parser.add_argument("model_name", type=str, help="Filename of the model (e.g. model_120.pth)")
    parser.add_argument("set_name", type=str, help="Name of the dataset (e.g. train or test)")
    parser.add_argument("img_folder", type=str, help="Path to directory containing train images (may contain subfolders)")
    parser.add_argument("--device", type=str, help="PyTorch device to train on (cpu or cuda)", default="cpu")
    parser.add_argument("--batch_size", type=int, help="Batch size (>=1)", default=32)
    parser.add_argument("--image_transforms", action="store_true", help="Truncate and resize images (only enable if the input images have not been truncated resized to the target size already)")
    
    args = parser.parse_args()

    if args.image_transforms:
        print("Image transforms enabled: Images will be truncated and resized.")
    else:
        print("Image transforms disabled: Images are expected to be of the right size.")
    
    dataloader = create_dataloader(args.img_folder, batch_size=args.batch_size, skip_transforms=not args.image_transforms)
    model = Autoencoder()
    print("Model:")
    summary(model, (args.batch_size, 3, 256, 256))
    print("Is CUDA available:", torch.cuda.is_available())
    print(f"Devices: ({torch.cuda.device_count()})")
    for i in range(torch.cuda.device_count()):
        print(torch.cuda.get_device_name(i))
    if args.noise:
        print("Adding Gaussian noise to model input")
    eval_autoencoder(model, dataloader, args.model_name, args.set_name, args.device)
