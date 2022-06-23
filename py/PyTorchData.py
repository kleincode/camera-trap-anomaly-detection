import os
import matplotlib.pyplot as plt
from torchvision import io, transforms
from torch.utils.data import DataLoader, Dataset

class ImageDataset(Dataset):
    def __init__(self, img_dir: str, transform = None, labeler = None, filter = lambda filename: True):
        self.img_dir = img_dir
        self.transform = transform
        self.labeler = labeler
        with os.scandir(img_dir) as it:
            self.files = [entry.name for entry in it if entry.name.endswith(".jpg") and entry.is_file() and filter(entry.name)]
        print(f"{len(self.files)} files found")
    
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.files[idx])
        img = io.read_image(img_path)
        if self.transform:
            img = self.transform(img)
        label = 0
        if self.labeler:
            label = self.labeler(self.files[idx])
        return img, label

def create_dataloader(img_folder: str, target_size: tuple = (256, 256), batch_size: int = 32, shuffle: bool = True, truncate_y: tuple = (40, 40), labeler = None, skip_transforms: bool = False, filter = lambda filename: True) -> DataLoader:
    """Creates a PyTorch DataLoader from the given image folder.

    Args:
        img_folder (str): Folder containing images. (All subfolders will be scanned for jpg images)
        target_size (tuple, optional): Model input size. Images are resized to this size. Defaults to (256, 256).
        batch_size (int, optional): Batch size. Defaults to 32.
        shuffle (bool, optional): Shuffle images. Good for training, useless for testing. Defaults to True.
        truncate_y (tuple, optional): (a, b), cut off the first a and the last b pixel rows of the unresized image. Defaults to (40, 40).
        labeler (lambda(filename: str) -> int, optional): Lambda that maps every filename to an int label. By default all labels are 0. Defaults to None.
        skip_transforms (bool, optional): Skip truncate and resize transforms. (If the images are already truncated and resized). Defaults to False.
        filter (lambda: str -> bool, optional): Additional filter by filename. Defaults to lambda filename: True.

    Returns:
        DataLoader: PyTorch DataLoader
    """
    def crop_lambda(img):
        return transforms.functional.crop(img, truncate_y[0], 0, img.shape[-2] - truncate_y[0] - truncate_y[1], img.shape[-1])

    transform = None
    if skip_transforms:
        transform = transforms.Compose([
            transforms.Lambda(lambda img: img.float()),
            transforms.Normalize((127.5), (127.5)) # min-max normalization to [-1, 1]
        ])
    else:
        transform = transforms.Compose([
            transforms.Lambda(crop_lambda),
            transforms.ToPILImage(),
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5)) # min-max normalization to [-1, 1]
        ])

    data = ImageDataset(img_folder, transform=transform, labeler=labeler, filter=filter)
    return DataLoader(data, batch_size=batch_size, shuffle=shuffle)

def model_output_to_image(y):
    y = 0.5 * (y + 1) # normalize back to [0, 1]
    y = y.clamp(0, 1) # clamp to [0, 1]
    y = y.view(y.size(0), 3, 256, 256)
    return y

def get_log(name: str, display: bool = False, figsize: tuple = (12, 6)):
    its = []
    losses = []
    with open(f"./ae_train_NoBackup/{name}/log.csv", "r") as f:
        for line in f:
            it, loss = line.rstrip().split(",")
            its.append(int(it))
            losses.append(float(loss))
    if display:
        plt.figure(figsize=figsize)
        plt.plot(its, losses)
        plt.title(f"Training curve ({name})")
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.show()
    return its, losses