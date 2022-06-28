from datetime import datetime
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def get_image_date(img_path: str) -> datetime:
    """Returns the date from the image EXIF data.

    Args:
        img_path (str): path to image

    Returns:
        datetime: datetime extracted from EXIF data
    """
    img = Image.open(img_path)
    date_raw = img.getexif()[306]
    return datetime.strptime(date_raw, "%Y:%m:%d %H:%M:%S")

def display_images(images: list, titles: list, colorbar=False, size=(8, 5), row_size=2, **imshowargs):
    """Displays the given images next to each other.

    Args:
        images (list of np.ndarray): list of image arrays
        titles (list of str): list of titles
        colorbar (bool, optional): Display colorbars. Defaults to False.
        size (tuple of ints, optional): plt size (width, height) per image. Defaults to (8, 5).
    """
    num_imgs = len(images)
    num_cols = row_size
    num_rows = (num_imgs - 1) // num_cols + 1
    plt.figure(figsize=(num_cols * size[0], num_rows * size[1]))
    for i, image, title in zip(range(num_imgs), images, titles):
        plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow(image, **imshowargs)
        plt.title(title)
        if colorbar:
            plt.colorbar()
    plt.tight_layout()
    plt.show()

def is_daytime(img, threshold=50) -> bool:
    return np.mean([abs(img[:,:,0] - img[:,:,1]), abs(img[:,:,1] - img[:,:,2]), abs(img[:,:,2] - img[:,:,0])]) > threshold
