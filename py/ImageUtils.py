from datetime import datetime
from PIL import Image
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

def display_images(images: list, titles: list, colorbar=False, size=8, **imshowargs):
    """Displays the given images next to each other.

    Args:
        images (list of np.ndarray): list of image arrays
        titles (list of str): list of titles
        colorbar (bool, optional): Display colorbars. Defaults to False.
        size (int, optional): plt size per image. Defaults to 8.
    """
    numImgs = len(images)
    plt.figure(figsize=(numImgs * size, size))
    for i, image, title in zip(range(numImgs), images, titles):
        plt.subplot(1, numImgs, i + 1)
        plt.imshow(image, **imshowargs)
        plt.title(title)
        if colorbar:
            plt.colorbar()
    plt.tight_layout()
    plt.show()


