from datetime import datetime
from PIL import Image

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
