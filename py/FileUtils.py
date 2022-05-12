from glob import glob
import os

def list_folders(path: str) -> list:
    """Returns the names of all immediate child folders of path.

    Args:
        path (str): path to search

    Returns:
        list: list of all child folder names
    """
    return [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]

def list_jpegs_recursive(path: str) -> list:
    """Recursively lists all jpeg files in path.

    Args:
        path (str): path to search

    Returns:
        list: list of all jpeg files
    """
    return [name for name in glob(os.path.join(path, "**/*.jpg"), recursive=True) if os.path.isfile(os.path.join(path, name))]